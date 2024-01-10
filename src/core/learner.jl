
"学習者"
mutable struct Learner
    brain::Brain
    "targetモデルの同期間隔"
    taget_update_cycle::Int64
    taget_update_count::Int64
    "学習状態"
    optim
end

function calc_expected_rewards(
    brain::Brain,
    prev_score_list::Vector{Float64},
    node_list::Vector{Node},
    next_node_list_list::Vector{Vector{Node}},
    current_expect_reward_array::Array{Float32,2},
    discount_rate::Float64,
)::Vector{Tuple{Float64,Float64}}
    ϵ = 1e-6
    calc_size = length(node_list)

    # 次の盤面の最大の価値を算出する
    max_node_list =
        select_node_list(next_node_list_list, [node.game_state for node in node_list], (x...) -> predict(brain.main_model, x))
    # ゲームオーバーではないノードがあるとき、次の盤面の最大の価値を算出する
    if (any(!node.game_state.game_over_flag for node in node_list))
        node_currentboard_array = [node.game_state.current_game_board.binary .|> Float32 for node in node_list if !node.game_state.game_over_flag] |> vector2array
        max_node_minopos_array = [generate_minopos(n.mino, n.position) .|> Float32 for n in max_node_list if !isnothing(n)] |> vector2array
        node_ren_array = [node.game_state.ren |> Float32 for node in node_list if !node.game_state.game_over_flag] |> vector2array
        node_back_to_back_array = [node.game_state.back_to_back_flag |> Float32 for node in node_list if !node.game_state.game_over_flag] |> vector2array
        max_node_tspin_array = [min(node.tspin, 1) |> Float32 for node in max_node_list if !isnothing(node)] |> vector2array
        node_holdnext = reduce((x, y) -> cat(x, y, dims=3), reshape(hcat([mino_to_array(mino) for mino in [node.game_state.hold_mino, node.game_state.mino_list[end-4:end]...]]...), 7, 6, 1) for node in node_list if !node.game_state.game_over_flag)
        next_score_list =
            predict(brain.target_model,
                (node_currentboard_array, max_node_minopos_array, node_ren_array, node_back_to_back_array, max_node_tspin_array, node_holdnext
                ))
    end
    response = Vector{Tuple{Float64,Float64}}(undef, calc_size)
    index = 1
    for i in 1:calc_size
        current_expect_reward = current_expect_reward_array[i]
        node = node_list[i]
        prev_score = prev_score_list[i]
        if node.game_state.game_over_flag
            response[i] = (rescaling_reward(-1000), abs(rescaling_reward(-1000 - inverse_rescaling_reward(current_expect_reward))) + ϵ)
            continue
        end
        # 設置後に得られるスコア
        reward = node.game_state.score - prev_score
        p = next_score_list[index]
        index += 1
        temporal_difference = rescaling_reward(reward + inverse_rescaling_reward(p) * discount_rate - inverse_rescaling_reward(current_expect_reward))
        expected_reward = rescaling_reward(inverse_rescaling_reward(p) * discount_rate + reward)
        response[i] = (expected_reward, abs(temporal_difference) + ϵ)
    end
    response
end

function qlearn(learner::Learner, batch_size, id_and_exp::Vector{Tuple{Int,Experience}}, discount_rate; use_gpu=true)
    try
        # 行動前の状態
        prev_game_board_array = Array{Float32}(undef, 24, 10, 1, batch_size)
        prev_ren_array = Array{Float32}(undef, 1, batch_size)
        prev_back_to_back_array = Array{Float32}(undef, 1, batch_size)
        prev_holdnext_array = Array{Float32}(undef, 7, 6, batch_size)
        # 行動前後の差分 (ミノの設置位置を示す)
        minopos_array = Array{Float32}(undef, 24, 10, 1, batch_size)
        # 行動により得た報酬
        expected_reward_array = Array{Float32}(undef, 1, batch_size)
        # その行動のtspin判定
        prev_tspin_array = Array{Float32}(undef, 1, batch_size)
        # 経験の新しい価値
        new_temporal_difference_list = Vector{Tuple{Int,Float32}}(undef, batch_size)

        # 次の盤面の価値を予測する為に必要な情報
        prev_score_list = Vector{Float64}(undef, batch_size)
        next_node_list_list = Vector{Vector{Node}}(undef, batch_size)
        selected_node_list = Vector{Node}(undef, batch_size)

        for (i, (id, (;
            current_state,
            selected_node,
            temporal_difference
        ))) in collect(enumerate(id_and_exp))
            prev_game_board_array[:, :, 1, i] = current_state.current_game_board.binary
            minopos_array[:, :, 1, i] = generate_minopos(selected_node.mino, selected_node.position)
            prev_ren_array[i] = current_state.ren
            prev_back_to_back_array[i] = current_state.back_to_back_flag
            prev_tspin_array[i] = min(selected_node.tspin, 1)
            prev_holdnext_array[:, :, i] = hcat([mino_to_array(mino) for mino in [current_state.hold_mino, current_state.mino_list[end-4:end]...]]...)

            prev_score_list[i] = current_state.score
            next_node_list_list[i] = get_node_list(selected_node.game_state)
            selected_node_list[i] = selected_node
        end
        current_expect_reward_array =
            predict(learner.brain.main_model,
                (prev_game_board_array, minopos_array,
                    prev_ren_array, prev_back_to_back_array, prev_tspin_array, prev_holdnext_array))

        res = calc_expected_rewards(
            learner.brain,
            prev_score_list,
            selected_node_list,
            next_node_list_list,
            current_expect_reward_array,
            discount_rate,
        )
        for (i, (expected_reward, new_temporal_difference)) in enumerate(res)
            expected_reward_array[i] = expected_reward
            new_temporal_difference_list[i] = (id_and_exp[i][1], new_temporal_difference)
        end
        learner.taget_update_count += 1
        if learner.taget_update_count % learner.taget_update_cycle == 0
            learner.brain.target_model.ps = learner.brain.main_model.ps
            learner.brain.target_model.st = learner.brain.main_model.st
        end
        ps, st, optim, trainingloss = fit(learner.brain.main_model.model, learner.brain.main_model.ps, learner.brain.main_model.st, learner.optim, (prev_game_board_array, minopos_array, prev_ren_array, prev_back_to_back_array, prev_tspin_array, prev_holdnext_array), expected_reward_array; use_gpu=use_gpu)
        (isnan(trainingloss) || isinf(trainingloss)) && throw(throw("trainingloss is invalid"))
        learner.brain.main_model.ps = ps
        learner.brain.main_model.st = st
        learner.optim = optim
        trainingloss, sum(expected_reward_array) / batch_size, sum(prev_tspin_array), new_temporal_difference_list
    catch e
        rethrow(e)
        GC.gc()
        0.0, 0.0, 0.0, Vector{Tuple{Int,Float32}}()
    end
end
