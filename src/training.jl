

function calc_expected_reward(
    brain::Brain,
    state::GameState,
    node::Node,
    next_node_list::Vector{Node},
    discount_rate::Float64,
)::Tuple{Float64,Float64}
    ϵ = 1e-5
    current_gameboard = state.current_game_board.binary
    minopos = generate_minopos(node.mino, node.position)
    holdnext = vcat([mino_to_array(mino) for mino in [state.hold_mino, state.mino_list[end-4:end]...]]...)
    current_gameboard_array = vector2array([current_gameboard .|> Float32])
    current_minopos_array = vector2array([minopos .|> Float32])
    current_combo_array = [state.combo |> Float32;;]
    current_btb_array = [state.back_to_back_flag |> Float32;;]
    current_tspin_array = [(node.tspin > 0) |> Float32;;]
    current_holdnext_array = reshape(holdnext, 6, 7, 1)

    current_expect_reward = lock(brain.mainlock) do
        predict(brain.main_model,
            (current_gameboard_array, current_minopos_array,
                current_combo_array, current_btb_array, current_tspin_array, current_holdnext_array))[1]
    end
    if node.game_state.game_over_flag
        return -0.5, abs(-0.5 - current_expect_reward) + ϵ
    end
    reward = node.game_state.score - state.score
    scaled_reward = rescaling_reward(reward)
    # 次の盤面の最大の価値を算出する
    max_node =
        select_node(brain.main_model, brain.mainlock, next_node_list, node.game_state)

    max_node_minopos = generate_minopos(max_node.mino, max_node.position)
    max_node_holdnext = reshape(vcat([mino_to_array(mino) for mino in [node.game_state.hold_mino, node.game_state.mino_list[end-4:end]...]]...), 6, 7, 1)
    p = lock(brain.targetlock) do
        predict(brain.target_model,
            (vector2array([node.game_state.current_game_board.binary .|> Float32]), vector2array([max_node_minopos .|> Float32]), [node.game_state.combo |> Float32;;], [node.game_state.back_to_back_flag |> Float32;;], [(max_node.tspin > 0) |> Float32;;], max_node_holdnext,
            ))[1]
    end
    temporal_difference = scaled_reward + p * discount_rate - current_expect_reward
    expected_reward = p * discount_rate + scaled_reward
    return expected_reward, abs(temporal_difference) + ϵ
end

function qlearn(learner::Learner, batch_size, id_and_exp::Vector{Tuple{Int,Experience}}, discount_rate)
    try
        # 行動前の状態
        prev_game_board_array = Array{Float32}(undef, 24, 10, 1, batch_size)
        prev_combo_array = Array{Float32}(undef, 1, batch_size)
        prev_back_to_back_array = Array{Float32}(undef, 1, batch_size)
        prev_holdnext_array = Array{Float32}(undef, 6, 7, batch_size)
        # 行動前後の差分 (ミノの設置位置を示す)
        minopos_array = Array{Float32}(undef, 24, 10, 1, batch_size)
        # その行動のtspin判定
        prev_tspin_array = Array{Float32}(undef, 1, batch_size)
        # 行動により得た報酬
        expected_reward_array = Array{Float32}(undef, 1, batch_size)
        # 経験の新しい価値
        new_temporal_difference_list = Vector{Tuple{Int, Float32}}(undef,batch_size)

        for (i, (id, (;
            current_state,
            selected_node,
            next_node_list,
            temporal_difference
        ))) in enumerate(id_and_exp)
            prev_game_board_array[:, :, 1, i] = current_state.current_game_board.binary
            minopos_array[:, :, 1, i] = generate_minopos(selected_node.mino, selected_node.position)
            prev_combo_array[i] = current_state.combo
            prev_back_to_back_array[i] = current_state.back_to_back_flag
            prev_tspin_array[i] = selected_node.tspin
            prev_holdnext_array[:, :, i] =  vcat([mino_to_array(mino) for mino in [current_state.hold_mino, current_state.mino_list[end-4:end]...]]...)
            (expected_reward, new_temporal_difference) = calc_expected_reward(learner.brain, current_state, selected_node, next_node_list, discount_rate)
            expected_reward_array[i] = expected_reward
            new_temporal_difference_list[i] = (id,new_temporal_difference)
        end

        if learner.taget_update_count % learner.taget_update_cycle == 0
            lock(learner.brain.targetlock) do
                loadmodel!(learner.brain.target_model, learner.brain.main_model)
            end
        end
        learner.taget_update_count += 1
        fit!(learner, (prev_game_board_array, minopos_array, prev_combo_array, prev_back_to_back_array, prev_tspin_array, prev_holdnext_array), expected_reward_array), sum(expected_reward_array) / batch_size, sum(prev_tspin_array), new_temporal_difference_list
    catch
        GC.gc(true)
        0.0, 0.0, 0.0, Vector{Tuple{Int, Float32}}()
    end
end