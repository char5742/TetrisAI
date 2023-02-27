

"学習者"
mutable struct Learner
    brain::Brain
    "targetモデルの同期間隔"
    taget_update_cycle::Int64
    taget_update_count::Int64
    "Optimiserの状態"
    optim
end

function make_experience(
    brain::Brain,
    state::GameState,
    node::Node,
    discount_rate::Float64,
)::Experience
    ϵ = 1e-5
    current_gameboard = state.current_game_board.binary
    minopos = generate_minopos(node.mino, node.position)
    current_expect_reward = lock(brain.mainlock) do
        brain.main_model((
            (reshape(current_gameboard |> float, 24, 10, 1, 1), reshape(minopos |> float, 24, 10, 1, 1)),
            state.combo |> Float32, state.back_to_back_flag |> Float32, (node.tspin > 0) |> Float32) |> gpu) |> x -> cpu(x)[1]
    end
    if node.game_state.game_over_flag
        return Experience(current_gameboard, minopos, state.combo, state.back_to_back_flag, node.tspin, -0.5, abs(-0.5 - current_expect_reward) + ϵ)
    end
    reward = (node.game_state.score - state.score)*(node.tspin>0)
    scaled_reward = rescaling_reward(reward)
    # 次の盤面の最大の価値を算出する
    node_list = get_node_list(node.game_state)
    max_node =
        select_node(brain.main_model, brain.mainlock, node_list, node.game_state)

    max_node_minopos = generate_minopos(max_node.mino, max_node.position)
    p = lock(brain.targetlock) do
        brain.target_model((
            (reshape(node.game_state.current_game_board.binary |> float, 24, 10, 1, 1), reshape(max_node_minopos |> float, 24, 10, 1, 1)), node.game_state.combo |> Float32, node.game_state.back_to_back_flag |> Float32, (max_node.tspin > 0) |> Float32
        ) |> gpu) |> x -> cpu(x)[1]
    end
    temporal_difference = scaled_reward + p * discount_rate - current_expect_reward
    expected_reward = p * discount_rate + scaled_reward
    return Experience(current_gameboard, minopos, state.combo, state.back_to_back_flag, node.tspin, expected_reward, abs(temporal_difference) + ϵ)
end

function qlearn(learner::Learner, batch_size, exp::Vector{Experience})
    try
        # 行動前の状態
        prev_game_bord_array = Array{Float64}(undef, 24, 10, 1, batch_size)
        prev_combo_array = Array{Float64}(undef, 1, batch_size)
        prev_back_to_back_array = Array{Float64}(undef, 1, batch_size)
        # 行動前後の差分 (ミノの設置位置を示す)
        minopos_array = Array{Float64}(undef, 24, 10, 1, batch_size)
        # その行動のtspin判定
        prev_tspin_array = Array{Float64}(undef, 1, batch_size)
        # 行動により得た報酬
        expected_reward_array = Array{Float64}(undef, 1, batch_size)
        for (i, (;
            prev_game_bord,
            minopos,
            prev_combo,
            prev_back_to_back,
            prev_tspin,
            expected_reward
        )) in enumerate(exp)
            prev_game_bord_array[:, :, 1, i] = prev_game_bord
            minopos_array[:, :, 1, i] = minopos
            prev_combo_array[i] = prev_combo
            prev_back_to_back_array[i] = prev_back_to_back
            prev_tspin_array[i] = prev_tspin
            expected_reward_array[i] = expected_reward
        end

        if learner.taget_update_count % learner.taget_update_cycle == 0
            lock(learner.brain.targetlock) do
                Flux.loadmodel!(learner.brain.target_model, learner.brain.main_model)
            end
        end
        learner.taget_update_count += 1
        fit!(learner, ((prev_game_bord_array, minopos_array), prev_combo_array, prev_back_to_back_array, prev_tspin_array) |> gpu, expected_reward_array |> gpu), sum(expected_reward_array) / batch_size, sum(prev_tspin_array)
    catch
        GC.gc(true)
        0.0, 0.0, 0.0
    end
end

function fit!(learner::Learner, x, y)
    model = learner.brain.main_model
    trainingloss, (∇model,) = Flux.withgradient(model) do m
        Flux.Losses.mse(m(x), y)
    end
    # CUDA.math_mode!(CUDA.DEFAULT_MATH; precision=:Float32)
    optim, model = Optimisers.update(learner.optim, model, ∇model)
    lock(learner.brain.mainlock) do
        learner.brain.main_model = model
    end
    learner.optim = optim
    # CUDA.math_mode!(CUDA.FAST_MATH; precision=:Float16)
    trainingloss
end