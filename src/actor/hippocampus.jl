module Hippocampus
using ..TetrisAICore
using Tetris
struct ActionData
    game_state::GameState
    node::Node
end

"１ゲーム中に集めた行動データ"
const action_data_list = Ref{Vector{ActionData}}(Vector{ActionData}())
end

function add_action_data(state::GameState, node::Node)
    push!(Hippocampus.action_data_list[], Hippocampus.ActionData(state, node))
end

function create_experience(actor::Actor, multisteps::Int, γ::Float64)::Vector{Experience}
    exp_list = Vector{Experience}()
    data_length = length(Hippocampus.action_data_list[])
    for i in 1:data_length
        # 盤面の価値を学習するターゲット
        target_data = Hippocampus.action_data_list[][i]
        # multistepで報酬を計算する
        prev_score = target_data.game_state.score
        n = min(multisteps - 1, data_length - i)
        multistep_reward = 0
        for j in 0:n
            data = Hippocampus.action_data_list[][i+j]
            multistep_reward += γ^j * (data.node.game_state.score - prev_score)
            prev_score = data.node.game_state.score
        end
        last_data = Hippocampus.action_data_list[][i+n]
        last_node = last_data.node
        next_nodelist = get_node_list(last_node.game_state)
        td_error = calc_td_error(target_data.game_state,
            target_data.node,
            last_node,
            next_nodelist,
            multistep_reward,
            n + 1,
            Config.γ,
            (x...) -> predict(actor.brain.main_model, x),
            (x...) -> predict(actor.brain.target_model, x),
        )
        exp = Experience(target_data.game_state, target_data.node, last_node, next_nodelist, multistep_reward, n + 1, td_error)
        GC.gc(false)
        push!(exp_list, exp)
    end
    Hippocampus.action_data_list[] = Vector{Hippocampus.ActionData}()
    return exp_list
end

using .TetrisAICore: vector2array, rescaling_reward, inverse_rescaling_reward
function calc_td_error(
    state::GameState,
    node::Node,
    multistep_node::Node,
    multistep_next_node_list::Vector{Node},
    multistep_reward::Float64,
    multisteps::Int,
    discount_rate::Float64,
    predicter,
    target_predicter,
)::Float64
    ϵ = 1e-6

    current_holdnext = [state.hold_mino, state.mino_list[end-4:end]...]
    current_expect_reward = predicter(
        [state.current_game_board.binary] |> vector2array .|> Float32,
        [generate_minopos(node.mino, node.position)] |> vector2array .|> Float32,
        [state.combo] |> vector2array .|> Float32,
        [state.back_to_back_flag] |> vector2array .|> Float32,
        [node.tspin] |> vector2array .|> Float32,
        reshape(hcat([mino_to_array(mino) for mino in current_holdnext]...), 7, 6, 1),
    )[1]


    if multistep_node.game_state.game_over_flag
        return abs(rescaling_reward(-1000* discount_rate^multisteps - inverse_rescaling_reward(current_expect_reward))) + ϵ
    end
    # 次の盤面の最大の価値を算出する
    max_node =
        select_node(multistep_next_node_list, multistep_node.game_state, predicter)
    node_holdnext = [multistep_node.game_state.hold_mino, multistep_node.game_state.mino_list[end-4:end]...]
    next_score = target_predicter(
        [multistep_node.game_state.current_game_board.binary] |> vector2array .|> Float32,
        [generate_minopos(max_node.mino, max_node.position)] |> vector2array .|> Float32,
        [multistep_node.game_state.combo] |> vector2array .|> Float32,
        [multistep_node.game_state.back_to_back_flag] |> vector2array .|> Float32,
        [multistep_node.tspin] |> vector2array .|> Float32,
        reshape(hcat([mino_to_array(mino) for mino in node_holdnext]...), 7, 6, 1),
    )[1]

    # 設置後に得られるスコア

    temporal_difference = rescaling_reward(multistep_reward + inverse_rescaling_reward(next_score) * discount_rate^multisteps - inverse_rescaling_reward(current_expect_reward))
    return abs(temporal_difference) + ϵ
end