struct Actor
    id::Int64
    epsilon::Float64
    brain::Brain
end

function select_node(actor::Actor, node_list::Vector{Node}, state::GameState)::Node
    if actor.epsilon > rand()
        return rand(node_list)
    else
        return select_node(node_list, state, (x...) -> predict(actor.brain.main_model, x))
    end
end
"""
node_list: 次の盤面ノードのリスト
state: 現在のゲーム状態
predicter: 盤面価値の予測関数
"""
function select_node(node_list::Vector{Node}, state::GameState, predicter::Function)::Node
    currentbord = state.current_game_board.binary
    current_combo = state.combo
    current_back_to_back = state.back_to_back_flag
    current_holdnext = [state.hold_mino, state.mino_list[end-4:end]...]
    minopos_array = [generate_minopos(n.mino, n.position) .|> Float32 for n in node_list] |> vector2array
    tspin_array = [(n.tspin > 1 ? 1 : 0) |> Float32 for n in node_list] |> vector2array
    currentbord_array = [currentbord .|> Float32 for _ in 1:length(node_list)] |> vector2array
    current_combo_array = [current_combo |> Float32 for _ in 1:length(node_list)] |> vector2array
    current_back_to_back_array = [current_back_to_back |> Float32 for _ in 1:length(node_list)] |> vector2array
    current_holdnext_array = repeat(hcat([mino_to_array(mino) for mino in current_holdnext]...), 1, 1, length(node_list))
    score_list =
        predicter(currentbord_array, minopos_array, current_combo_array, current_back_to_back_array, tspin_array, current_holdnext_array)
    @views index = argmax(score_list[1, :])
    return node_list[index]
end



function select_node_list(node_list_list::Vector{Vector{Node}}, state_list::Vector{GameState}, predicer::Function)::Vector{Union{Node,Nothing}}

    array_size = sum(length(node_list) for node_list in node_list_list)
    currentbord_array = Array{Float32,4}(undef, 24, 10, 1, array_size)
    minopos_array = Array{Float32,4}(undef, 24, 10, 1, array_size)
    tspin_array = Array{Float32,2}(undef, 1, array_size)
    current_combo_array = Array{Float32,2}(undef, 1, array_size)
    current_back_to_back_array = Array{Float32,2}(undef, 1, array_size)
    current_holdnext_array = Array{Float32,3}(undef, 7, 6, array_size)
    response = Vector{Union{Node,Nothing}}(undef, length(node_list_list))

    index = 1
    for (node_list, state) in zip(node_list_list, state_list)
        current_size = length(node_list)
        if (current_size == 0)
            continue
        end
        current_range = index:index+current_size-1
        index += current_size
        currentbord = state.current_game_board.binary
        current_combo = state.combo
        current_back_to_back = state.back_to_back_flag
        current_holdnext = [state.hold_mino, state.mino_list[end-4:end]...]
        minopos_array[:, :, 1, current_range] = [generate_minopos(n.mino, n.position) .|> Float32 for n in node_list] |> vector2array
        tspin_array[:, current_range] = [(n.tspin > 1 ? 1 : 0) |> Float32 for n in node_list] |> vector2array
        currentbord_array[:, :, 1, current_range] = [currentbord .|> Float32 for _ in 1:length(node_list)] |> vector2array
        current_combo_array[:, current_range] = [current_combo |> Float32 for _ in 1:length(node_list)] |> vector2array
        current_back_to_back_array[:, current_range] = [current_back_to_back |> Float32 for _ in 1:length(node_list)] |> vector2array
        current_holdnext_array[:, :, current_range] = repeat(hcat([mino_to_array(mino) for mino in current_holdnext]...), 1, 1, length(node_list))
    end
    score_list =
        predicer(currentbord_array, minopos_array, current_combo_array, current_back_to_back_array, tspin_array, current_holdnext_array)
    index = 1
    for (i, node_list) in enumerate(node_list_list)
        current_size = length(node_list)
        if (state_list[i].game_over_flag)
            response[i] = nothing
            continue
        end
        @views current_maxscore_index = argmax(score_list[1, index:index+current_size-1])
        response[i] = node_list[current_maxscore_index]
        index += current_size
    end
    return response
end


function calc_td_error(
    state::GameState,
    node::Node,
    next_node_list::Vector{Node},
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


    if node.game_state.game_over_flag
        return abs(rescaling_reward(-1000 - inverse_rescaling_reward(current_expect_reward))) + ϵ
    end
    # 次の盤面の最大の価値を算出する
    max_node =
        select_node(next_node_list, node.game_state, predicter)
    node_holdnext = [node.game_state.hold_mino, node.game_state.mino_list[end-4:end]...]
    next_score = target_predicter(
        [node.game_state.current_game_board.binary] |> vector2array .|> Float32,
        [generate_minopos(max_node.mino, max_node.position)] |> vector2array .|> Float32,
        [node.game_state.combo] |> vector2array .|> Float32,
        [node.game_state.back_to_back_flag] |> vector2array .|> Float32,
        [max_node.tspin] |> vector2array .|> Float32,
        reshape(hcat([mino_to_array(mino) for mino in node_holdnext]...), 7, 6, 1),
    )[1]

    # 設置後に得られるスコア
    reward = node.game_state.score - state.score
    temporal_difference = rescaling_reward(reward + inverse_rescaling_reward(next_score) * discount_rate - inverse_rescaling_reward(current_expect_reward))
    return abs(temporal_difference) + ϵ
end
