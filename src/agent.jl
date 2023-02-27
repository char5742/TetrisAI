struct Agent
    id::Int64
    epsilon::Float64
    brain::Brain
end

function select_node(agent::Agent, node_list::Vector{Node}, state::GameState)::Node
    if agent.epsilon > rand()
        return rand(node_list)
    else
        return select_node(agent.brain.main_model, agent.brain.mainlock, node_list, state)
    end
end

function select_node(model, modellock, node_list::Vector{Node}, state::GameState)::Node
    currentbord = state.current_game_board.binary
    current_combo = state.combo
    current_back_to_back = state.back_to_back_flag
    minopos_array = reduce((x, y) -> cat(x, y, dims=4), reshape(float(generate_minopos(n.mino, n.position)), 24, 10, 1, 1) for n in node_list)
    tspin_array = [(n.tspin > 1 ? 1 : 0) |> Float32 for _ in 1:1, n in node_list]
    currentbord_array = repeat(float(currentbord), 1, 1, 1, length(node_list))
    current_combo_array = repeat([current_combo |> Float32], 1, length(node_list))
    current_back_to_back_array = repeat([current_back_to_back |> Float32], 1, length(node_list))
    score_list = lock(modellock) do
        # old version
        # model((currentbord_array + minopos_array, current_combo_array, current_back_to_back_array)  |> gpu) |> cpu
        model(((currentbord_array, minopos_array), current_combo_array, current_back_to_back_array, tspin_array) |> gpu)
    end
    @views index = argmax(score_list[1,:])
    return node_list[index]
end


function select_node(model, node_list::Vector{Node}, state::GameState)::Node
    currentbord = state.current_game_board.binary
    current_combo = state.combo
    current_back_to_back = state.back_to_back_flag
    minopos_array = reduce((x, y) -> cat(x, y, dims=4), reshape(float(generate_minopos(n.mino, n.position)), 24, 10, 1, 1) for n in node_list)
    tspin_array = [(n.tspin > 1 ? 1 : 0) |> Float32 for _ in 1:1, n in node_list]
    currentbord_array = repeat(float(currentbord), 1, 1, 1, length(node_list))
    current_combo_array = repeat([current_combo |> Float32], 1, length(node_list))
    current_back_to_back_array = repeat([current_back_to_back |> Float32], 1, length(node_list))
    # old version
    # score_list = model((currentbord_array + minopos_array, current_combo_array, current_back_to_back_array)  |> gpu) |> cpu
    score_list = model(((currentbord_array, minopos_array), current_combo_array, current_back_to_back_array, tspin_array) |> gpu)
    @views index = argmax(score_list[1,:])
    return node_list[index]
end

