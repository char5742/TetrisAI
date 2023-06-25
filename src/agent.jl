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
    current_holdnext = [state.hold_mino, state.mino_list[end-4:end]...]
    minopos_array = [generate_minopos(n.mino, n.position) .|> Float32 for n in node_list] |> vector2array
    tspin_array = [(n.tspin > 1 ? 1 : 0) |> Float32 for n in node_list] |> vector2array
    currentbord_array = [currentbord .|> Float32 for _ in 1:length(node_list)] |> vector2array
    current_combo_array = [current_combo |> Float32 for _ in 1:length(node_list)] |> vector2array
    current_back_to_back_array = [current_back_to_back |> Float32 for _ in 1:length(node_list)] |> vector2array
    current_holdnext_array = repeat(hcat([mino_to_array(mino) for mino in current_holdnext]...), 1, 1, length(node_list))
    score_list =
        predict(model, (currentbord_array, minopos_array, current_combo_array, current_back_to_back_array, tspin_array, current_holdnext_array))
    @views index = argmax(score_list[1, :])
    return node_list[index]
end


function select_node_list(model, modellock, node_list_list::Vector{Vector{Node}}, state_list::Vector{GameState})::Vector{Union{Node, Nothing}}

    array_size = sum(length(node_list) for node_list in node_list_list)
    currentbord_array = Array{Float32,4}(undef, 24, 10, 1, array_size)
    minopos_array = Array{Float32,4}(undef, 24, 10, 1, array_size)
    tspin_array = Array{Float32,2}(undef, 1, array_size)
    current_combo_array = Array{Float32,2}(undef, 1, array_size)
    current_back_to_back_array = Array{Float32,2}(undef, 1, array_size)
    current_holdnext_array = Array{Float32,3}(undef, 7, 6, array_size)
    response = Vector{Union{Node, Nothing}}(undef, length(node_list_list))

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
        predict(model, (currentbord_array, minopos_array, current_combo_array, current_back_to_back_array, tspin_array, current_holdnext_array))
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
