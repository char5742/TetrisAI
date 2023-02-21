struct Agent
    id::Int64
    epsilon::Float64
    brain::Brain
end

function select_node(agent::Agent,node_list::Vector{Node}, state::GameState)::Node
    if agent.epsilon> rand()
        return rand(node_list)
    else
        return select_node(agent.brain.main_model, node_list,state)
    end
end


function select_node(model,node_list::Vector{Node}, state::GameState)::Node
    currentbord = state.current_game_board.binary
    current_combo = state.combo
    current_back_to_back = state.back_to_back_flag
    next_bord_list = [n.game_state.current_game_board.binary for n in node_list]
    tspin_array = [n.tspin > 1 ? 1 : 0 for _ in 1:1, n in node_list]
    minopos_array = reduce((x, y) -> cat(x, y, dims=4), reshape(float(nextbord - currentbord), 24, 10, 1, 1) for nextbord in next_bord_list)
    currentbord_array = repeat(float(currentbord), 1, 1, 1, length(next_bord_list))
    current_combo_array = repeat([current_combo], 1, length(next_bord_list))
    current_back_to_back_array = repeat([current_back_to_back], 1, length(next_bord_list))
    # old version
    # score_list = model((currentbord_array + minopos_array, current_combo_array, current_back_to_back_array) |> gpu) |> cpu
    score_list = model(((currentbord_array, minopos_array), current_combo_array, current_back_to_back_array, tspin_array) |> gpu) |> cpu
    @views index = argmax(score_list[1, :])
    return node_list[index]  
end

