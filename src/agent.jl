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
    batch_size = 4
    num_batches = ceil(Int, length(node_list) / batch_size)
    currentbord = state.current_game_board.binary
    current_combo = state.combo
    current_back_to_back = state.back_to_back_flag
    current_holdnext = [state.hold_mino, state.mino_list[end-4:end]...]

    score_list = Vector{Float32}(undef, length(node_list))

    for batch_idx in 1:num_batches
        batch_start = (batch_idx - 1) * batch_size + 1
        batch_end = min(batch_idx * batch_size, length(node_list))
        current_batch = node_list[batch_start:batch_end]
        batch_length = length(current_batch)

        minopos_array = [generate_minopos(n.mino, n.position) .|> Float32 for n in current_batch] |> vector2array
        tspin_array = [(n.tspin > 1 ? 1 : 0) |> Float32 for n in current_batch] |> vector2array
        currentbord_array = [currentbord .|> Float32 for _ in 1:batch_length] |> vector2array
        current_combo_array = [current_combo |> Float32 for _ in 1:batch_length] |> vector2array
        current_back_to_back_array = [current_back_to_back |> Float32 for _ in 1:batch_length] |> vector2array
        current_holdnext_array = repeat(vcat([mino_to_array(mino) for mino in current_holdnext]...), 1, 1, batch_length)

        batch_scores = lock(modellock) do
            predict(model, (currentbord_array, minopos_array, current_combo_array, current_back_to_back_array, tspin_array, current_holdnext_array))
        end

        @views score_list[batch_start:batch_end] = batch_scores[1, 1:batch_length]
    end

    index = argmax(score_list)
    return node_list[index]
end

