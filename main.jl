using Tetris
include("src/TetrisAI.jl")
using .TetrisAI
using JLD2
using Lux

module TetrisJulia
cat3(args...) = cat(args..., dims=3)

neg(x::AbstractArray{T}) where {T} = convert(T, -1.0) * x .+ convert(T, 1.0)

combo_normalize(x) = x / 30.0
end
using .TetrisJulia
function main()
    model, ps, st = loadmodel("model/mymodel_128_pre.jld2")
    display(model)
    ps = ps |> gpu
    st = st |> gpu

    game_state = GameState()
    move_state = MoveState()
    init_screen()
    ai(Model(model, ps, st), game_state, move_state)
    endwin()
end


function select_node(model, node_list::Vector{Node}, state::GameState)::Node
    currentbord = state.current_game_board.binary
    current_combo = state.combo
    current_back_to_back = state.back_to_back_flag
    current_holdnext = [state.hold_mino, state.mino_list[end-4:end]...]
    minopos_array = reduce((x, y) -> cat(x, y, dims=4), reshape(generate_minopos(n.mino, n.position) .|> Float32, 24, 10, 1, 1) for n in node_list)
    tspin_array = [(n.tspin > 1 ? 1 : 0) |> Float32 for _ in 1:1, n in node_list]
    currentbord_array = repeat(currentbord .|> Float32, 1, 1, 1, length(node_list))
    current_combo_array = repeat([current_combo |> Float32], 1, length(node_list))
    current_back_to_back_array = repeat([current_back_to_back |> Float32], 1, length(node_list))
    current_holdnext_array = repeat(vcat([TetrisAI.mino_to_array(mino) for mino in current_holdnext]...), 1, 1, length(node_list))
    score_list = predict(model, (currentbord_array, minopos_array, current_combo_array, current_back_to_back_array, tspin_array, current_holdnext_array))
    @views index = argmax(score_list[1, :])
    return node_list[index]
end



function ai(model, game_state::GameState, move_state::MoveState)
    # ゲームオーバーになるまで繰り返す
    draw_game(game_state)
    start_time = time_ns()
    step = 0
    while !game_state.game_over_flag
        node_list = get_node_list(game_state)
        node = select_node(model, node_list, game_state)
        step += 1
        for action in node.action_list
            get_key_state(:VK_ESCAPE) == 1 && exit()
            if move_state.set_count > 0 && (action.x != 0 || action.y != 0 || action.rotate != 0) && move_state.set_safe_cout < 15
                move_state.set_safe_cout += 1
                move_state.set_count = 0
            end
            action!(game_state, action)


            move_state.fall_count += 1
            if move_state.fall_count == 60
                move_state.fall_count = 0
                action = Action(0, 1, 0)
                if valid_movement(game_state.current_mino, game_state.current_position, game_state.current_game_board.binary, 0, 1)
                    game_state.current_position = move(game_state.current_position, 0, 1)
                    move_state.set_count = 0
                end
            end

            if !valid_movement(game_state.current_mino, game_state.current_position, game_state.current_game_board.binary, 0, 1)
                move_state.set_count += 1
            end
            if move_state.set_count == 30 || game_state.hard_drop_flag
                move_state.set_count = 0
                move_state.set_safe_cout = 0
                put_mino!(game_state)

            end

            sleep30fps(start_time)
            start_time = time_ns()
            draw_game(game_state; step=step)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()

end