using Tetris
include("src/TetrisAI.jl")
using .TetrisAI
using JLD2
using Flux

module TetrisJulia
cat3(args...) = cat(args..., dims=3)

neg(x::AbstractArray{T}) where {T} = convert(T, -1.0) * x .+ convert(T, 1.0)

combo_normalize(x) = x / 30.0
end
using .TetrisJulia
function main()
    model = load("mymodel_tspin.jld2")["model"]
    display(model)
    model = model |> gpu

    game_state = GameState()
    move_state = MoveState()
    init_screen()
    ai(model, game_state, move_state)
    endwin()
end

function ai(model, game_state::GameState, move_state::MoveState)
    # ゲームオーバーになるまで繰り返す
    draw_game(game_state)
    start_time = time_ns()
    while !game_state.game_over_flag
        node_list = get_can_action_list(game_state)
        node = select_node(model,node_list, game_state)
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
            draw_game(game_state)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()

end