using Tetris
include("../../core/TetrisAICore.jl")
import .TetrisAICore: loadmodel, predict, Model,
    GameState, MoveState, Action, valid_movement,
    move, put_mino!, generate_minopos, get_node_list,
    Node, draw_game, init_screen, get_key_state, mysleep,
    mino_to_array, AILux, create_model, select_node, vector2array
using CUDA
using JLD2
using HTTP
using CodecZstd
using Serialization
CUDA.math_mode!(CUDA.PEDANTIC_MATH)
module TetrisAICore
module AILux
using CUDA
using Lux, LuxCUDA, NNlib, MLUtils, Zygote
import Lux: gpu_device, cpu_device
gpu = gpu_device()
cpu = cpu_device()
export gpu, cpu
using JLD2, Optimisers
using Statistics, Random
using NamedTupleTools
include("layers.jl")
include("network.jl")
export QNetwork
end
using .AILux
export QNetwork, gpu, cpu
end
using .TetrisAICore

function main()
    # model, _, _ = create_model(128, 5, 128; use_gpu=true)
    model, ps, st = loadmodel("mainmodel.jld2")
    display(model)
    ps = ps |> gpu
    st = st |> gpu
    open("log.csv", "w") do f
        println(f, "mscore")
    end
    for _ in 1:1000
        game_state = GameState()
        move_state = MoveState()
        # init_screen()
        ai(Model(model, ps, st), game_state, move_state)
    end
    # endwin()
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
    # use 
    NEXT = 4
    current_holdnext_array[:, 2:end-NEXT, :] .= 0
    if NEXT == 0
        current_holdnext_array[:, 1, :] .= 0
    end
    score_list =
        predicter(currentbord_array, minopos_array, current_combo_array, current_back_to_back_array, tspin_array, current_holdnext_array)
    if any(isnan, score_list) || any(isinf, score_list)
        throw("score_list is invalid")
    end
    @views index = argmax(score_list[1, :])
    GC.gc(false)
    return node_list[index]
end


function ai(model, game_state::GameState, move_state::MoveState)
    # ゲームオーバーになるまで繰り返す
    draw_game(game_state)
    start_time = time_ns()
    step = 0
    while !game_state.game_over_flag
        node_list = get_node_list(game_state)
        node = select_node(node_list, game_state, (x...) -> predict(model, x))
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

            # sleep30fps(start_time)
            start_time = time_ns()
            draw_game(game_state; step=step)
        end
        if step == 250
            game_end!(game_state)
        end
    end
    open("log.csv", "a") do f
        println(f, game_state.score)
    end
end

function sleep30fps(start_time)
    diff = (1 / 30) - (time_ns() - start_time) / 1e9
    if diff < 0
        return false
    else
        mysleep(diff)
        return true
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()

end