using Tetris
include("../../core/TetrisAICore.jl")
import .TetrisAICore: loadmodel, predict, Model,
    GameState, MoveState, Action, is_valid_mino_movement,
    move, put_mino!, generate_minopos, get_node_list,
    Node,
    mino_to_array, AILux, create_model, select_node, vector2array
import Tetris:
    is_valid_mino_movement, CursesModel, update, set_state!, init, fin, sleep60fps
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
    model, ps, st = loadmodel("mainmodel.jld2")
    display(model)
    ps = ps |> gpu
    st = st |> gpu
    game_state = GameState()
    move_state = MoveState()
    display_model = CursesModel()
    try
        init(display_model)
        ai(Model(model, ps, st), game_state, move_state, display_model)
    catch e
        open("error.log", "w") do io
            showerror(io, e, catch_backtrace())
        end
        rethrow(e)
    finally
        fin(display_model)
    end

end

"""
node_list: 次の盤面ノードのリスト
state: 現在のゲーム状態
predicter: 盤面価値の予測関数
"""
function select_node(node_list::Vector{Node}, state::GameState, predicter::Function)::Node
    currentbord = state.current_game_board.binary
    current_ren = state.ren
    current_back_to_back = state.back_to_back_flag
    current_holdnext = [state.hold_mino, state.mino_list[end-4:end]...]
    minopos_array = [generate_minopos(n.mino, n.position) .|> Float32 for n in node_list] |> vector2array
    tspin_array = [(n.tspin > 1 ? 1 : 0) |> Float32 for n in node_list] |> vector2array
    currentbord_array = [currentbord .|> Float32 for _ in 1:length(node_list)] |> vector2array
    current_ren_array = [current_ren |> Float32 for _ in 1:length(node_list)] |> vector2array
    current_back_to_back_array = [current_back_to_back |> Float32 for _ in 1:length(node_list)] |> vector2array
    current_holdnext_array = repeat(hcat([mino_to_array(mino) for mino in current_holdnext]...), 1, 1, length(node_list))
    # use 
    NEXT = 4
    current_holdnext_array[:, 2:end-NEXT, :] .= 0
    if NEXT == 0
        current_holdnext_array[:, 1, :] .= 0
    end
    score_list =
        predicter(currentbord_array, minopos_array, current_ren_array, current_back_to_back_array, tspin_array, current_holdnext_array)
    if any(isnan, score_list) || any(isinf, score_list)
        throw("score_list is invalid")
    end
    @views index = argmax(score_list[1, :])
    GC.gc(false)
    return node_list[index]
end


function ai(model, game_state::GameState, move_state::MoveState, display::CursesModel)
    # ゲームオーバーになるまで繰り返す
    set_state!(display, game_state)
    update(display)
    start_time = time_ns()
    step = 0
    while !game_state.game_over_flag
        node_list = get_node_list(game_state)
        node = select_node(node_list, game_state, (x...) -> predict(model, x))
        step += 1
        for action in node.action_list
            key_state = get_current_key_state()
            is_pushed(key_state, :VK_ESCAPE) == 1 && exit()

            action!(game_state, action)

            fall!(move_state, game_state, action)
            put_mino!(move_state, game_state)

            sleep60fps(start_time)
            start_time = time_ns()
            set_state!(display, game_state)
            update(display, step)
        end
    end
end

function Tetris.update(d::CursesModel, step)
    update(d)
    Tetris.Curses.mvaddstr(11, 34, string("mscore: ", d.score / step))
    Tetris.Curses.refresh()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()

end