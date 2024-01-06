# 計測用

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

NEXT = parse(Int, ARGS[1])
PREFIX = "./output/256_NEXT"
function main()
    # model, _, _ = create_model(128, 5, 128; use_gpu=true)
    model, ps, st = loadmodel("mainmodel.jld2")
    display(model)
    ps = ps |> gpu
    st = st |> gpu

    open("$PREFIX$NEXT.csv", "w") do f
    end
    for _ in 1:1000
        game_state = GameState()
        move_state = MoveState()
        ai(Model(model, ps, st), game_state, move_state; NEXT=NEXT)
    end
end

"""
node_list: 次の盤面ノードのリスト
state: 現在のゲーム状態
predicter: 盤面価値の予測関数
"""
function select_node(node_list::Vector{Node}, state::GameState, predicter::Function;
    NEXT)::Node
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


function ai(model, game_state::GameState, move_state::MoveState; NEXT=5)
    # ゲームオーバーになるまで繰り返す
    step = 0
    while !game_state.game_over_flag
        node_list = get_node_list(game_state)
        node = select_node(node_list, game_state, (x...) -> predict(model, x); NEXT=NEXT)
        step += 1
        for action in node.action_list
            key_state = get_current_key_state()
            is_pushed(key_state, :VK_ESCAPE) == 1 && exit()

            action!(game_state, action)

            fall!(move_state, game_state, action)
            put_mino!(move_state, game_state)

        end
        if step == 250
            game_end!(game_state)
        end
    end
    open("$PREFIX$NEXT.csv", "a") do f
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