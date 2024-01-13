include("../core/TetrisAICore.jl")
using .TetrisAICore
using Tetris
using HTTP
using Printf
using Base.Threads
using Dates
using Random
using CUDA
import Serialization

actor_id = parse(Int, ARGS[1])
epsilon = parse(Float64, ARGS[2])
use_gpu = parse(Bool, ARGS[3])

const ROOT = "http://127.0.0.1:10513"
const MEMORYSERVER = "$ROOT/memory"
const PARAMSERVER = "$ROOT/param"

include("config.jl")
include("../lib/compress.jl")

function main()
    actor = initialize_actor(
        actor_id,
        Config.channel_size,
        Config.res_blocks,
        epsilon,
        use_gpu=use_gpu,
    )
    total_step = 0
    while true
        try
            game_state = GameState()
            current_step = 0
            while !game_state.game_over_flag
                current_step += 1
                exp = onestep!(game_state, actor, current_step)
                upload_exp(exp)
                if actor.id == 1
                    sleep(0.2)
                    draw_game2file(game_state.current_game_board.color[5:end, :]; score=game_state.score)
                end
                total_step += 1
            end

            if actor.id == 0
                open("log.csv", "a") do io
                    println(io, @sprintf("%s, %d, %3.1f", Dates.format(now(), "yyyy/mm/dd HH:MM:SS"), game_state.score, game_state.score / total_step))
                end
            end
            total_step = 0
        catch e
            @error exception = (e, catch_backtrace())
            GC.gc(true)
            total_step = 0
        end
        try
            # パラメータを取得する
            update_params(actor; use_gpu=use_gpu)
        catch e
            @error exception = (e, catch_backtrace())
        end
        GC.gc()
    end
end

function update_params(actor::Actor; use_gpu=false)
    if use_gpu
        ps, st = get_model_params("mainmodel") .|> gpu
        actor.brain.main_model.ps = ps
        actor.brain.main_model.st = st
        ps, st = get_model_params("targetmodel") .|> gpu
        actor.brain.target_model.ps = ps
        actor.brain.target_model.st = st
    else
        ps, st = get_model_params("mainmodel")
        actor.brain.main_model.ps = ps
        actor.brain.main_model.st = st
        ps, st = get_model_params("targetmodel")
        actor.brain.target_model.ps = ps
        actor.brain.target_model.st = st
    end
end

"""
Actorを初期化する
"""
function initialize_actor(
    actor_id::Int64,
    channel_size::Int64,
    resblock_size::Int64,
    epsilon::Float64;
    use_gpu=false
)
    model, ps, st = create_model(channel_size, resblock_size, channel_size; use_gpu=use_gpu)
    t_ps, t_st = ps, st
    try
        ps, st = get_model_params("mainmodel")
        t_ps, t_st = get_model_params("targetmodel")
    catch
        # もしサーバーから取得できなくても何もしない
    end
    if use_gpu
        ps, st = (ps, st) .|> gpu
        t_ps, t_st = (t_ps, t_st) .|> gpu
    end
    brain = Brain(Model(model, ps, st), Model(model, t_ps, t_st))
    return Actor(actor_id, epsilon, brain)
end

function onestep!(game_state::GameState, actor::Actor, current_step::Int)
    node_list = get_node_list(game_state)

    node = select_node(actor, node_list, game_state, (x...) -> predict(actor.brain.main_model, x))
    GC.gc(false)

    tmp_nodelist = get_node_list(node.game_state)
    td_error = calc_td_error(game_state, node, tmp_nodelist, Config.γ, (x...) -> predict(actor.brain.main_model, x), (x...) -> predict(actor.brain.target_model, x))
    GC.gc(false)

    exp = Experience(GameState(game_state), node, tmp_nodelist, td_error)
    for action in node.action_list
        action!(game_state, action)
    end
    put_mino!(game_state)
    if current_step == 250
        game_end!(game_state)
    end

    return exp
end

"""
Paramサーバーからパラメータを取得する
return ps, st
"""
function get_model_params(name::String)
    res = HTTP.request("GET", "$PARAMSERVER/$name")
    if res.status == 200
        ps, st = deserialize(res.body)
    else
        throw("Paramサーバーからparamを取得できませんでした")
    end
    return ps, st
end

"""
経験をmemoryサーバーに送信する
"""
function upload_exp(exp::Experience)
    res = HTTP.request("POST", MEMORYSERVER, body=serialize(exp))
    if res.status != 200
        throw("Memoryサーバーに経験を送信できませんでした")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
