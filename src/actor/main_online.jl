include("../core/TetrisAICore.jl")
using .TetrisAICore
using Tetris
using Serialization
using HTTP
using Printf
using Base.Threads
using CodecZstd
using Dates, JLD2
using Random
using CUDA
CUDA.math_mode!(CUDA.DEFAULT_MATH)

actor_id = parse(Int, ARGS[1])
epsilon = parse(Float64, ARGS[2])
use_gpu = parse(Bool, ARGS[3])

const root = "http://127.0.0.1:10513"
const memoryserver = "$root/memory"
const paramserver = "$root/param"

include("config.jl")

function main()
    actor = initialize_actor(
        actor_id,
        Config.kernel_size,
        Config.res_blocks,
        epsilon,
        use_gpu=use_gpu,
    )
    while true
        game_state = GameState()
        total_step = 0
        exp_list = Experience[]
        try
            current_step = 0
            while !game_state.game_over_flag
                current_step += 1
                exp = onestep!(game_state, actor, current_step)
                GC.gc(false)
                push!(exp_list, exp)
                if actor.id == 1
                    sleep(0.2)
                    draw_game2file(game_state.current_game_board.color[5:end, :]; score=game_state.score)
                end
                total_step += 1
            end
            id = actor.id
            open("output/log_$(id).csv", "a") do io
                println(io, @sprintf("%s, %d, %3.1f", Dates.format(now(), "yyyy/mm/dd HH:MM:SS"), game_state.score, game_state.score / total_step))
            end
        catch e
            @error exception = (e, catch_backtrace())
            GC.gc(true)
        end
        try
            @sync begin
                for exp in exp_list
                    @async upload_exp(exp)
                end
                # パラメータを取得する
                @async update_params(actor; use_gpu=use_gpu)
            end
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
    kernel_size::Int64,
    resblock_size::Int64,
    epsilon::Float64;
    use_gpu=false
)
    model, ps, st = create_model(kernel_size, resblock_size, kernel_size; use_gpu=use_gpu)
    t_ps, t_st = ps, st
    try
        ps, st = get_model_params("mainmodel")
        t_ps, t_st = get_model_params("targetmodel")
    catch e
        @error e
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
    tmp_nodelist = get_node_list(node.game_state)
    td_error = calc_td_error(game_state, node, tmp_nodelist, Config.γ, (x...) -> predict(actor.brain.main_model, x), (x...) -> predict(actor.brain.target_model, x))
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
    while isfile("model/model.lock")
        sleep(1)
    end
    data = load("model/$(name).jld2")
    ps = data["ps"]
    st = data["st"]
    return ps, st
end

"""
経験をmemoryサーバーに送信する
"""
function upload_exp(exp::Experience)
    buffer = IOBuffer()
    serialize(buffer, exp)
    compressed = transcode(ZstdCompressor, take!(buffer))
    res = HTTP.request("POST", memoryserver, body=compressed)
    if res.status != 200
        throw("Memoryサーバーに経験を送信できませんでした")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
