include("../core/TetrisAICore.jl")
using .TetrisAICore
using Tetris
using Serialization
using HTTP
using Printf
using CodecZstd
using Dates

use_gpu = true
const root = "http://127.0.0.1:10513"
const memoryserver = "$root/memory"
const paramserver = "$root/param"


include("config.jl")
include("hippocampus.jl")

function main()
    actors, brain = initialize_actors(
        Config.kernel_size,
        Config.res_blocks,
        [0, 0, 0.05, 0.1, 0.3]
    )
    while true

        Threads.@threads for a in actors
            run(a)
        end
        Threads.@threads for a in actors
            run(a)
        end
        Threads.@threads for a in actors
            run(a)
        end
        update_model(brain)
    end


end

function update_model(brain::Brain)
    current_time = time()
    while true
        try
            # パラメータを取得する
            if time() - current_time > 60
                current_time = time()
                ps, st = get_model_params("mainmodel")
                if use_gpu
                    ps, st = (ps, st) .|> gpu
                end
                brain.main_model.ps = ps
                brain.main_model.st = st
                ps, st = get_model_params("targetmodel")
                if use_gpu
                    ps, st = (ps, st) .|> gpu
                end
                brain.target_model.ps = ps
                brain.target_model.st = st
                GC.gc()
            end
        catch e
            @error exception = (e, catch_backtrace())
        end
        sleep(1)
    end
end

function run(actor::Actor)
    game_state = GameState()
    campus = Hippocampus()
    total_step = 0
    current_step = 0
    while !game_state.game_over_flag
        current_step += 1
        onestep!(campus, game_state, actor, current_step)
        if actor.id == 1
            sleep(0.2)
            draw_game2file(game_state.current_game_board.color[5:end, :]; score=game_state.score)
        end
        total_step += 1
    end
    exp_list = create_experience(campus, actor, Config.multisteps, Config.γ)
    for exp in exp_list
        upload_exp(exp)
    end
end

"""
Actorを初期化する
"""
function initialize_actor(
    actor_id::Int64,
    kernel_size::Int64,
    resblock_size::Int64,
    epsilon::Float64
)
    model, ps, st = create_model(kernel_size, resblock_size, 128; use_gpu=false)
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


"""
複数のActorを初期化する
"""
function initialize_actors(
    kernel_size::Int64,
    resblock_size::Int64,
    epsilon_list::Vector{Float64}
)
    model, ps, st = create_model(kernel_size, resblock_size, 128; use_gpu=false)
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
    return [Actor(i, epsilon_list[i], brain) for i in 1:Threads.nthreads], brain
end

function onestep!(campus::Hippocampus, game_state::GameState, actor::Actor, current_step::Int)::Experience
    node_list = get_node_list(game_state)
    node = select_node(actor, node_list, game_state)
    add_action_data(campus, GameState(game_state), node)
    GC.gc(false)
    for action in node.action_list
        action!(game_state, action)
    end
    put_mino!(game_state)
    if current_step == 250
        game_end!(game_state)
    end
end

"""
Paramサーバーからパラメータを取得する
return ps, st
"""
function get_model_params(name::String)
    res = HTTP.request("GET", "$paramserver/$name")
    if res.status == 200
        buffer = IOBuffer(res.body)
        stream = ZstdDecompressorStream(buffer)
        ps, st = deserialize(stream)
        close(stream)
    else
        throw("Paramサーバーからparamを取得できませんでした")
    end
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


function check_ready_learning(memorysize::Int64)::Bool
    res = HTTP.request("GET", "$memoryserver/index")
    if res.status == 200
        return parse(Int64, String(res.body)) >= memorysize
    else
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
