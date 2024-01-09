using ProgressBars
using Printf
include("../core/TetrisAICore.jl")
using .TetrisAICore

using HTTP
using CUDA
using Dates
# https://github.com/JuliaGPU/CUDA.jl/pull/1943
# CUDA.math_mode!(CUDA.FAST_MATH)

const root = "http://127.0.0.1:10513"
const memoryserver = "$root/memory"
const paramserver = "$root/param"

import Serialization
include("../server/config.jl")
include("../lib/compress.jl")


function main(Config::_Config)
    current_learning_rate = Config.learning_rate
    learner = initialize_learner(
        Config.kernel_size,
        Config.res_blocks,
        current_learning_rate,
        Config.ddqn_timing,
        0,
    )
    while (check_ready_learning(Config.batchsize * Config.memoryscale) == false)
        sleep(1)
    end

    @info "Start Warmup"
    minibatch = get_minibatch()
    @time loss, qmean, tspin, new_temporal_difference_list = update_weight(learner, minibatch, Config.γ)
    update_priority(new_temporal_difference_list)
    @info "End Warmup"

    show_progress = true
    iter = show_progress ? ProgressBar(1:1_000_000) : 1:1_000_000
    minibatch = get_minibatch()
    for i in iter
        # 学習率を更新する
        if i % 50_000 == 0
            current_learning_rate *= 0.1
            update_learningrate!(learner.optim, current_learning_rate)
        end
        @sync begin
            try
                minibatch_task = @async get_minibatch()
                loss, qmean, tspin, new_temporal_difference_list = update_weight(learner, minibatch, Config.γ)
                show_progress && set_description(iter, string(@sprintf("Loss: %9.4g, Qmean: %9.4g, tspin: %d", loss, qmean, tspin)))
                # open("log.csv", "a") do io
                #     println(io, @sprintf("%s, %9.4g, %9.4g",Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), loss, qmean))
                # end
                @async update_priority(new_temporal_difference_list)
                minibatch = fetch(minibatch_task)
            catch e
                @error exception = (e, catch_backtrace())
            finally
                GC.gc(false)
            end
        end
    end
end



"""
Paramサーバーからパラメータを取得する
return ps, st
"""
function get_model_params(name::String)
    res = HTTP.request("GET", "$paramserver/$name")
    if res.status == 200
        ps, st = deserialize(res.body)
    else
        throw("Paramサーバーからparamを取得できませんでした")
    end
    return ps, st
end

"""
Paramサーバーに新しいパラメータを送信する
"""
function update_model_params(name::String, ps, st)
    data = serialize((ps, st) .|> cpu)
    res = HTTP.request("POST", "$paramserver/$name", body=data)
    if res.status == 200
    else
        throw("Paramサーバーにparamを送信できませんでした")
    end
end

"""
Memoryサーバーからミニバッチを取得する
"""
function get_minibatch()::Vector{Tuple{Int,Experience}}
    res = HTTP.request("GET", memoryserver)
    if res.status == 200
        minibatch = deserialize(res.body)
    else
        throw("Memoryサーバーからミニバッチを取得できませんでした")
    end
    return minibatch
end
"""
Memoryサーバー上のPriority, TD Errorを更新する
"""
function update_priority(new_temporal_difference_list::Vector{Tuple{Int,Float32}})
    res = HTTP.request("POST", "$memoryserver/priority", body=serialize(new_temporal_difference_list))
    if res.status != 200
        throw("Memoryサーバー上のPriority, TD Errorを更新できませんでした")
    end
end


module UpdateTimer
last_time = time()
function is_update_time()
    global last_time
    now = time()
    dt = now - last_time
    if dt > 10.0
        last_time = now
        true
    else
        false
    end
end
end

"""
ミニバッチを用いて重みを更新する
"""
function update_weight(learner::Learner, minibatch, γ)
    loss, qmean, tspin, new_temporal_difference_list = qlearn(learner, Config.batchsize, minibatch, γ)# i < 10e3 ? 200 :

    if UpdateTimer.is_update_time()
        @async update_model_params("mainmodel", learner.brain.main_model.ps, learner.brain.main_model.st)
    end
    if learner.taget_update_count % learner.taget_update_cycle == 0
        @async update_model_params("targetmodel", learner.brain.target_model.ps, learner.brain.target_model.st)
    end
    return loss, qmean, tspin, new_temporal_difference_list
end

"""
Learnerを初期化する
"""
function initialize_learner(
    kernel_size::Int64,
    resblock_size::Int64,
    learning_rate,
    taget_update_cycle::Int64,
    taget_update_count::Int64,
)::Learner
    model, _, _ = TetrisAICore.create_model(kernel_size, resblock_size, kernel_size)
    display(model)
    ps, st = get_model_params("mainmodel") .|> gpu
    t_ps, t_st = get_model_params("targetmodel") .|> gpu
    brain = Brain(Model(model, ps, st), Model(model, t_ps, t_st))
    optim = create_optim(learning_rate, ps)
    # optim = set_weightdecay(optim, learning_rate)
    learner = Learner(brain, taget_update_cycle, taget_update_count, optim)
    return learner
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
    main(Config)
end
