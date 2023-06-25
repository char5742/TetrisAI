ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "90%"
include("config.jl")
include("src/TetrisAI.jl")
using Tetris
using .TetrisAI
using ProgressBars
using Printf
using Random
using Lux
using JLD2

function onestep!(game_state::GameState, current_step::Int)::Experience
    node_list = get_node_list(game_state)
    node = rand(game_state.rng, node_list)
    exp = Experience(GameState(game_state), node, get_node_list(node.game_state), 100)
    for action in node.action_list
        action!(game_state, action)
    end
    put_mino!(game_state)
    if current_step == 250
        game_end!(game_state)
    end
    return exp
end


function main()

    model, ps, st = create_model(Config.kernel_size, Config.res_blocks)
    # 学習パラメータ取得
    optim = create_optim(Config.learning_rate, ps)
    # TetrisAI.freeze_boardnet!(optim)
    brain = Brain(Model(model, ps, st), Model(model, ps, st))
    learner = Learner(brain, Config.ddqn_timing, 0, optim)

    # memory = Memory(Config.batchsize * Config.memoryscale)
    memory = load(raw"D:\memory.jld2")["memory"]
    # rng = MersenneTwister(1234)
    # initial_data_iter = ProgressBar(1:memory.capacity)
    # state = 1
    # game_state = GameState(rng)
    # while memory.index <= memory.capacity
    #     current_step = 0
    #     while !game_state.game_over_flag
    #         current_step += 1
    #         exp = onestep!(game_state, current_step)
    #         add!(memory, exp)
    #         if memory.index > memory.capacity
    #             break
    #         end
    #         (_, state) = iterate(initial_data_iter, state)
    #     end
    #     game_state = GameState(rng)
    # end
    @info "wormup start"
    loss, qmean, tspin = qlearn(learner, Config.batchsize, sample(memory, Config.batchsize), Config.γ)# i < 10e3 ? 200 :
    # update_temporal_difference(memory, new_temporal_difference_list)
    @info "wormup end"
    iter = ProgressBar(1:1000000)
    for i in iter
        loss, qmean, tspin = qlearn(learner, Config.batchsize, sample(memory, Config.batchsize), Config.γ)# i < 10e3 ? 200 :
        # update_temporal_difference(memory, new_temporal_difference_list)
        # 学習している間に何エピソード生成されたか
        if (loss != 0)
            open("output/log.csv", "a") do io
                println(io, @sprintf("%g, %g", loss, qmean))
            end
        end
        if i % 200 == 0
            try
                savemodel("model/mymodel_128_pre.jld2", brain.main_model)
            catch e
                @warn e
            end
        end
        set_description(iter, string(@sprintf("Loss: %9.4g, Qmean: %9.4g, tspin: %d", loss, qmean, tspin)))
        sleep(0.0001)
    end
end
main()
# savemodel("model/mymodel_128_pre.jld2",brain.main_model)
# data = (rand(rng, Float32, 24, 10, 1, batch_size), rand(rng, Float32, 24, 10, 1, batch_size), rand(rng, Float32, 1, batch_size), rand(rng, Float32, 1, batch_size), rand(rng, Float32, 1, batch_size), rand(rng, Float32, 7, 6, batch_size))
# target = rand(Float32, 1, batch_size)
# for i in iter
#     loss = TetrisAI.fit!(learner, data, target)
#     # 学習している間に何エピソード生成されたか
#     open("output/log.txt", "a") do io
#         println(io, @sprintf("%g", loss))
#     end
#     set_description(iter, string(@sprintf("Loss: %9.4g", loss)))

# end
