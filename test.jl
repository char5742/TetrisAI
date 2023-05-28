include("config.jl")
include("src/TetrisAI.jl")
using Tetris
using .TetrisAI
using ProgressBars
using Printf
using Random
using Lux

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

    memory = Memory(Config.batchsize * Config.memoryscale)
    rng = MersenneTwister(1234)
    initial_data_iter = ProgressBar(1:memory.capacity)
    state = 1
    game_state = GameState(rng)
    while memory.index <= memory.capacity
        current_step = 0
        while !game_state.game_over_flag
            current_step += 1
            exp = onestep!(game_state, current_step)
            add!(memory, exp)
            if memory.index > memory.capacity
                break
            end
            (_, state) = iterate(initial_data_iter, state)
        end
        game_state = GameState(rng)
    end

    iter = ProgressBar(1:1000000)

    for i in iter
        loss, qmean, tspin, new_temporal_difference_list = qlearn(learner, Config.batchsize, prioritized_sample!(memory, Config.batchsize; priority=1), Config.γ)# i < 10e3 ? 200 :
        update_temporal_difference(memory, new_temporal_difference_list)
        # 学習している間に何エピソード生成されたか
        open("output/log.txt", "a") do io
            println(io, @sprintf("%g, %g", loss, qmean))
        end
        set_description(iter, string(@sprintf("Loss: %9.4g, Qmean: %9.4g, tspin: %d", loss, qmean, tspin)))
    end
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
end
main()