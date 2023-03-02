include("config.jl")
include("src/TetrisAI.jl")
using Tetris
using .TetrisAI
using Flux, JLD2, Optimisers
using ProgressBars
using Printf
using Random



function learning()
    batch_size = 16
    main_model = QNetwork(Config.kernel_size, Config.res_blocks)
    display(main_model)
    if Config.load_params
        model = load("model/mymodel.jld2")["model"]
        Flux.loadmodel!(main_model, model)
    end
    main_model = main_model |> gpu

    target_model = QNetwork(Config.kernel_size, Config.res_blocks) |> gpu
    optim = Optimisers.setup(Optimisers.AdaBelief(1f-5), main_model)
    brain = Brain(main_model, target_model)
    agent = Agent(0, 1.0, brain)
    memory = Memory(batch_size * 16^2)


    game_state = GameState(MersenneTwister(1))
    while memory.index <= memory.capacity
        current_step = 0
        while !game_state.game_over_flag
            current_step += 1
            exp = onestep!(game_state, agent, current_step)
            add!(memory, exp)
        end
        game_state = GameState(MersenneTwister(1))
    end
    epsilon_list = Float64[0, 0, 0.01, 0.05, 0.1]
    for i in eachindex(epsilon_list)
        Threads.@spawn playing(Agent(i, epsilon_list[i], brain), memory)
    end
    learner = Learner(brain, Config.ddqn_timing, 0, optim)
    iter = ProgressBar(1:1000000)
    current_index = memory.index
    for i in iter
        if i % 2000 == 10
            save("model/mymodel.jld2", "model", cpu(brain.main_model))
        end
        sleep(0.03)
        loss, qmean, tspin = qlearn(learner, batch_size, prioritized_sample!(memory, batch_size; priority=1))# i < 10e3 ? 200 :
        set_description(iter, string(@sprintf("Loss: %9.4g, Qmean: %9.4g, tspin: %d, TDES: %9.4g, STEP: %2d", loss, qmean, tspin, sum_td(memory), memory.index - current_index)))
        current_index = memory.index
    end
end

function playing(agent::Agent, memory::Memory)
    game_state = GameState(MersenneTwister(1))
    total_step = 0
    while true
        try
            current_step = 0
            while !game_state.game_over_flag
                current_step += 1
                exp = onestep!(game_state, agent, current_step)
                add!(memory, exp)
                if agent.id == 1
                    sleep(0.2)
                    save_matrix(game_state.current_game_board.color[5:end, :]; filename="data/bord.txt")
                    write("data/score.txt", string(game_state.score))
                    draw_game2file(game_state.current_game_board.color[5:end, :]; score=game_state.score)
                end
                total_step += 1
            end

            if agent.id == 1
                open("output/log.txt", "a") do io
                    println(io, @sprintf("%d, %3.1f", game_state.score, game_state.score / total_step))
                end
            end
            total_step = 0
            game_state = GameState(MersenneTwister(1))
        catch
            GC.gc(true)
            total_step = 0
            game_state = GameState(MersenneTwister(1))
        end
    end

end

function onestep!(game_state::GameState, agent::Agent, current_step::Int)::Experience
    node_list = get_node_list(game_state)
    node = select_node(agent, node_list, game_state)
    previous_state = GameState(game_state)
    for action in node.action_list
        action!(game_state, action)
    end

    exp = make_experience(agent.brain, previous_state, node, Config.γ)
    put_mino!(game_state)
    if current_step == 250
        game_end!(game_state)
    end
    return exp
end

if abspath(PROGRAM_FILE) == @__FILE__
    learning()
end

