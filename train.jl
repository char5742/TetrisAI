ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"]="90%"
include("config.jl")
include("src/TetrisAI.jl")
using Tetris
using .TetrisAI
using ProgressBars
using Printf
using Random



function learning()
    main_model = QNetworkNextV2(Config.kernel_size, Config.res_blocks)
    target_model = QNetworkNextV2(Config.kernel_size, Config.res_blocks) 
    # value_model = TetrisAI.ValueNetwork(Config.kernel_size)
    if Config.load_params
        model = loadmodel("model/mymodel.jld2")

        loadmodel!(main_model[1][1], model[1][1])
        loadmodel!(target_model, main_model)
    end
    display(main_model)
    optim = create_optim(Config.learning_rate, main_model)
    TetrisAI.freeze_boardnet!(optim)
    brain = Brain(main_model, target_model)
    agent = Agent(0, 1.0, brain)
    memory = Memory(Config.batchsize * Config.memoryscale)


    game_state = GameState()
    while memory.index <= memory.capacity
        current_step = 0
        while !game_state.game_over_flag
            current_step += 1
            exp = onestep!(game_state, agent, current_step)
            add!(memory, exp)
        end
        game_state = GameState()
    end

    for i in eachindex(Config.epsilon_list)
        Threads.@spawn playing(Agent(i, Config.epsilon_list[i], brain), memory)
    end
    learner = Learner(brain, Config.ddqn_timing, 0, optim)
    iter = ProgressBar(1:1000000)
    current_index = memory.index
    for i in iter
        if i % 2000 == 10
            savemodel("model/mymodel.jld2", brain.main_model)
        end
        sleep(0.03)
        loss, qmean, tspin = qlearn(learner, Config.batchsize, prioritized_sample!(memory, Config.batchsize; priority=1))# i < 10e3 ? 200 :
        set_description(iter, string(@sprintf("Loss: %9.4g, Qmean: %9.4g, tspin: %d, TDES: %9.4g, STEP: %2d", loss, qmean, tspin, sum_td(memory), memory.index - current_index)))
        current_index = memory.index
    end
end

function playing(agent::Agent, memory::Memory)
    game_state = GameState()
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
            game_state = GameState()
        catch
            GC.gc(true)
            total_step = 0
            game_state = GameState()
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

