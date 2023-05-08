using Distributed

actors = 3
learners = 1
addprocs(actors + learners, exeflags="--project=$(Base.active_project())")
ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "30%"
@everywhere include("config.jl")
@everywhere include("src/TetrisAI.jl")
@everywhere using Tetris
@everywhere using .TetrisAI
@everywhere using ProgressBars
@everywhere using Printf
@everywhere using Random


function main()
    exp_channel = RemoteChannel(() -> Channel{Experience}(2^4))
    @sync begin
        for i in 1:actors
            @spawn actor(exp_channel, i)
        end
        for i in 1:learners
             learner(exp_channel)
        end
    end
end

@everywhere function learner(exp_channel::RemoteChannel)
    # モデル読み込み
    main_model = TetrisAI.AIFlux.QNetwork(Config.kernel_size, Config.res_blocks)
    target_model = TetrisAI.AIFlux.QNetwork(Config.kernel_size, Config.res_blocks)
    if Config.load_params
        model = loadmodel("model/mymodel.jld2")

        loadmodel!(main_model, model)
        loadmodel!(target_model, main_model)
    end

    # 学習パラメータ取得
    optim = create_optim(Config.learning_rate, main_model)
    # TetrisAI.freeze_boardnet!(optim)
    brain = Brain(main_model, target_model)
    memory = Memory(Config.batchsize * Config.memoryscale)

    learner = Learner(brain, Config.ddqn_timing, 0, optim)

    # 初期データ待ち
    initial_data_iter = ProgressBar(1:memory.capacity)
    state = 1
    while (memory.index <= memory.capacity)
        if isready(exp_channel)
            add!(memory, take!(exp_channel))
            if memory.index > memory.capacity
                break
            end
            (_, state) = iterate(initial_data_iter, state)
        end
        sleep(0.001)
    end

    # while loop でも良いけど、こっちの方が分かりやすい
    iter = ProgressBar(1:1000000)
    current_index = memory.index
    for i in iter
        while isready(exp_channel)
            add!(memory, take!(exp_channel))
        end
        if i % 20 == 0
            try
                savemodel("model/mymodel.jld2", brain.main_model)
            catch e
                @warn e
            end
            
        end
        loss, qmean, tspin, new_temporal_difference_list = qlearn(learner, Config.batchsize, prioritized_sample!(memory, Config.batchsize; priority=1), Config.γ)# i < 10e3 ? 200 :
        update_temporal_difference(memory, new_temporal_difference_list)
        # 学習している間に何エピソード生成されたか
        generated_episode = memory.index - current_index
        set_description(iter, string(@sprintf("Loss: %9.4g, Qmean: %9.4g, tspin: %d, TDES: %9.4g, STEP: %2d", loss, qmean, tspin, sum_td(memory), generated_episode)))
        current_index = memory.index
    end
end

@everywhere function actor(exp_channel::RemoteChannel, id::Int)
    # モデル読み込み
    main_model = TetrisAI.AIFlux.QNetwork(Config.kernel_size, Config.res_blocks)
    target_model = TetrisAI.AIFlux.QNetwork(Config.kernel_size, Config.res_blocks)
    if Config.load_params
        model = loadmodel("model/mymodel.jld2")

        loadmodel!(main_model, model)
        loadmodel!(target_model, main_model)
    end

    # TetrisAI.freeze_boardnet!(optim)
    brain = Brain(main_model, target_model)
    playing(Agent(id, Config.epsilon_list[id], brain), exp_channel)
end



@everywhere function playing(agent::Agent, exp_channel::RemoteChannel)
    game_state = GameState()
    total_step = 0
    while true
        try
            current_step = 0
            while !game_state.game_over_flag
                current_step += 1
                exp = onestep!(game_state, agent, current_step)
                put!(exp_channel, exp)
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
        catch e
            @error e
            rethrow(e)
            GC.gc(true)
            total_step = 0
            game_state = GameState()
        end
        loadmodel!(agent.brain.main_model, loadmodel("model/mymodel.jld2"))
    end

end

@everywhere function onestep!(game_state::GameState, agent::Agent, current_step::Int)::Experience
    node_list = get_node_list(game_state)
    node = select_node(agent, node_list, game_state)
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

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

