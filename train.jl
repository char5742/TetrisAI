using Distributed

actors = 5
learners = 1
addprocs(actors + learners, exeflags="--project=$(Base.active_project())")
ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "90%"
@everywhere include("config.jl")
@everywhere include("src/TetrisAI.jl")
@everywhere using Tetris
@everywhere using .TetrisAI
@everywhere using ProgressBars
@everywhere using Printf
@everywhere using Random
@everywhere using Lux


function main()
    exp_channel = RemoteChannel(() -> Channel{Experience}(2^10))
    state_channel_list = [RemoteChannel(() -> Channel(1)) for _ in 1:actors]
    @sync begin
        for i in 1:actors
            @spawn actor(exp_channel, state_channel_list[i], i)
        end
        for i in 1:learners
            learner(exp_channel, state_channel_list)
        end
    end
end

@everywhere function learner(exp_channel::RemoteChannel, state_channel_list::Vector{RemoteChannel{Channel{Any}}})
    # モデル読み込み
    model, ps, st = create_model(Config.kernel_size, Config.res_blocks)
    if Config.load_params
        _, ps, st = loadmodel("model/mymodel.jld2")
        for state_channel in state_channel_list
            put!(state_channel, ps)
        end
        ps = ps |> gpu
        st = st |> gpu
        @info "load model"
    end

    # 学習パラメータ取得
    optim = create_optim(Config.learning_rate, ps)
    # TetrisAI.freeze_boardnet!(optim)
    brain = Brain(Model(model, ps, st), Model(model, ps, st))
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
                tmp_ps = brain.main_model.ps |> cpu
                for state_channel in state_channel_list
                    # 空であれば補充
                    if !isready(state_channel)
                        put!(state_channel, tmp_ps)
                    end
                end
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

@everywhere function actor(exp_channel::RemoteChannel, state_channel::RemoteChannel, id::Int)
    # モデル読み込み
    model, ps, st = create_model(Config.kernel_size, Config.res_blocks; use_gpu=false)
    main_model = Model(model, ps, st)
    brain = Brain(main_model, main_model)
    playing(Agent(id, Config.epsilon_list[id], brain), exp_channel, state_channel)
end



@everywhere function playing(agent::Agent, exp_channel::RemoteChannel, state_channel::RemoteChannel)
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
            @error exception = (e, catch_backtrace())
            rethrow(e)
            GC.gc(true)
            total_step = 0
            game_state = GameState()
        end
        if isready(state_channel)
            agent.brain.main_model.ps = take!(state_channel)
        end
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

