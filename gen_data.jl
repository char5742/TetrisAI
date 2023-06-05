include("config.jl")
include("src/TetrisAI.jl")
using Tetris
using .TetrisAI
using ProgressBars
using Printf
using Random
using Lux

mutable struct MAgent
    id::Int64
    epsilon::Float64
    brain::Brain
end

function TetrisAI.select_node(agent::MAgent, node_list::Vector{Node}, state::GameState)::Node
    if agent.epsilon > rand()
        return rand(node_list)
    else
        return select_node(agent.brain.main_model, agent.brain.mainlock, node_list, state)
    end
end

function onestep!(game_state::GameState, agent::MAgent, current_step::Int)::Experience
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

function playing(agent::MAgent, memory::Memory, memsize::Int, epsilon_list::Vector{Float64}=[0, 0.05, 0.1, 0.3])
    game_state = GameState()
    total_step = 0
    iter = ProgressBar(1:memsize)
    state = 1
    currnet_index = memory.index
    epsilon_index = 1
    while (memory.index <= memory.capacity)
        agent.epsilon = epsilon_list[mod1(epsilon_index, length(epsilon_list))]
        epsilon_index += 1
        try
            current_step = 0
            while !game_state.game_over_flag
                current_step += 1
                exp = onestep!(game_state, agent, current_step)
                add!(memory, exp)
                total_step += 1
                sleep(0.0001)
            end

            if agent.id == 1
                open("output/log.csv", "a") do io
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
        for _ in 1:(memory.index-currnet_index)
            res = iterate(iter, state)
            if isnothing(res)
                return
            end
            (_, state) = res

        end
        currnet_index = memory.index
    end
end

using JLD2
memsize = 2^16
model, ps, st = loadmodel("model/mymodel_high.jld2")
display(model)
ps = ps |> gpu
st = st |> gpu
brain = Brain(Model(model, ps, st), Model(model, ps, st))
agent = MAgent(1, 0.0, brain)
memory = Memory(memsize)
playing(agent, memory, memsize)
# jldsave(raw"D:\memory.jld2"; memory=memory)
