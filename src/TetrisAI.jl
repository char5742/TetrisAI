module TetrisAI
using JLD2
using Tetris
using Flux
using Optimisers
include("utils.jl")
export sleep30fps, save_matrix
include("component/Component.jl")
export Experience, Brain, Memory, add!, prioritized_sample
include("network.jl")
export QNetwork
include("analyzer.jl")
export get_node_list
include("agent.jl")
export Agent, select_node
include("training.jl")
export make_experience, qlearn, Learner
end # module TetrisAI
