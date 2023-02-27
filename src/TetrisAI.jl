module TetrisAI
using JLD2
using Tetris
using Flux
using Optimisers
using Flux
using CUDA
using MLUtils
include("utils.jl")
export sleep30fps, save_matrix,print_matrix
include("component/Component.jl")
export Experience, Node, Brain, Memory, add!, prioritized_sample!, sum_td
include("network.jl")
export QNetwork
include("analyzer.jl")
export get_node_list,generate_minopos
include("agent.jl")
export Agent, select_node
include("training.jl")
export make_experience, qlearn, Learner
end # module TetrisAI
