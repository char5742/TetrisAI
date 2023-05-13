module TetrisAI
using JLD2
using Tetris
include("utils.jl")
export sleep30fps, save_matrix, print_matrix
include("component/Component.jl")
using .Component
import .Component: Experience, Node, Memory, add!, prioritized_sample!, sum_td, update_temporal_difference
export Experience, Node, Memory, add!, prioritized_sample!, sum_td, update_temporal_difference
include("ai/lux/Lux.jl")
using .AILux
import .AILux: Learner, Brain, loadmodel, savemodel, QNetwork, create_optim, predict, create_model, Model
export Learner, Brain, loadmodel, savemodel, QNetwork, create_optim, predict, create_model, Model
include("analyzer.jl")
export get_node_list, generate_minopos
include("agent.jl")
export Agent, select_node
include("training.jl")
export  qlearn
end # module TetrisAI
