module TetrisAI
using JLD2
using Tetris
include("utils.jl")
export sleep30fps, save_matrix, print_matrix
include("component/Component.jl")
using .Component
import .Component:Experience, Node, Memory, add!, prioritized_sample!, sum_td
export Experience, Node, Memory, add!, prioritized_sample!, sum_td
include("ai/flux/Flux.jl")
using .AIFlux
import .AIFlux:Learner, Brain, loadmodel!, loadmodel, savemodel, QNetwork, create_optim
export Learner, Brain, loadmodel!, loadmodel, savemodel, QNetwork, create_optim
include("analyzer.jl")
export get_node_list, generate_minopos
include("agent.jl")
export Agent, select_node
include("training.jl")
export make_experience, qlearn
end # module TetrisAI
