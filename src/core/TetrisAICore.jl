module TetrisAICore

using JLD2
using Tetris
include("component/Component.jl")
using .Component
import .Component: Experience, Node, Memory, add!, sample, prioritized_sample!, sum_td, update_temporal_difference
export Experience, Node, Memory, add!, sample, prioritized_sample!, sum_td, update_temporal_difference
include("ai/lux/Lux.jl")
using .AILux
import .AILux: Brain, loadmodel, savemodel, QNetwork, create_optim, predict, create_model, Model, gpu, cpu
export  Brain, loadmodel, savemodel, QNetwork, create_optim, predict, create_model, Model, gpu, cpu
include("analyzer.jl")
export get_node_list, generate_minopos, mino_to_array
include("actor.jl")
export Actor, select_node, select_node_list, calc_td_error
include("learner.jl")
export Learner, qlearn


end # module TetrisAICore
