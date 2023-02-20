module TetrisAI
using JLD2
using Tetris
using Flux
include("utils.jl")
export sleep30fps
include("node.jl")
include("analyzer.jl")
include("agent.jl")
export select_node, get_can_action_list
end # module TetrisAI
