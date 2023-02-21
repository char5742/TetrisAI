include("src/TetrisAI.jl")
using .TetrisAI
using Tetris
state = GameState()
@time  length(get_node_list(state))