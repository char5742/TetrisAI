include("../core/TetrisAICore.jl")
using .TetrisAICore
using Tetris

game_state = GameState()



cnt = Ref{Int64}(0)
start = time_ns()
function recursion_search(game_state::GameState)
    node_list = get_node_list(game_state)
    for node in node_list
        if node.game_state.game_over_flag
            continue
        end
        cnt[] += 1
        elapsed = (time_ns() - start) / 1e9
        @show cnt[] / elapsed
        recursion_search(node.game_state)
    end
end

recursion_search(game_state)