# 一ステップの経験
# 現在と次の盤面の情報が入る
mutable struct Experience
    current_state::GameState
    selected_node::Node
    next_node_list::Vector{Node}
    temporal_difference::Float64
end

