# 一ステップの経験
# 現在と次の盤面の情報が入る
mutable struct Experience
    current_state::GameState
    selected_node::Node
    multistep_node::Node
    multistep_next_node_list::Vector{Node}
    multistep_reward::Float64
    multisteps::Int
    temporal_difference::Float64
end

