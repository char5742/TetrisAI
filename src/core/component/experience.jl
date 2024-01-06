# 一ステップの経験
# 現在と次の盤面の情報が入る
mutable struct Experience
    current_state::GameState
    selected_node::Node
    temporal_difference::Float64
end