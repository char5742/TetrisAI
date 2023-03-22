mutable struct Experience
    prev_game_bord::Matrix{Int32}
    minopos::Matrix{Int32}
    prev_combo::Int
    prev_back_to_back::Int
    prev_tspin::Int
    prev_holdnext::Matrix{Int32}
    expected_reward::Float64
    temporal_difference::Float64
end

