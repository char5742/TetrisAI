struct Experience
    prev_game_bord::Matrix{Int64}
    minopos::Matrix{Int64}
    prev_combo::Int
    prev_back_to_back::Int
    prev_tspin::Int
    expected_reward::Float64
    temporal_difference::Float64
end

