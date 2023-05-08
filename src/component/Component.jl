module Component
using Tetris
include("node.jl")
export Node
include("experience.jl")
export Experience
include("memory.jl")
export Memory, add!, prioritized_sample!, update_temporal_difference, sum_td
include("reward.jl")
import .Reward: rescaling_reward
export rescaling_reward
end