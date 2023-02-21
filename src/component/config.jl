Base.@kwdef struct Config
    load_params::Bool
    kernel_size::Int
    res_blocks::Int
    "time discount"
    γ::Float64
    ddqn_timing::Int
end
