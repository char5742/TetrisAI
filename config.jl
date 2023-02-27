Base.@kwdef struct _Config
    load_params::Bool
    kernel_size::Int
    res_blocks::Int
    "時間割引率"
    γ::Float64
    ddqn_timing::Int
end

const Config = _Config(
    load_params=false,
    kernel_size=128,
    res_blocks=4,
    γ=0.93,
    ddqn_timing=400,
)
