Base.@kwdef struct _Config
    load_params::Bool
    kernel_size::Int
    res_blocks::Int
    "時間割引率"
    γ::Float64
    ddqn_timing::Int
    learning_rate::Float32
    batchsize::Int
    memoryscale::Int
    epsilon_list::Vector{Float32}
end


const Config = _Config(
    load_params=false,
    kernel_size=128,
    res_blocks=4,
    γ=0.95,
    ddqn_timing=400,
    learning_rate=1.0f-5,
    batchsize=4,
    memoryscale=4*4,
    epsilon_list=Float32[
     0, 0, 0.01, 0.01,
     0.01, 0.01, 0.01, 0.01,
     0.05, 0.05, 0.05, 0.1, 
     0.01, 0.01, 0.01, 0.01,
     0.05, 0.05, 0.05, 0.1, 
     0.05, 0.05, 0.05, 0.1, 
     0.1, 0.1, 0.5, 0.5,
     0.5, 0.5, 0.7,
     ],
)
