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
    multisteps::Int
end


Config = _Config(
    load_params=true,
    kernel_size=64,
    res_blocks=5,
    γ=0.95,
    ddqn_timing=400,
    learning_rate=1.0f-5,
    batchsize=16,
    memoryscale=16^2 * 4,
    epsilon_list=Float32[
        0, 0, 0.01, 0.01,
        #  0.01, 0.1, 0.05, 0.05,
        0.1, 0.05, 0.05, 0.3,
        #  0.01, 0.01, 0.01, 0.01,
        0.05, 0.05, 0.05, 0.1,
        #  0.05, 0.05, 0.05, 0.1, 
        #  0.1, 0.1, 0.5, 0.5,
        #  0.5, 0.5, 0.7,
    ],
    multisteps=3,
)

function config_route(request::HTTP.Request)
    target = request.target
    if contains(target, "/config")
        buffer = IOBuffer()
        serialize(buffer, Config)
        return HTTP.Response(200, take!(buffer))
    end
    nothing
end