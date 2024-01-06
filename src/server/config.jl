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
    compress::Bool
end


Config = _Config(
    load_params=false,
    kernel_size=256,
    res_blocks=8,
    γ=0.997,
    ddqn_timing=400,
    learning_rate=1.0f-4,
    batchsize=16,
    memoryscale=16^3,
    epsilon_list=Float32[
        0, 0, 0.01, 0.05,
        0.1,
    ],
    compress=false,
)

function config_route(request::HTTP.Request)
    target = request.target
    if contains(target, "/config")
        buffer = IOBuffer()
        Serialization.serialize(buffer, Config)
        return HTTP.Response(200, take!(buffer))
    end
    nothing
end