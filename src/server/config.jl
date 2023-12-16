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
end


Config = _Config(
    load_params=false,
    kernel_size=128,
    res_blocks=5,
    γ=0.93,
    ddqn_timing=400,
    learning_rate=1.0f-4,
    batchsize=16,
    memoryscale=16^3,
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