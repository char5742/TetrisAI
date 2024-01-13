Base.@kwdef struct _Config
    load_params::Bool
    channel_size::Int
    res_blocks::Int
    "時間割引率"
    γ::Float64
    ddqn_timing::Int
    learning_rate::Float32
    batchsize::Int
    memoryscale::Int
    compress::Bool
end



function get_config()
    r= HTTP.request("GET", "$ROOT/config")
    if r.status == 200
        buffer = IOBuffer(r.body)
        config = Serialization.deserialize(buffer)
    else
        throw("Configを取得できませんでした")
    end
    return config
end


Config = get_config()
