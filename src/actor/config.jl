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



function get_config()
    r= HTTP.request("GET", "$root/config")
    if r.status == 200
        buffer = IOBuffer(r.body)
        config = deserialize(buffer)
    else
        throw("Configを取得できませんでした")
    end
    return config
end


Config = get_config()
