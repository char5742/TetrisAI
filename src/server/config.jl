Base.@kwdef struct _Config
    "学習済みモデルから学習を再開するかどうか"
    load_params::Bool
    "チャンネル数"
    channel_size::Int
    "残渣ブロックの数"
    res_blocks::Int
    "時間割引率"
    γ::Float64
    "targetmodelの更新間隔"
    ddqn_timing::Int
    "学習率"
    learning_rate::Float32
    batchsize::Int
    "経験の保持数を算出する際に利用する。経験の保持数 = memoryscale * batchsize"
    memoryscale::Int
    "通信時に圧縮を行うかどうか"
    compress::Bool
end


Config = _Config(
    load_params=false,
    channel_size=256,
    res_blocks=8,
    γ=0.997,
    ddqn_timing=400,
    learning_rate=1.0f-4,
    batchsize=16,
    memoryscale=16^3,
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