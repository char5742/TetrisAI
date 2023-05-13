cat3(args...) = cat(args..., dims=3)

neg(x::AbstractArray{T}) where {T} = convert(T, -1.0) * x .+ convert(T, 1.0)

combo_normalize(x) = x / 30.0f0

struct BoardNet <: Lux.AbstractExplicitContainerLayer{(:conv1, :gn1, :resblocks, :conv2, :gn2, :gmp)}
    conv1
    gn1
    resblocks
    conv2
    gn2
    gmp
end
function BoardNet(kernel_size, resblock_size, output_size)
    return BoardNet(
        Conv((3, 3), 2 => kernel_size; pad=SamePad()),
        GroupNorm(kernel_size, 32),
        Chain([Chain(ResNetBlock(kernel_size), se_block(kernel_size)) for _ in 1:resblock_size]),
        Conv((1, 1), kernel_size => output_size; pad=SamePad()),
        GroupNorm(output_size, 32),
        GlobalMeanPool(),
    )
end
function (m::BoardNet)((board, minopos), ps, st)
    z = cat3(neg(board), neg(minopos))
    z, _ = m.conv1(z, ps[1], st[1])
    z, _ = m.gn1(z, ps[2], st[2])
    z, _ = m.resblocks(z, ps[3], st[3])
    z, _ = m.conv2(z, ps[4], st[4])
    z, _ = m.gn2(z, ps[5], st[5])
    z = swish(z)
    z, _ = m.gmp(z, ps[6], st[6])
    z = flatten(z)
    z, st
end


struct _QNetwork <: Lux.AbstractExplicitContainerLayer{(:board_net, :score_net,)}
    board_net
    score_net
end
Base.getindex(m::_QNetwork, i) = i == 1 ? m.board_net() : m.score_net()
function (m::_QNetwork)((board, minopos, ren, btb, tspin, mino_list), ps, st)
    # mino_vector = flatten(mino_list)
    board_feature, _ = m.board_net((board, minopos), ps[1], st[1])
    y = vcat(board_feature, combo_normalize(ren), btb, tspin)
    score, _ = m.score_net(y, ps[2], st[2])
    score, st
end


"""
bord_input_prev: size(24,10, B)  現在の盤面
minopos: size(24,10, B) ミノの配置箇所
combo_input: size(1,B) コンボ数
back_to_back: size(1,B)
tspin: size(1, B)
mino_list: size(6, 7, B) HOLD+NEXT

arg: (bord_input_prev ,minopos, combo_input,back_to_back, tspin, mino_list)  
return score  
"""
function QNetwork(kernel_size::Int64, resblock_size::Int64; use_gpu::Bool=true)
    use_gpu ?
    (Chain(
        _QNetwork(
            BoardNet(kernel_size, resblock_size, 128),
            ScoreNet(128 + 3),
        )
    ) |> gpu) :
    Chain(
        _QNetwork(
            BoardNet(kernel_size, resblock_size, 128),
            ScoreNet(128 + 3),
        )
    )
end

# 1_258_497, kernel_size=32, res_blocks=4,
function ResNetBlock(n)
    layers = Chain(
        Conv((3, 3), n => n, pad=SamePad()),
        GroupNorm(n, 32),
        swish,
        Conv((3, 3), n => n, pad=SamePad()),
        GroupNorm(n, 32),
    )

    return Chain(SkipConnection(layers, +), swish)
end



function se_block(ch, ratio=4)
    layers = Chain(
        GlobalMeanPool(),
        Conv((1, 1), ch => ch ÷ ratio, swish),
        Conv((1, 1), ch ÷ ratio => ch, sigmoid),
    )
    return Chain(SkipConnection(layers, .*))
end

"""
抽出した特徴から価値を計算するネットワーク  

arg: (feature)  
return: score  
"""
function ScoreNet(feature_size)
    Chain(
        Dense(feature_size => 1024, swish), # 41, 42
        Dense(1024 => 256, swish),# 43, 44
        Dense(256 => 1),# 45, 46
    )
end



function create_model(kernel_size::Int64, resblock_size::Int64; use_gpu::Bool=true)
    rng = Random.default_rng()
    if (use_gpu)
        model = QNetwork(kernel_size, resblock_size)
        ps, st = Lux.setup(rng, model) .|> gpu
        return model, ps, st
    else
        model = QNetwork(kernel_size, resblock_size)
        ps, st = Lux.setup(rng, model)
        return model, ps, st
    end
end