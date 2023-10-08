cat3(args...) = cat(args..., dims=3)

neg(x::AbstractArray{T}) where {T} = convert(T, -1.0) * x .+ convert(T, 1.0)

combo_normalize(x) = x / 30.0f0

struct BoardNet <: Lux.LuxCore.AbstractExplicitContainerLayer{(:conv1, :norm1, :resblocks, :conv2, :norm2, :gmp)}
    conv1
    norm1
    resblocks
    conv2
    norm2
    gmp
end
function BoardNet(kernel_size, resblock_size, output_size)
    return BoardNet(
        Conv((3, 3), 2 => kernel_size; pad=SamePad()),
        BatchNorm(kernel_size),
        Chain([Chain(ResNetBlock(kernel_size), se_block(kernel_size)) for _ in 1:resblock_size]),
        Conv((3, 3), kernel_size => output_size; pad=SamePad()),
        BatchNorm(output_size),
        GlobalMeanPool(), # (24, 10, output_size) -> (1, 1, output_size)
    )
end
function (m::BoardNet)((board, minopos), ps, st)
    z = cat3(neg(board), neg(minopos))
    z, _ = m.conv1(z, ps.conv1, st.conv1)
    z, st_norm1 = m.norm1(z, ps.norm1, st.norm1)
    z = swish(z)
    z, st_resblocks = m.resblocks(z, ps.resblocks, st.resblocks)
    z, _ = m.conv2(z, ps.conv2, st.conv2)
    z, st_norm2 = m.norm2(z, ps.norm2, st.norm2)
    z = swish(z)
    z, _ = m.gmp(z, ps.gmp, st.gmp)
    z = flatten(z)
    st = merge(st, (norm2=st_norm2, norm1=st_norm1, resblocks=st_resblocks))
    z, st
end

struct _QNetwork <: Lux.LuxCore.AbstractExplicitContainerLayer{(:board_net, :score_net,)}
    board_net
    score_net
end
Base.getindex(m::_QNetwork, i) = i == 1 ? m.board_net() : m.score_net()
function (m::_QNetwork)((board, minopos, ren, btb, tspin, mino_list), ps, st)
    # mino_vector = flatten(mino_list)
    board_feature, st_board_net = m.board_net((board, minopos), ps.board_net, st.board_net)

    y = vcat(board_feature, combo_normalize(ren), btb, tspin)
    score, st_score_net = m.score_net(y, ps.score_net, st.score_net)
    st = merge(st, (board_net=st_board_net, score_net=st_score_net))
    score, st
end


"""
bord_input_prev: size(24,10, 1, B)  現在の盤面
minopos: size(24,10, 1, B) ミノの配置箇所
combo_input: size(1,B) コンボ数
back_to_back: size(1,B)
tspin: size(1, B)
mino_list: size(7, 6, B) HOLD+NEXT

arg: (bord_input_prev ,minopos, combo_input,back_to_back, tspin, mino_list)  
return score  
"""
function QNetwork(kernel_size::Int64, resblock_size::Int64, boardhidden_size::Int64)
    Chain(
        _QNetwork(
            BoardNet(kernel_size, resblock_size, boardhidden_size),
            ScoreNet(boardhidden_size + 3),
        )
    )
end

# 1_258_497, kernel_size=32, res_blocks=4,
function ResNetBlock(n)
    layers = Chain(
        Conv((3, 3), n => n, pad=SamePad()),
        BatchNorm(n),
        swish,
        Conv((3, 3), n => n, pad=SamePad()),
        BatchNorm(n),
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



function create_model(kernel_size::Int64, resblock_size::Int64, boardhidden_size::Int64; use_gpu::Bool=true)
    rng = Random.default_rng()
    if (use_gpu)
        model = QNetwork(kernel_size, resblock_size, boardhidden_size)
        ps, st = Lux.LuxCore.setup(rng, model) .|> gpu
        return model, ps, st
    else
        model = QNetwork(kernel_size, resblock_size, boardhidden_size)
        ps, st = Lux.LuxCore.setup(rng, model)
        return model, ps, st
    end
end