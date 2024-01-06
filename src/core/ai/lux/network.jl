cat3(args...) = cat(args..., dims=3)

neg(x::AbstractArray{T}) where {T} = T(-1.0) * x .+ T(1.0)

ren_normalize(x::AbstractArray{T}) where {T} = x / T(30.0f0)

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
        BatchNorm(kernel_size; epsilon=Float16(1.0f-6), momentum=Float16(0.1f0)),
        Chain(
            [ResNetBlock(kernel_size) for _ in 1:resblock_size]...,
        ),
        Conv((3, 3), kernel_size => output_size; pad=SamePad()),
        BatchNorm(output_size; epsilon=Float16(1.0f-6), momentum=Float16(0.1f0)),
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
    # z, _ = m.gmp(z, ps.gmp, st.gmp)
    z = flatten(z)
    # z = reshape(z, (240, size(z, 3), size(z, 4)))
    st = merge(st, (norm2=st_norm2, norm1=st_norm1, resblocks=st_resblocks))
    z, st
end

struct _QNetwork <: Lux.LuxCore.AbstractExplicitContainerLayer{(:board_net, :board_encoder, :ren_encoder, :btb_encoder, :tspin_encoder, :mino_list_encoder, :attention, :score_net)}
    board_net
    board_encoder
    ren_encoder
    btb_encoder
    tspin_encoder
    mino_list_encoder
    attention
    score_net
end
function (m::_QNetwork)((board, minopos, ren, btb, tspin, mino_list), ps, st)
    raw_board_feature, st_board_net = m.board_net((board, minopos), ps.board_net, st.board_net)
    # board_feature = unsqueeze(raw_board_feature, dims=1)
    # board_feature, _ = m.board_encoder(board_feature, ps.board_encoder, st.board_encoder)
    # ren_feature, _ = m.ren_encoder(ren_normalize(ren), ps.ren_encoder, st.ren_encoder)
    # ren_feature = unsqueeze(ren_feature, dims=2)
    # btb_feature, _ = m.btb_encoder(btb, ps.btb_encoder, st.btb_encoder)
    # btb_feature = unsqueeze(btb_feature, dims=2)
    # tspin_feature, _ = m.tspin_encoder(tspin, ps.tspin_encoder, st.tspin_encoder)
    # tspin_feature = unsqueeze(tspin_feature, dims=2)
    # mino_list_feature, _ = m.mino_list_encoder(mino_list, ps.mino_list_encoder, st.mino_list_encoder)
    # z = hcat(board_feature, ren_feature, btb_feature, tspin_feature, mino_list_feature)

    # attentioned, st_attention = m.attention((z, nothing), ps.attention, st.attention)
    # score, st_score_net = m.score_net(attentioned, ps.score_net, st.score_net)

    z = vcat(raw_board_feature, ren_normalize(ren), btb, tspin, flatten(mino_list))
    score, st_score_net = m.score_net(z, ps.score_net, st.score_net)
    st = merge(st, (
        board_net=st_board_net,
        score_net=st_score_net,
        # attention=st_attention,
    ))
    score, st
end


"""
bord_input_prev: size(24,10, 1, B)  現在の盤面
minopos: size(24,10, 1, B) ミノの配置箇所
ren_input: size(1,B) コンボ数
back_to_back: size(1,B)
tspin: size(1, B)
mino_list: size(7, 6, B) HOLD+NEXT

arg: (bord_input_prev ,minopos, ren_input,back_to_back, tspin, mino_list)  
return score  
"""
function QNetwork(kernel_size::Int64, resblock_size::Int64, boardhidden_size::Int64)
    matrix_size = 48
    nheads = 3
    # return Chain(
    #     _QNetwork(
    #         BoardNet(kernel_size, resblock_size, boardhidden_size),
    #         Dense(1 => matrix_size, gelu),
    #         Dense(1 => matrix_size, gelu),
    #         Dense(1 => matrix_size, gelu),
    #         Dense(1 => matrix_size, gelu),
    #         Dense(7 => matrix_size, gelu),
    #         attention(boardhidden_size + 3 + 6, matrix_size, nheads),
    #         ScoreNetSimple(boardhidden_size + 3 + 6),
    #     )
    # )
    Chain(
        _QNetwork(
            BoardNet(kernel_size, resblock_size, 1),
            NoOpLayer(),
            NoOpLayer(),
            NoOpLayer(),
            NoOpLayer(),
            NoOpLayer(),
            NoOpLayer(),
            ScoreNetSimple(240 + 3 + 42),
        )
    )
end

# 1_258_497, kernel_size=32, res_blocks=4,
function ResNetBlock(n)
    layers = Chain(
        Conv((3, 3), n => n, pad=SamePad()),
        BatchNorm(n, swish; epsilon=Float16(1.0f-5), momentum=Float16(0.1f0)),
        Conv((3, 3), n => n, pad=SamePad()),
        BatchNorm(n; epsilon=Float16(1.0f-5), momentum=Float16(0.1f0)),
        se_block(n),
    )

    return Chain(SkipConnection(layers, +), swish)
end

function ConvNeXtBlock(n)
    layers = Chain(
        Conv((5, 5), n => n; pad=SamePad(), groups=n),
        LayerNorm((24, 10, n); epsilon=Float16(1.0f-6)),
        WrappedFunction(x -> permutedims(x, (3, 1, 2, 4))),
        Dense(n => 4n, gelu),
        Dense(4n => n),
        WrappedFunction(x -> permutedims(x, (2, 3, 1, 4))),
    )

    return SkipConnection(layers, +)
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
function ScoreNetSimple(features)
    Chain(
        Dense(features => 1024, swish),
        Dense(1024 => 256, swish),
        Dense(256 => 1),
    )
end

function attention(seq_len, matrix_size, nheads)
    features = matrix_size
    Decoder(
        Dense(matrix_size => features),
        PositionalEncodingLayer(features, seq_len),
        Chain([
            DecoderBlock(
                MultiHeadAttention(
                    Dense(features => features),
                    Dense(features => features),
                    Dense(features => features),
                    Dropout(0.0),
                    Dense(features => features),
                    nheads
                ),
                LayerNorm((features, seq_len)),
                Chain(Dense(features => features * 4, gelu), Dense(features * 4 => features), Dropout(0.0)),
                Dropout(0.0),
                LayerNorm((features, seq_len))
            )
            for i in 1:6
        ]),
        Chain(
            GlobalMeanPool(),
            FlattenLayer(),
        ),
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

function warmup(model, ps, st; use_gpu=true)
    board = rand(24, 10, 1, 1)
    minopos = rand(24, 10, 1, 1)
    ren = rand(1, 1)
    btb = rand(1, 1)
    tspin = rand(1, 1)
    mino_list = rand(7, 6, 1)
    if (use_gpu)
        model((board, minopos, ren, btb, tspin, mino_list) |> gpu, ps, st) |> cpu
    else
        model((board, minopos, ren, btb, tspin, mino_list), ps, st) |> cpu
    end
end