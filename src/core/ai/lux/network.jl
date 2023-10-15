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
    # z = reshape(z, (240, size(z, 3), size(z, 4)))
    st = merge(st, (norm2=st_norm2, norm1=st_norm1, resblocks=st_resblocks))
    z, st
end

struct _QNetwork <: Lux.LuxCore.AbstractExplicitContainerLayer{(:board_net, :board_encoder, :combo_encoder, :btb_encoder, :tspin_encoder, :score_net, :mino_list_encoder)}
    board_net
    board_encoder
    combo_encoder
    btb_encoder
    tspin_encoder
    mino_list_encoder
    score_net
end
Base.getindex(m::_QNetwork, i) = i == 1 ? m.board_net() : m.score_net()
function (m::_QNetwork)((board, minopos, ren, btb, tspin, mino_list), ps, st)
    board_feature, st_board_net = m.board_net((board, minopos), ps.board_net, st.board_net)
    board_feature = unsqueeze(board_feature, dims=1)
    board_feature, _ = m.board_encoder(board_feature, ps.board_encoder, st.board_encoder)
    combo_feature, _ = m.combo_encoder(combo_normalize(ren), ps.combo_encoder, st.combo_encoder)
    combo_feature = unsqueeze(combo_feature, dims=2)
    btb_feature, _ = m.btb_encoder(btb, ps.btb_encoder, st.btb_encoder)
    btb_feature = unsqueeze(btb_feature, dims=2)
    tspin_feature, _ = m.tspin_encoder(tspin, ps.tspin_encoder, st.tspin_encoder)
    tspin_feature = unsqueeze(tspin_feature, dims=2)
    mino_list_feature, _ = m.mino_list_encoder(mino_list, ps.mino_list_encoder, st.mino_list_encoder)
    y = hcat(board_feature, combo_feature, btb_feature, tspin_feature, mino_list_feature)
    score, st_score_net = m.score_net((y, nothing), ps.score_net, st.score_net)
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
            Dense(1 => 48),
            Dense(1 => 48),
            Dense(1 => 48),
            Dense(1 => 48),
            Dense(7 => 48),
            ScoreNet(boardhidden_size + 3 + 6),
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
    features = 48
    nheads = 3
    matrix_size = 48
    seq_len = feature_size
    Decoder(
        Dense(matrix_size => features),
        PositionalEncodingLayer(features, seq_len),
        Chain([
            DecoderBlock(
                MultiHeadAttention(
                    Dense(features => features),
                    Dense(features => features),
                    Dense(features => features),
                    Dropout(0.1),
                    Dense(features => features),
                    nheads
                ),
                LayerNorm((features, 1)),
                Chain(Dense(features => features * 4, gelu), Dense(features * 4 => features), Dropout(0.1)),
                Dropout(0.1),
                LayerNorm((features, 1))
            )
            for i in 1:6
        ]),
        Chain(
            Dense(features * seq_len => 256, gelu),
            Dense(256 => 1),
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
    combo = rand(1, 1)
    btb = rand(1, 1)
    tspin = rand(1, 1)
    mino_list = rand(7, 6, 1)
    if (use_gpu)
        model((board, minopos, combo, btb, tspin, mino_list) |> gpu, ps, st) |> cpu
    else
        model((board, minopos, combo, btb, tspin, mino_list), ps, st) |> cpu
    end
end