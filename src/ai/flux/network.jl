cat3(args...) = cat(args..., dims=3)

neg(x::AbstractArray{T}) where {T} = convert(T, -1.0) * x .+ convert(T, 1.0)

combo_normalize(x) = x / 30.0f0

struct BoardNet
    conv1
    gn1
    resblocks
    conv2
    gn2
    gmp
end
Flux.@functor BoardNet
function BoardNet(kernel_size, resblock_size, output_size)
    return BoardNet(
        Conv((3, 3), 2 => kernel_size; pad=SamePad()),
        GroupNorm(kernel_size, 32),
        [Chain(ResNetBlock(kernel_size), se_block(kernel_size)) for _ in 1:resblock_size],
        Conv((1, 1), kernel_size => output_size; pad=SamePad()),
        GroupNorm(output_size, 32),
        GlobalMeanPool(),
    )
end

function (m::BoardNet)((board, minopos))
    z = cat3(neg(board), neg(minopos))
    z = m.conv1(z)
    z = m.gn1(z)
    for resblock in m.resblocks
        z = resblock(z)
    end
    z = m.conv2(z)
    z = m.gn2(z)
    z = swish(z)
    z = m.gmp(z)
    z = flatten(z)
    z
end


struct _QNetwork{B,S}
    board_net::B
    score_net::S
end

Flux.@functor _QNetwork
Base.getindex(m::_QNetwork, i) = i == 1 ? m.board_net : m.score_net
function (m::_QNetwork)((board, minopos, ren, btb, tspin, mino_list))
    # mino_vector = flatten(mino_list)
    board_feature = m.board_net((board, minopos))
    y = vcat(board_feature, combo_normalize(ren), btb, tspin)
    m.score_net(y)
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
function QNetwork(kernel_size::Int64, resblock_size::Int64)
    Chain(
        _QNetwork(
            BoardNet(kernel_size, resblock_size, 128),
            ScoreNet(128+3),
        )
    ) |> gpu
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

"""
bord_input_prev: size(24,10, B)  現在の盤面  
minopos: size(24,10, B) ミノの配置箇所  
combo_input: size(1,B) コンボ数  
back_to_back: size(1,B)  
tspin: size(1, B)  
mino_list: size(6, 7, B) HOLD+NEXT  

arg: ((bord_input_prev ,minopos), combo_input,back_to_back, tspin, mino_list)  
return score  
"""
function QNetworkNextV2(kernel_size::Int64, resblock_size::Int64)
    mino_list = Chain(
        flatten,
    )
    Chain(
        Parallel(vcat, BoardNet(kernel_size, resblock_size), Chain(combo_normalize, swish), swish, swish, mino_list),
        ScoreNet(kernel_size + 3 + 6 * 7),
    ) |> gpu
end




struct BoardNetV2
    dense1
    conv1
    resblocks
    conv2
    gn
    gmp
end
Flux.@functor BoardNetV2
function BoardNetV2(kernel_size, resblock_size)
    return BoardNetV2(
        Chain(Dense(3 + 6 * 7 => kernel_size, swish), Dense(kernel_size => kernel_size, sigmoid)),
        Conv((3, 3), 2 => kernel_size; pad=SamePad()),
        [Chain(ResNetBlock(kernel_size),se_block(kernel_size)) for _ in 1:resblock_size],
        Conv((3, 3), kernel_size => kernel_size; pad=SamePad()),
        GroupNorm(kernel_size, 32),
        GlobalMeanPool(),
    )
end

function (m::BoardNetV2)((board, minopos, ren, btb, tspin, mino_vector))
    attention = m.dense1(vcat(combo_normalize(ren), btb, tspin, mino_vector))
    attention = reshape(attention, 1, 1, size(attention)...)
    z = cat3(neg(board), neg(minopos))
    z = m.conv1(z)
    for resblock in m.resblocks
        z = resblock(z)
    end
    z = z .* attention
    z = m.conv2(z)
    z = m.gn(z)
    z = swish(z)
    z = m.gmp(z)
    z = flatten(z)
    z
end



struct _QNetworkNextV3{B,S}
    board_net::B
    score_net::S
end

Flux.@functor _QNetworkNextV3
Base.getindex(m::_QNetworkNextV3, i) = i == 1 ? m.board_net : m.score_net
function (m::_QNetworkNextV3)((board, minopos, ren, btb, tspin, mino_list))
    mino_vector = flatten(mino_list)
    board_feature = m.board_net((board, minopos, combo_normalize(ren), btb, tspin, mino_vector))
    # y = vcat(board_feature, combo_normalize(ren), btb, tspin, mino_vector)
    m.score_net(board_feature)
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
function QNetworkNextV3(kernel_size::Int64, resblock_size::Int64)
    Chain(
        _QNetworkNextV3(
            BoardNetV2(kernel_size, resblock_size),
            ScoreNet(kernel_size),
        )
    ) |> gpu
end