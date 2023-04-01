cat3(args...) = cat(args..., dims=3)

neg(x::AbstractArray{T}) where {T} = convert(T, -1.0) * x .+ convert(T, 1.0)

combo_normalize(x) = x / 30.0f0


"""
盤面から特徴を抽出するネットワーク
bord_input_prev: size(24,10, 1, B)  現在の盤面  
minopos: size(24,10, 1, B) ミノの配置箇所  
combo_input: size(1,B) コンボ数  
back_to_back: size(1,B)  
tspin: size(1, B)

arg: ((bord_input_prev ,minopos), combo_input, back_to_back, tspin)  
return: feature size(kernel_size)
"""
function BoardNet(kernel_size::Int64, resblock_size::Int64)
    Chain(
        Parallel(cat3, neg, neg),
        Conv((3, 3), 2 => kernel_size; pad=SamePad()), # 1, 2
        GroupNorm(kernel_size, 32), # 3, 4
        swish,
        # [FusedMBConvBlock(kernel_size) for _ in 1:resblock_size]...,
        [ResNetBlock(kernel_size) for _ in 1:resblock_size]..., # 5~36
        Conv((3, 3), kernel_size => kernel_size; pad=SamePad()), # 37, 38
        GroupNorm(kernel_size, 32), # 39, 40
        swish,
        GlobalMeanPool(),
        flatten,
    )
end

"""
bord_input_prev: size(24,10, B)  現在の盤面  
minopos: size(24,10, B) ミノの配置箇所  
combo_input: size(1,B) コンボ数  
back_to_back: size(1,B)  
tspin: size(1, B)  

arg: ((bord_input_prev ,minopos), combo_input, back_to_back, tspin)  
return: score  
"""
function QNetwork(kernel_size::Int64, resblock_size::Int64)
    return Chain(
        Parallel(vcat, BoardNet(kernel_size, resblock_size), Chain(combo_normalize, swish), swish, swish),
        Dense(kernel_size + 3 => 1024, swish), # 41, 42
        Dense(1024 => 256, swish),# 43, 44
        Dense(256 => 1),# 45, 46
    ) |> gpu
end

function ValueNetwork(kernel_size::Int64)
    return Chain(
        Dense(kernel_size + 3 => 1024, swish), # 41, 42
        Dense(1024 => 256, swish),# 43, 44
        Dense(256 => 1),# 45, 46
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
        Dense(1024 => 1024, swish),# 43, 44
        Dense(1024 => 256, swish),# 43, 44
        Dense(256 => 256, swish),# 43, 44
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
        ScoreNet(kernel_size+3+6*7),
    ) |> gpu
end



struct TetrisNet{A,B,C,D,E,G}
    board::A
    minopos::B
    combo::C
    back_to_back::D
    tspin::E
    mino_list::G
end

Flux.@functor TetrisNet

function (m::TetrisNet)(x)
    ((net.board(x[1]), net.minopos(x[1])), net.combo(x[2]), net.back_to_back(x[3]), net.tspin(x[4]), net.mino_list(x[5]))
end