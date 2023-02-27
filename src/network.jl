

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

cat3(args...) = cat(args..., dims=3)

neg(x::AbstractArray{T}) where {T} = convert(T, -1.0) * x .+ convert(T, 1.0)

combo_normalize(x) = x / 30.0f0


"""
bord_input_prev: size(24,10, B)  現在の盤面
combo_input: size(1,B) コンボ数
back_to_back: size(1,B) 

arg: (bord_input_prev,combo_input,back_to_back)
"""
function QNetwork(kernel_size::Int64, resblock_size::Int64)
    board = Chain(
        Parallel(cat3, neg, neg),
        Conv((3, 3), 2 => kernel_size; pad=SamePad()),
        BatchNorm(kernel_size),
        leakyrelu,
        # [FusedMBConvBlock(kernel_size) for _ in 1:resblock_size]...,
        [ResNetBlock(kernel_size) for _ in 1:resblock_size]...,
        Conv((3, 3), kernel_size => kernel_size; pad=SamePad()),
        BatchNorm(kernel_size),
        leakyrelu,
        GlobalMeanPool(),
        flatten,
    )
    return Chain(
        Parallel(vcat, board, Chain(combo_normalize, leakyrelu), leakyrelu, leakyrelu),
        Dense(kernel_size + 3 => 1024, leakyrelu),
        Dense(1024 => 256, leakyrelu),
        Dense(256 => 1),
    )
end
# 1_258_497, kernel_size=32, res_blocks=4,
function ResNetBlock(n)
    layers = Chain(
        Conv((3, 3), n => n, pad=SamePad()),
        BatchNorm(n),
        leakyrelu,
        Conv((3, 3), n => n, pad=SamePad()),
        BatchNorm(n),
    )

    return Chain(SkipConnection(layers, +), leakyrelu)
end

# 1_414_209, kernel_size=8, res_blocks=2,
function FusedMBConvBlock(n)
    scale = 4
    layers = Chain(
        Conv((3, 3), n => scale * n, pad=SamePad()),
        BatchNorm(scale * n),
        leakyrelu,
        se_block(scale * n),
        Conv((1, 1), scale * n => n, pad=SamePad()),
        BatchNorm(n),
    )
    return Chain(SkipConnection(layers, +), leakyrelu)
end

function MBConvBlock(n)
    scale = 4
    layers = Chain(
        Conv((1, 1), n => scale * n, pad=SamePad()),
        BatchNorm(scale * n),
        leakyrelu,
        DepthwiseConv((3, 3), scale * n => scale * n, pad=SamePad()),
        BatchNorm(scale * n),
        leakyrelu,
        se_block(scale * n),
        Conv((1, 1), scale * n => n, pad=SamePad()),
        BatchNorm(n),
    )
    return Chain(SkipConnection(layers, +), leakyrelu)
end


function se_block(ch, ratio=4)
    layers = Chain(
        GlobalMeanPool(),
        Conv((1, 1), ch => ch ÷ ratio, leakyrelu),
        Conv((1, 1), ch ÷ ratio => ch, sigmoid),
    )
    return Chain(SkipConnection(layers, .*))
end