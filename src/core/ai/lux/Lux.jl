module AILux
using CUDA
using Lux, LuxCUDA, NNlib, MLUtils, Zygote
import Lux: gpu_device, cpu_device
gpu = gpu_device()
cpu = cpu_device()
export gpu, cpu
using JLD2, Optimisers
using Statistics, Random
using NamedTupleTools
function __init__()
    if CUDA.has_cuda()
        @info "CUDA is on"
        CUDA.allowscalar(false)
    end
end
include("model.jl")
export Model
include("../brain.jl")
export Brain
include("utils.jl")
export loadmodel, savemodel
include("layers.jl")
include("network.jl")
export QNetwork, create_model
include("predict.jl")
export predict, vector2array
include("fit.jl")
export fit, create_optim, set_weightdecay
end
