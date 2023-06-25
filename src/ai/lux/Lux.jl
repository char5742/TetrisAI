module AILux
using CUDA
using Lux, NNlib, MLUtils, Zygote, Setfield
using JLD2, Optimisers
using Statistics, Random
function __init__()
    if CUDA.has_cuda()
        @info "CUDA is on"
        CUDA.allowscalar(false)
    end
end
include("model.jl")
export Model
include("../brain.jl")
export Learner, Brain
include("utils.jl")
export loadmodel, savemodel
include("network.jl")
export QNetwork, create_model
include("predict.jl")
export predict, vector2array
include("training.jl")
export fit!, create_optim
end
