module AIFlux
using CUDA
using Flux, NNlib, MLUtils
using JLD2, Optimisers
function __init__()
    if CUDA.has_cuda()
        @info "CUDA is on"
        CUDA.allowscalar(false)
    end
end
include("../brain.jl")
export Learner, Brain
include("utils.jl")
export loadmodel!, loadmodel, savemodel, freeze_boardnet!, loadmodel_source!
include("network.jl")
export QNetwork, QNetworkNextV2
include("predict.jl")
export predict, vector2array
include("training.jl")
export fit!, create_optim

end