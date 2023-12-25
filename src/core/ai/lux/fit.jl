"""
DQNRegLoss
過小評価することでモデルの汎化性能を上げるloss関数
https://arxiv.org/pdf/2101.03958.pdf
"""
function DQNRegLoss(ŷ, y; weight=0.1)
    δ = ŷ - y
    mean(weight * ŷ + δ .^ 2)
end


function fit(model, ps, st, optim, x, y; use_gpu=true)
    if use_gpu
        x = x |> gpu
        y = y |> gpu
    end
    _st = Lux.LuxCore.trainmode(st)
    (trainingloss, st), back = pullback(ps) do ps
        ŷ, st = model(x, ps, _st)
        DQNRegLoss(ŷ, y), st
    end
    gs = back((one(trainingloss), nothing))[1]
    st = Lux.LuxCore.testmode(st)
    st_opt, ps = Optimisers.update(optim, ps, gs)
    x = nothing
    y = nothing
    ps, st, st_opt, trainingloss
end

Optimisers.@def struct WeightDecayWithEta <: AbstractRule
    eta = 0.001
    gamma = 5e-4
end

Optimisers.init(o::WeightDecayWithEta, x::AbstractArray) = nothing

function Optimisers.apply!(o::WeightDecayWithEta, state, x::AbstractArray{T}, dx) where {T}
    η, γ = T(o.eta), T(o.gamma)
    dx′ = Optimisers.@lazy dx + η * γ * x

    return state, dx′
end


create_optim(learning_rate, ps) = Optimisers.setup(Optimisers.OptimiserChain(Adam(learning_rate, (0.9, 0.999)), WeightDecayWithEta(learning_rate, 0.05)), ps)

update_learningrate!(optim, learning_rate) = Optimisers.adjust!(optim, learning_rate)

function set_weightdecay(optim, learning_rate)
    """
    bias, scale の重みのWeightDecayを再帰的に0にする
    """
    function _set_weightdecay(t)
        chache = (;)
        for k in keys(t)
            if t[k] isa Optimisers.Leaf
                if k == :bias || k == :scale
                    chache = merge_recursive(chache, (; k => Optimisers.Leaf(OptimiserChain(Adam(learning_rate, (0.9, 0.95), 1.0e-8), WeightDecay(0)), t[k].state,)),)
                end
            else
                chache = merge_recursive(chache, (; k => _set_weightdecay(t[k])))
            end
        end
        chache
    end
    merge_recursive(optim, _set_weightdecay(optim))
end