
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