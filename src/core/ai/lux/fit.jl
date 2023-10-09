function fit(model, ps, st, optim, x, y; use_gpu=true)
    if use_gpu
        x = x |> gpu
        y = y |> gpu
    end
    _st = Lux.LuxCore.trainmode(st)
    (trainingloss, st), back = pullback(ps) do ps
        ŷ, st = model(x, ps, _st)
        mean(abs2.(ŷ .- y)), st
    end
    gs = back((one(trainingloss), nothing))[1]
    st = Lux.LuxCore.testmode(st)
    st_opt, ps = Optimisers.update(optim, ps, gs)
    x = nothing
    y = nothing
    ps, st, st_opt, trainingloss
end

create_optim(learning_rate, ps) = Optimisers.setup(Optimisers.AdamW(learning_rate, (0.9, 0.95), learning_rate * 0.1), ps)


"""
bias, scale の重みのWeightDecayを再帰的に0にする
"""
function _set_weightdecay(t)
    chache = (;)
    for k in keys(t)
        if t[k] isa Optimisers.Leaf
            if k == :bias || k == :scale
                chache = merge_recursive(chache, (; k => Optimisers.Leaf(OptimiserChain(Adam(0.0, (0.9, 0.95), 1.0e-8), WeightDecay(0)), t[k].state,)),)
            end
        else
            chache = merge_recursive(chache, (; k => _set_weightdecay(t[k])))
        end
    end
    chache
end

set_weightdecay(optim) = merge_recursive(optim, _set_weightdecay(optim))