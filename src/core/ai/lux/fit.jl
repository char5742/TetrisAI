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

create_optim(learning_rate, ps) = Optimisers.setup(Optimisers.AdaBelief(learning_rate), ps)