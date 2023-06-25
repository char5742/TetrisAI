function fit!(learner::Learner, x, y)
    x = x |> gpu
    y = y |> gpu
    _st = Lux.trainmode(learner.brain.main_model.st)
    (trainingloss, st), back = pullback(learner.brain.main_model.ps) do ps
        ŷ, st = learner.brain.main_model.model(x, ps, _st)
        mean(abs2.(ŷ .- y)), st
    end
    gs = back((one(trainingloss), nothing))[1]
    learner.brain.main_model.st = Lux.testmode(st)
    st_opt, ps = Optimisers.update(learner.optim, learner.brain.main_model.ps, gs)
    learner.brain.main_model.ps = ps
    learner.optim = st_opt
    x = nothing
    y = nothing
    trainingloss
end

create_optim(learning_rate, ps) = Optimisers.setup(Optimisers.AdamW(learning_rate), ps)