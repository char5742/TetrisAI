function fit!(learner::Learner, x, y)
    x = x |> gpu
    y = y |> gpu
    trainingloss = lock(learner.brain.mainlock) do
        learner.brain.main_model.st = Lux.trainmode(learner.brain.main_model.st)
        (l, st), back = pullback(learner.brain.main_model.ps) do ps
            ŷ, st = learner.brain.main_model.model(x, ps, learner.brain.main_model.st)
            mean(abs2.(ŷ .- y)), st
        end
        gs = back((one(l), nothing))[1]
        learner.brain.main_model.st = Lux.testmode(st)
        st_opt, ps = Optimisers.update(learner.optim, learner.brain.main_model.ps, gs)
        learner.brain.main_model.ps = ps
        learner.optim = st_opt
        l
    end
    x = nothing
    y = nothing
    trainingloss
end

create_optim(learning_rate, ps) = Optimisers.setup(Optimisers.AdaBelief(learning_rate), ps)