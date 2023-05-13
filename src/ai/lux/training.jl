function fit!(learner::Learner, x, y)
    x = x |> gpu
    y = y |> gpu
    trainingloss = lock(learner.brain.mainlock) do
        trainingloss, (gs, ) = withgradient(learner.brain.main_model.ps) do ps
            ŷ, st = learner.brain.main_model.model(x, ps, learner.brain.main_model.st)
            learner.brain.main_model.st = st
            mean(abs2.(ŷ .- y))
        end
        st_opt, ps = Optimisers.update(learner.optim, learner.brain.main_model.ps, gs)
        learner.brain.main_model.ps = ps
        learner.optim = st_opt
        trainingloss
    end
    x = nothing
    y = nothing
    trainingloss
end

create_optim(learning_rate, ps) = Optimisers.setup(Optimisers.AdaBelief(learning_rate), ps)