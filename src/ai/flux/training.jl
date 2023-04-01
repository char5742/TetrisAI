function fit!(learner::Learner, x, y)
    # hidden = learner.brain.main_model[1](x)
    # newlayer = learner.brain.main_model[2:end]
    # trainingloss, (∇model,) = Flux.withgradient(newlayer) do m
    #     Flux.Losses.mse(m(hidden), y)
    # end
    x = x |> gpu
    y = y |> gpu
    lock(learner.brain.mainlock) do
        trainingloss, (∇model,) = Flux.withgradient(learner.brain.main_model) do m
            Flux.Losses.mse(m(x), y)
        end

        Optimisers.update!(learner.optim, learner.brain.main_model, ∇model)
        trainingloss
    end
end

create_optim(learning_rate, model) = Optimisers.setup(Optimisers.AdaBelief(learning_rate), model)