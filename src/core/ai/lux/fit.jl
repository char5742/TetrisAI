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

update_learningrate!(optim, learning_rate) = Optimisers.adjust!(optim, learning_rate)
