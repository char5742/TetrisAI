loadmodel!(target, source) = Flux.loadmodel!(target,  cpu(source))
loadmodel_source!(target, path) = Flux.loadmodel!(target,  load(path, "model"))

function loadmodel(path::String)
    load(path)["model"]
end

function savemodel(path::String, model)
    jldsave(path; model=Flux.state(cpu(model)))
end

function freeze_boardnet!(optim)
    Optimisers.freeze!(optim[1][1])
end