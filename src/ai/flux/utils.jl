loadmodel!(target, source; to_cpu=false) = Flux.loadmodel!(target, to_cpu ? cpu(source) : source)

function loadmodel(path::String)
    load(path)["model"]
end

function savemodel(path::String, model)
    save(path, "model", cpu(model))
end

function freeze_boardnet!(optim)
    Optimisers.freeze!(optim[1][1])
end