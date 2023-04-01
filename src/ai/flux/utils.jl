loadmodel!(target, source) = Flux.loadmodel!(target, source)

function loadmodel(path::String)
    load(path)["model"]
end

function savemodel(path::String, model)
    save(path, "model", cpu(model))
end

function freeze_boardnet!(optim)
    Optimisers.freeze!(optim.layers[1].layers[1])
end