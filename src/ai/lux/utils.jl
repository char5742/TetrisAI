
function loadmodel(path::String)
    data = load(path)
    data["model"], data["ps"], data["st"]
end

function savemodel(path::String, model)
    jldsave(path; model=model.model, ps=model.ps |> cpu, st=model.st |> cpu)
end

function freeze_boardnet!(optim)
    Optimisers.freeze!(optim[1][1])
end