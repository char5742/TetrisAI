

function initialize_brain()
    model, ps, st = create_model(Config.kernel_size, Config.res_blocks, Config.kernel_size; use_gpu=false)
    t_ps, t_st = ps, st
    if Config.load_params
        _, ps, st = loadmodel("mainmodel.jld2")
        _, t_ps, t_st = loadmodel("targetmodel.jld2")
    end
    Brain(Model(model, ps, st), Model(model, t_ps, t_st))
end

brain = initialize_brain()

function param_route(request::HTTP.Request)
    target = request.target
    method = request.method
    if contains(target, "/param")
        target = replace(target, "/param" => "")
        if (target == "/mainmodel")
            method == "GET" && return get_mainmodel_param(brain)
            method == "POST" && return set_mainmodel_param(brain, request.body)
        end
        if (target == "/targetmodel")
            method == "GET" && return get_targetmodel_param(brain)
            method == "POST" && return set_targetmodel_param(brain, request.body)
        end
    end
    nothing
end

function get_mainmodel_param(brain::Brain)
    data = serialize((brain.main_model.ps, brain.main_model.st))
    HTTP.Response(200, data)
end

function get_targetmodel_param(brain::Brain)
    data = serialize((brain.target_model.ps, brain.target_model.st))
    HTTP.Response(200, data)
end

function set_mainmodel_param(brain::Brain, body)
    ps, st = deserialize(body)
    brain.main_model.ps = ps
    brain.main_model.st = st
    savemodel("mainmodel.jld2", brain.main_model)
    HTTP.Response(200, "OK")
end

function set_targetmodel_param(brain::Brain, body)
    ps, st = deserialize(body)
    brain.target_model.ps = ps
    brain.target_model.st = st
    savemodel("targetmodel.jld2", brain.target_model)
    HTTP.Response(200, "OK")
    GC.gc(true)
end