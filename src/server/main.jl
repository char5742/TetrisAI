using HTTP
include("../core/TetrisAICore.jl")
using .TetrisAICore
import Serialization
include("config.jl")
include("../lib/compress.jl")

include("memory.jl")
include("param.jl")

const PORT = 10513

function router(request::HTTP.Request)
    r = config_route(request)
    if !isnothing(r)
        return r
    end
    r = memory_route(request)
    if !isnothing(r)
        return r
    end
    r = param_route(request)
    if !isnothing(r)
        return r
    end
    return HTTP.Response(404, "Not Found") 
end

HTTP.serve("0.0.0.0", PORT) do request::HTTP.Request
    try
        t = Threads.@spawn router(request)
        return fetch(t)
    catch e
        return HTTP.Response(400, "Error: $e")
    end
end
