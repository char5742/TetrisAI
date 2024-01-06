using HTTP
include("../core/TetrisAICore.jl")
using .TetrisAICore
import Serialization
include("config.jl")
include("../lib/compress.jl")

include("memory.jl")
include("param.jl")

const SSL_CERT_FILE = "./cert.pem"
const SSL_KEY_FILE = "./key.pem"

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

# HTTP.listen! and HTTP.serve! are the non-blocking versions of HTTP.listen/HTTP.serve
HTTP.serve("0.0.0.0", 10513) do request::HTTP.Request
    try
        t = Threads.@spawn router(request)
        return fetch(t)
    catch e
        return HTTP.Response(400, "Error: $e")
    end
end


# HTTP.serve! returns an `HTTP.Server` object that we can close manually
# close(server)