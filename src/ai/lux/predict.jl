"予測部分"


function predict(model,
    (board, minopos, combo, btb, tspin)::Tuple{Array{T,4},Array{T,4},Array{T,2},Array{T,2},Array{T,2}}
) where {T}
    if (isa(model.ps[1][1].weight, CUDA.CuArray))
        model.model((board, minopos, combo, btb, tspin) |> gpu, model.ps, model.st) |> cpu
    else
        model.model((board, minopos, combo, btb, tspin), model.ps, model.st) |> cpu
    end
end


function predict(model,
    (board, minopos, combo, btb, tspin, holdnext)::Tuple{Array{T,4},Array{T,4},Array{T,2},Array{T,2},Array{T,2},Array{T,3}}
) where {T}
    if (isa(model.ps[1][1].weight, CUDA.CuArray))
        data = (board, minopos, combo, btb, tspin, holdnext) |> gpu
        response, _ = model.model(data, model.ps, model.st) |> cpu
        data = nothing
        return response
    else
        response, _ = model.model((board, minopos, combo, btb, tspin, holdnext), model.ps, model.st) |> cpu
        response
    end
end


function vector2array(v::Vector{Matrix{T}})::Array{T,4} where {T}
    reduce((x, y) -> cat(x, y, dims=4), reshape(n, 24, 10, 1, 1) for n in v)
end

function vector2array(v::Matrix{T})::Array{T,4} where {T}
    reshape(v, 24, 10, 1, 1)
end

function vector2array(v::Vector{T})::Array{T,2} where {T}
    reduce((x, y) -> cat(x, y, dims=2), [n;;] for n in v)
end