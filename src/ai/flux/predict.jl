"予測部分"


function predict(model,
    (board, minopos, combo, btb, tspin)::Tuple{Array{T,4},Array{T,4},Array{T,2},Array{T,2},Array{T,2}}
) where {T}
    model((board, minopos, combo, btb, tspin) |> gpu) |> cpu
end

function predict(model,
    (board, minopos, combo, btb, tspin, holdnext)::Tuple{Array{T,4},Array{T,4},Array{T,2},Array{T,2},Array{T,2},Array{T,3}}
) where {T}
    model(((board, minopos), combo, btb, tspin, holdnext) |> gpu) |> cpu
end

function predict(model,
    (board, minopos, combo, btb, tspin, holdnext)::Tuple{Array{T,4},Array{T,4},Array{T,2},Array{T,2},Array{T,2},Array{T,3}}
) where {T}
    model((board, minopos, combo, btb, tspin, holdnext) |> gpu) |> cpu
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