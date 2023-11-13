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
        data = (board, minopos, combo, btb, tspin, holdnext)
        response = batch_predict(model, data, 128)
        data = nothing
        return response
    else
        response, _ = model.model((board, minopos, combo, btb, tspin, holdnext), model.ps, model.st) |> cpu
        response
    end
end

function batch_predict(model, data, batch_size)
    # Unpack the data tuple
    board_input_prev, minopos, combo_input, back_to_back, tspin, mino_list = data

    n = size(board_input_prev, 4)  # Assuming data is in the form of (24, 10, 1, N)
    num_batches = ceil(Int, n / batch_size)
    responses = []

    for i in 1:num_batches
        start_idx = (i - 1) * batch_size + 1
        end_idx = min(i * batch_size, n)

        # Split the data into batches
        board_input_prev_batch = board_input_prev[:, :, :, start_idx:end_idx]
        minopos_batch = minopos[:, :, :, start_idx:end_idx] 
        combo_input_batch = combo_input[:, start_idx:end_idx] 
        back_to_back_batch = back_to_back[:, start_idx:end_idx] 
        tspin_batch = tspin[:, start_idx:end_idx] 
        mino_list_batch = mino_list[:, :, start_idx:end_idx] 

        # Send the batched data to the model
        data_batch = (board_input_prev_batch, minopos_batch, combo_input_batch, back_to_back_batch, tspin_batch, mino_list_batch)  |> gpu
        response_batch, _ = model.model(data_batch, model.ps, model.st)
        response_batch = response_batch |> cpu
        push!(responses, response_batch)
    end

    return hcat(responses...)
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
