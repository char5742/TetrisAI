using Plots

# output/log.txt に書き込まれたデータを100区間の移動平均でプロットする

function read_csv(file_path::String, delimiter::Char=',', header::Bool=true)
    data = Vector{Vector{Float64}}()
    header_line = header ? [] : nothing

    open(file_path, "r") do file
        for (line_num, line) in enumerate(eachline(file))
            if header && line_num == 1
                header_line = split(line, delimiter)
                continue
            end
            row = split(line, delimiter) .|> x -> parse(Float64, x)
            push!(data, row)
        end
    end

    return header ? (header_line, data) : data
end

data = read_csv("log.txt", ',', false)
data = [row[2] for row in data]

# Moving average
window_size = 100
moving_average = [(sum(data[i:i+window_size-1]) / window_size) for i in 1:length(data)-window_size+1]

# Moving variance
moving_variance = [(sum((data[i:i+window_size-1] .- moving_average[i]).^2) / window_size) for i in 1:length(data)-window_size+1]

# Plot
plot(layout=(2,1), size=(800, 600))
plot!(data, lw=0.5, label="Raw Scores", subplot=1)
plot!(moving_average, lw=2, label="Moving Average", subplot=1)
plot!(moving_variance, lw=2, label="Moving Variance", subplot=2)
xlabel!("Episode")
ylabel!("Score", subplot=1)
ylabel!("Variance", subplot=2)
title!("Learning Progress Visualization", subplot=1)