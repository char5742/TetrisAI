ENV["GKS_ENCODING"] = "utf8"
using Plots
gr(fontfamily="Meiryo")

using Dates
using Statistics

Plots.default(show=true)
# CSVからデータを読み込む
data = open(ARGS[1], "r") do io
    lines = readlines(io)
    splited = [split(line, ',') for line in lines]
    try
    splited = [[DateTime(line[1], "yyyy/mm/dd HH:MM:SS"), parse(Int64, line[2])] for line in splited]
    catch
        splited = [[DateTime(line[1], "HH:MM:SS"), parse(Int64, line[2])] for line in splited]
    end
    m = hcat(splited...)
    permutedims(m, (2, 1))
end
# 経過時間に変換する
data[:, 1] = data[:, 1] .- data[1]

# CSVからデータを読み込む
# data = open("log.csv", "r") do io
#     lines = readlines(io)
#     splited = [split(line, ',') for line in lines]
#     splited = [[Time(line[9], "HH:MM:SS") , parse(Int64, line[10])] for line in splited]
#     m = hcat(splited...)
#     permutedims(m, (2,1))
# end


# 100区間移動平均を計算する
window_size = 100
moving_average = [i > window_size ? mean(data[i-window_size+1:i, 2]) : mean(data[1:i, 2]) for i in 1:size(data, 1)]

# プロットを作成する
yticks_label = [string(i ÷ 1000) * "K" for i in 0:1000:16000]
hour = 60 * 60 * 1000
xticks_label = [string(i ÷ hour) for i in 0:3hour:72hour]
plot(data[:, 1], moving_average, xlabel="",
 ylims=(0, 16000), yticks=(0:1000:16000, yticks_label), 
xticks=(0:3hour:72hour, xticks_label), xlims=(0, 72hour),
legend=false,
# size=(1000, 800)
)
xlabel!("学習時間 (Hours)")
ylabel!("獲得スコア     \n(Mean)")
savefig("plot.svg")
readline()