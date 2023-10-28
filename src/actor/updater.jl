include("../core/TetrisAICore.jl")
using .TetrisAICore
using Serialization
using HTTP
using CodecZstd
using JLD2
const root = "http://127.0.0.1:10513"
const paramserver = "$root/param"
include("config.jl")

"""
Paramサーバーからパラメータを取得する
return ps, st
"""
function get_model_params(name::String)
    res = HTTP.request("GET", "$paramserver/$name")
    if res.status == 200
        buffer = IOBuffer(res.body)
        stream = ZstdDecompressorStream(buffer)
        ps, st = deserialize(stream)
    else
        throw("Paramサーバーからparamを取得できませんでした")
    end
    return ps, st
end

"""
分散学習時にパラメーター送受信のトラフィックコストを削減するために、
一定間隔置きにサーバーから取得したパラメーターをローカルに保存する
"""
function main()
    # 60秒に一回パラメータを取得する
    while true
        touch("model/model.lock")
        try
            ps, st = get_model_params("mainmodel")
            jldsave("model/mainmodel.jld2"; ps=ps, st=st)
            ps, st = get_model_params("targetmodel")
            jldsave("model/targetmodel.jld2"; ps=ps, st=st)
        catch
        finally
            rm("model/model.lock")
        end
        GC.gc()
        sleep(60)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
