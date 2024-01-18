include("../core/TetrisAICore.jl")
using .TetrisAICore
using HTTP
using JLD2

const ROOT = "http://127.0.0.1:10513"
const PARAMSERVER = "$ROOT/param"
const UPDATE_INTERVAL = 60.0

include("../lib/compress.jl")
include("config.jl")

"""
Paramサーバーからパラメータを取得する
return ps, st
"""
function get_model_params(name::String)
    res = HTTP.request("GET", "$PARAMSERVER/$name")
    if res.status == 200
        ps, st = deserialize(res.body)
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
        try
            try
                ps, st = get_model_params("mainmodel")
                jldsave("model/temp_mainmodel.jld2"; ps=ps, st=st)
                ps, st = get_model_params("targetmodel")
                jldsave("model/temp_targetmodel.jld2"; ps=ps, st=st)
            catch
            end
            touch("model/model.lock")
            try
                mv("model/temp_mainmodel.jld2", "model/mainmodel.jld2"; force=true)
                mv("model/temp_targetmodel.jld2", "model/targetmodel.jld2"; force=true)
            catch
            finally
                rm("model/model.lock")
            end
            GC.gc()
            sleep(UPDATE_INTERVAL)
        catch e
            @error e
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
