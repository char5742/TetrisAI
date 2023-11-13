
memory = Memory(Config.batchsize * Config.memoryscale)

"ゲーム生成速度計測器"
module GenerateSpeedMeasurement
"１秒間あたりのゲーム経験生成速度"
const generate_speed = Ref{Float64}(0.0)
"最後にゲーム経験生成速度を計測したときのmemory.index"
const last_index = Ref{Int64}(1)
"最後にゲーム経験生成速度を計測した時刻"
const last_time = Ref{Float64}(time())
end



function memory_route(request::HTTP.Request)
    target = request.target
    method = request.method
    if contains(target, "/memory")
        target = replace(target, "/memory" => "")
        target == "/priority" && return update_priority(memory, request.body)
        target == "/index" && return get_current_memory_index(memory)
        target == "/generate_speed" && return get_generate_speed(memory)
        method == "GET" && return get_minibatch(memory)
        method == "POST" && return add_exp(memory, request.body)
    end
    nothing
end

function get_minibatch(memory::Memory)
    batch = prioritized_sample!(memory, 16;priority=0.6)
    buffer = IOBuffer()
    serialize(buffer, batch)
    compressed = transcode(ZstdCompressor, take!(buffer))
    HTTP.Response(200, compressed)
end

function add_exp(memory::Memory, body)
    stream = ZstdDecompressorStream(IOBuffer(body))
    experience = deserialize(stream)
    close(stream)
    add!(memory, experience)
    @info "現在のメモリindex: " * string(memory.index)
    HTTP.Response(200, "OK")
end


function update_priority(memory::Memory, body)
    new_temporal_difference_list = deserialize(IOBuffer(body))
    update_temporal_difference(memory, new_temporal_difference_list)
    HTTP.Response(200, "OK")
end

"メモリの現在のindexを取得する"
function get_current_memory_index(memory::Memory)
    HTTP.Response(200, string(memory.index))
end

"ゲーム経験の生成速度を計測する"
function measure_generate_speed(memory::Memory)
    now = time()
    generated_experience_count = memory.index - GenerateSpeedMeasurement.last_index[]
    GenerateSpeedMeasurement.generate_speed[] = generated_experience_count / (now - GenerateSpeedMeasurement.last_time[])
    GenerateSpeedMeasurement.last_time[] = now
    GenerateSpeedMeasurement.last_index[] = memory.index
end

"ゲーム経験の生成速度を取得する"
function get_generate_speed(memory::Memory)
    # 本当は一定期間ごとに取得するようにしたい
    # 現在の実装では短い期間で取得すると正確な値が得られない
    measure_generate_speed(memory)
    HTTP.Response(200, string(GenerateSpeedMeasurement.generate_speed[]))
end