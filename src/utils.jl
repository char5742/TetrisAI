function print_matrix(x::Matrix)
    show(IOContext(stdout, :limit => false), "text/plain", Int.(x))
end
function sleep30fps(start_time)
    diff = (1 / 30) - (time_ns() - start_time) / 1e9
    if diff < 0
        return false
    else
        mysleep(diff)
        return true
    end
end