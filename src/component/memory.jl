import StatsBase

mutable struct Memory
    capacity::Int
    index::Int
    data::Vector{Experience}
    function Memory(capacity::Int)
        data = Vector(undef, capacity)
        new(capacity, 1, data)
    end
end

function add!(m::Memory, experience::Experience)
    m.data[(m.index-1)%m.capacity+1] = experience
    m.index += 1
end

function sample(m::Memory, batch_size)
    rand(m.data, batch_size)
end

function prioritized_sample(m::Memory, batch_size; priority=1)
    StatsBase.sample(m.data, StatsBase.Weights([d.temporal_difference^priority for d in m.data]), batch_size, replace=true)
end
