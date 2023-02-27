import StatsBase

mutable struct Memory
    capacity::Int
    index::Int
    data::Vector{Experience}
    lock::ReentrantLock
    function Memory(capacity::Int)
        data = Vector(undef, capacity)
        new(capacity, 1, data, ReentrantLock())
    end
end

function add!(m::Memory, experience::Experience)
    lock(m.lock) do
        m.data[(m.index-1)%m.capacity+1] = experience
        m.index += 1
    end
end

function sample(m::Memory, batch_size)::Vector{Experience}
    lock(m.lock) do
        rand(m.data, batch_size)
    end
end

function prioritized_sample!(m::Memory, batch_size; priority=1)::Vector{Experience}
    lock(m.lock) do
        index_list = StatsBase.sample(1:m.capacity, StatsBase.Weights([d.temporal_difference^priority for d in m.data]), batch_size, replace=false)
        for index in index_list
            m.data[index].temporal_difference *= 0.9
        end
        return [m.data[i] for i in index_list]
    end
end

function sum_td(m::Memory)::Float64
    lock(m.lock) do
        sum([e.temporal_difference for e in m.data])
    end
end