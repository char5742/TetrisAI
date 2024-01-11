
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

function prioritized_sample!(m::Memory, batch_size; priority=1)::Vector{Tuple{Int,Experience}}
    lock(m.lock) do
        index_list = StatsBase.sample(1:m.capacity, StatsBase.Weights([d.temporal_difference^priority for d in m.data]), batch_size, replace=false)
        return [(i, m.data[i]) for i in index_list]
    end
end

function update_temporal_difference(m::Memory, new_temporal_difference_list::Vector{Tuple{Int,Float32}})
    lock(m.lock) do
        for (i, td) in new_temporal_difference_list
            m.data[i].temporal_difference = td
        end
    end
end

function sum_td(m::Memory)::Float64
    lock(m.lock) do
        sum([e.temporal_difference for e in m.data])
    end
end