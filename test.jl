# using Distributed


# addprocs(["fshuu@35.225.182.238 35.225.182.238:8888"]; tunnel=true, sshflags=["-i", "C:/Users/fshuu/.ssh/tetris-ai"], exename="/home/fshuu/workspace/julia-1.9.0/bin/julia", dir="/home/fshuu/workspace/TetrisAI", exeflags="--project=$(Base.active_project())")
# @everywhere mutable struct Model
#     a::Int
#     b::Int
# end

# @everywhere global_model = Model(1, 2)

# @everywhere function f()
#     sleep(6)
#     global global_model = Model(3, 4)
# end

# @everywhere function g()
#     sleep(10)
#     @show global_model
# end
# @sync begin
#     @spawn f()
#     @spawn g()

# end

ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
using Flux
using CUDA

model = Chain(Dense(10, 4096), Dense(4096, 1)) |> gpu

for _ in 1:10
    a = rand(10, 100) |> gpu
    model(a)
    CUDA.unsafe_free!(a)
    CUDA.memory_status()
end
CUDA.memory_status()