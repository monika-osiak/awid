
using Test

benchmark_varialbe = "TEST_PERFORMANCE"
include("./utils.jl")

if haskey(ENV, benchmark_varialbe)
    include("./bench_methods.jl")
    include("./bench_functions.jl")
end
