using OptimizationMethods.Functions
using Random
using Test

benchmark_varialbe = "TEST_PERFORMANCE"
include("./utils.jl")

my_seed = 1620689075631
Random.seed!(my_seed)
f = f_rosenbrock
∇f = ∇f_rosenbrock 
n = 2
x = zeros(n)
iters = 100
err = 0.0001
rand!(x)

if haskey(ENV, benchmark_varialbe)
    include("./bench_methods.jl")
    include("./bench_functions.jl")

end
include("./optima_test.jl")