using Functions

using BenchmarkTools
using Random
using Test

n = 4
x = zeros(n)
my_seed = 1620689075631
Random.seed!(my_seed)
rand!(x)
println(x)

pp(b) = show(stdout, MIME"text/plain"(), b)

info = @benchmark f_rosenbrock(x)
pp(info)
println("*")

info = @benchmark ∇f_rosenbrock(x)
pp(info)

info = @benchmark f_ackley(x)
pp(info)

info = @benchmark ∇f_ackley(x)
pp(info)