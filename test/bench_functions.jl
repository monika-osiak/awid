using OptimizationMethods.Functions

using BenchmarkTools
using Random
using Test

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