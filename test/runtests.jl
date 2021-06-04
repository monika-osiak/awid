using Optim
using OptimizationMethods.Functions
using OptimizationMethods.Methods
using Logging
using Random
using Test

const benchmark_varialbe = "TEST_PERFORMANCE"
include("./utils.jl")

const test_set = Set([
    "moment"
    # "bfgs",
    # "lbfgs"
])

const my_seed = 1620689075631
Random.seed!(my_seed)
const f = f_rosenbrock
const ∇f = ∇f_rosenbrock 
const n = 2
const iters = 100
const err = 0.0001
const x = zeros(n)
rand!(x)

@info "Chosen point: $x"

if "omtim_bfgs" in test_set
    l = optimize(f, ∇f, x, method=Optim.BFGS(), iterations=iters;inplace=false)
    opti_res = Optim.minimizer(l)
    @info "Optim answer $opti_res"
end

if "moment" in test_set
    mom::Vector{Float64} = zeros(length(x))
    momentum = Momentum(0.00000000000001, 0.01, mom)
    pts, errs, i = optimalize(f, ∇f, x, momentum, eps(), iters)
    # @test pts[end] == [1.0000000002538125, 1.0000000005044984]
    @info pts[end]
    @info errs[end], i
end

# in order to force max iterations
if "bfgs" in test_set
    bfgs = Methods.BFGS(length(x))
    pts, errs, i = optimalize(f, ∇f, x, bfgs, eps(), iters)
    @test pts[end] == [1.0000000002538125, 1.0000000005044984]
    @info pts[end]
    @info errs[end], i
end

if "lbfgs" in test_set
    const lbfgs_res = [
    [0.5262262899257493, 0.27691410820901874],
    [1.000000000000055, 0.9999999999989565],
    [1.0000000000474711, 0.9999999998824036]
    ]

    for i = 1:3
        lbfgs = Methods.LBFGS()
        init!(lbfgs, i)
        local pts, errs, iter = optimalize(f, ∇f, x, lbfgs, eps(), iters)
        @test pts[end] == lbfgs_res[i]
        @info "L-BFGS history: $i"
        @info "iterations $iter"
        @info pts[end]
        @info errs[end]
    end
end
if haskey(ENV, benchmark_varialbe)
    include("./bench_methods.jl")
    include("./bench_functions.jl")
end