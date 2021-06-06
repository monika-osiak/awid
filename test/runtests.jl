using Optim
using OptimizationMethods.Functions
using OptimizationMethods.Methods
using Logging
using Random
using Test
include("./utils.jl")
using .Utils

const test_set = Set([
    "moment",
    "gd",
    "bfgs",
    "lbfgs"
])
const benchmark_varialbe = "TEST_PERFORMANCE"
const my_seed = 1620689075631
Random.seed!(my_seed)
const f = f_ackley # f_rosenbrock
const ∇f =  ∇f_ackley # ∇f_rosenbrock 
const f_name = "ackley"
const n = 4
const iters = 100
const err = 0.0001
const l_rate = 0.00001
const x = zeros(n)
const logger = SimpleLogger(stdout, Logging.Debug)

rand!(x)
@info "Chosen point: $x"

if "optim_bfgs" in test_set
    l = optimize(f, ∇f, x, method=Optim.BFGS(), iterations=iters;inplace=false)
    opti_res = Optim.minimizer(l)
    @info "Optim answer $opti_res"
end

if "moment" in test_set
    # with_logger(logger) do
    mom = zeros(length(x))
    momentum = Methods.Momentum(l_rate, 0.01, mom)
    pts, errs, i = optimalize(f, ∇f, x, momentum, eps(), iters)
    @test f_name != "rosenbrock" || pts[end] == [0.3122846156075936, 0.6981553088628102]
    @info "Momentum"
    @info pts[end]
    @info errs[end], i
    # end
end

if "gd" in test_set
    # with_logger(logger) do
    mom = zeros(length(x))
    gd = Methods.GradientDescent(l_rate)
    pts, errs, i = optimalize(f, ∇f, x, gd, err, iters)
    @test f_name != "rosenbrock" ||  pts[end] ==  [0.3115209459667753, 0.6993648285144705]
    @info "GradientDescent"
    @info pts[end]
    @info errs[end], i
    # end
end

# in order to force max iterations
if "bfgs" in test_set
    bfgs = Methods.BFGS(length(x))
    # with_logger(logger) do
    pts, errs, i = optimalize(f, ∇f, x, bfgs, eps(), iters)
    bfgs = Methods.BFGS(length(x))
    @test f_name != "rosenbrock" || pts[end] == [1.0000000002538125, 1.0000000005044984]
    @info "BFGS"
    @info pts[end]
    @info errs[end], i
    # end
end

if "lbfgs" in test_set
    lbfgs_res = [
    [0.5262262899257493, 0.27691410820901874],
    [1.000000000000055, 0.9999999999989565],
    [1.0000000000474711, 0.9999999998824036]
    ]

    for i = 1:3
        lbfgs = Methods.LBFGS()
        init!(lbfgs, i)
    # with_logger(logger) do
        local pts, errs, iter = optimalize(f, ∇f, x, lbfgs, eps(), iters)
        @test f_name != "rosenbrock" || pts[end] == lbfgs_res[i]
        @info "L-BFGS history: $i"
        @info "iterations $iter"
        @info pts[end]
        @info errs[end]
    # end
    end
end
if haskey(ENV, benchmark_varialbe)
    add_metadata("function", f_name)
    add_metadata("dims", n)
    add_metadata("max_iters", iters)
    add_metadata("err_tolerance", err)
    add_metadata("lrate", l_rate)
    add_metadata("xs", x)

    include("./bench_methods.jl")
    # include("./bench_functions.jl")
end