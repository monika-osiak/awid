using OptimizationMethods.Methods

using BenchmarkTools
using Logging
using .Utils

BenchmarkTools.DEFAULT_PARAMETERS.samples = 100

mom = zeros(length(x))
momentum = Methods.Momentum(l_rate, 0.01, mom)

info = @benchmark pts, errs, i = optimalize(f, ∇f, x, momentum, err, iters)
add_test("Momentum",info)
@info "Momentum: done"

gd = Methods.GradientDescent(l_rate)
info = @benchmark pts, errs, i = optimalize(f, ∇f, x, gd, err, iters)
add_test("GradientDescent",info)
@info "GradientDescent: done"

# bfgs = Methods.BFGS(length(x))
# @info "BFGS: $bfgs"
# info = @benchmark pts, errs, i = optimalize(f, ∇f, x, bfgs, err, iters)
# add_test("BFGS",info)
# @info "BFGS: done"
# save_test()

lbfgs = Methods.LBFGS()
for i = 1:3
    init!(lbfgs, i)
    @info "L-BFGS-$i: $lbfgs"
    local info = @benchmark pts, errs, i = optimalize(f, ∇f, x, lbfgs, err, iters)
    add_test("L-BFGS-$i", info)
    @info "L-BFGS-$i: done"
end

save_test()