using OptimizationMethods.Methods

using BenchmarkTools
using .Utils

mom = zeros(length(x))
momentum = Methods.Momentum(l_rate, 0.01, mom)

info = @benchmark pts, errs, i = optimalize(f, ∇f, x, momentum, err, iters)
add_test("Momentum",info)

gd = Methods.GradientDescent(l_rate)
info = @benchmark pts, errs, i = optimalize(f, ∇f, x, gd, err, iters)
add_test("GradientDescent",info)

bfgs = Methods.BFGS(length(x))
info = @benchmark pts, errs, i = optimalize(f, ∇f, x, bfgs, err, iters)
add_test("BFGS",info)

lbfgs = Methods.LBFGS()
for i = 1:3
    init!(lbfgs, i)
    local info = @benchmark pts, errs, i = optimalize(f, ∇f, x, lbfgs, err, iters)
    add_test("L-BFGS-$(i)", info)
end

save_test()