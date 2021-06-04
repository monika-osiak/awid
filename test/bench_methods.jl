using OptimizationMethods.Methods

using BenchmarkTools
using .Utils

add_metadata("function", "rosenbrock")

momentum = Momentum(0.00000000000001, 0.01, length(x))
info = @benchmark pts, errs, i = optimalize(f, ∇f, x, momentum, err, iters)
add_test("Momentum",info)

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