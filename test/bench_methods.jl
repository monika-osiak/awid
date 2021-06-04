using OptimizationMethods.Functions
# import optimization_methods:Functions
using OptimizationMethods.Methods

using BenchmarkTools
using Random
using .Utils

function optimalize(f, ∇f, x₀, opt, e, i)
    pts = [x₀] # kolejne wektory x
    err = Float64[] # kolejne wartości f. straty
    p = 0
    while true
        prev = f(pts[end])
        push!(err, prev) # odłóż wynik funkcji dla najnowszego wektora x (miara błędu)
        @debug   "Iteracja p= $(p)"
        @debug   "Wektor x: $(pts[end])"
        @debug   "Error: $(err[end])"
        if prev < e || isnan(prev)  || p > i 
            break
        end
        # if length(pts) > 1 && pts[end] == pts[end-1]
        # break # when we stop to progress
        push!(pts, step!(opt, f, ∇f, pts[end]))
        p += 1
    end
    
    pts, err, p
end

my_seed = 1620689075631
Random.seed!(my_seed)
f = f_rosenbrock
∇f = ∇f_rosenbrock 
n = 2
x = zeros(n)
iters = 100
err = 0.0001
rand!(x)

file = "stats"
io = open(file, "w+");

add_metadata("function", "rosenbrock")

momentum = Momentum(0.00000000000001, 0.01, length(x))
info = @benchmark pts, err, i = optimalize(f, ∇f, x, momentum, err, iters)
add_test("Momentum",info)

bfgs = BFGS(length(x))
info = @benchmark pts, err, i = optimalize(f, ∇f, x, bfgs, err, iters)
add_test("BFGS",info)

lbfgs = LBFGS()
for i = 1:3
    init!(lbfgs, i)
    local info = @benchmark pts, err, i = optimalize(f, ∇f, x, lbfgs, err, iters)
    add_test("L-BFGS-$(i)", info)
end

save_test()