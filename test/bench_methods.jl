using OptimizationMethods.Functions
# import optimization_methods:Functions
using OptimizationMethods.Methods

using BenchmarkTools
using Random

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

f = f_rosenbrock
∇f = ∇f_rosenbrock 
n = 2
x = zeros(n)
my_seed = 1620689075631
Random.seed!(my_seed)
rand!(x)
println(x)

function pp(b, file="stats")
    io = open(file, "w+");
    show(io, MIME"text/plain"(), b)
end

momentum = Momentum(0.00000000000001, 0.01, length(x))
info = @benchmark pts, err, i = optimalize(f, ∇f, x, momentum, 0.01, 100)
pp(info)
println("*")

bfgs = BFGS(length(x))
info = @benchmark pts, err, i = optimalize(f, ∇f, x, bfgs, 0.0001, 100)
pp(info)
println("*")

lbfgs = LBFGS()
for i = 1:3
    init!(lbfgs, i)
    info = @benchmark pts, err, i = optimalize(f, ∇f, x, lbfgs, 0.0001, 100)
    pp(info)
    println("*")
end