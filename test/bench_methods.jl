include("functions.jl")
include("methods.jl")
using .Functions
using .Methids

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


momentum = Momentum(0.00000000000001, 0.01, length(x))
@benchmark pts, err, i = optimalize(f, ∇f, x, momentum, 0.01, 100)

bfgs = BFGS(length(x))
info = @benchmark pts, err, i = optimalize(f, ∇f, x, bfgs, 0.0001, 100)

for i = 1:3
    lbfgs = LBFGS(); init!(lbfgs, i)
    @benchmark pts, err, i = optimalize(f_rosenbrock, ∇f_rosenbrock, x, lbfgs, 0.0001, 100)
end