module Functions

using Zygote
using Statistics

# potensial problem
function f_rosenbrock(x::T)::Q where T <: AbstractArray{Q} where Q <: Float64
    100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
end
function ∇f_rosenbrock(x::T)::T  where T <: AbstractArray{Q} where Q <: Float64
    f_rosenbrock'(x)
end
f_ackley(x::Vector{Float64}, a=20.0,b=0.2,c=2.0 * pi)::Float64 = -a * exp(-b * sqrt(mean(x.^2.0))) - exp(mean(cos.(c .* x))) + a + exp(1.0)
∇f_ackley(x::Vector{Float64})::Vector{Float64} = f_ackley'(x)

export f_rosenbrock, ∇f_rosenbrock, f_ackley, ∇f_ackley
end