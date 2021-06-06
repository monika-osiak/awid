module Methods
using FastClosures
using LinearAlgebra
using ExportAll

export Momentum, BFGS, LBFGS, step!, init!, DescentMethod, optimalize, GradientDescent

function optimalize(f, ∇f, x₀, opt, e, i::Int64)
    pts = [x₀] # kolejne wektory x
    err = Vector{Float64}(undef, i) # kolejne wartości f. straty
    # minimalizacja realokacji
    p = 0
    while true
        p += 1
        prev = f(pts[end])
        # push!(err, prev) # odłóż wynik funkcji dla najnowszego wektora x (miara błędu)
        if prev < e || isnan(prev)  || p > i 
            break
        end
        if length(pts) > 1 && pts[end] == pts[end - 1]
            break # when we stop to progress
        end
        err[p] = prev
        @debug   "Iteracja p= $(p)"
        @debug   "Wektor x: $(pts[p])"
        @debug   "Error: $(err[p])"
        push!(pts, step!(opt, f, ∇f, pts[end]))
    end
    
    pts, err, p
end

abstract type DescentMethod end

struct GradientDescent <: DescentMethod 
    α # learning rate
end

function step!(M::GradientDescent, f, ∇f, θ)
    α, g = M.α, ∇f(θ)
    return θ - α * g
end

mutable struct Momentum <: DescentMethod
    α # learning rate
    β # momentum decay
    v# momentum
end


Momentum(α, β, n::Int64) = Momentum(α, β, zeros(n))
# Momentum(α, β, n) = Momentum(α, β, n)


function step!(M::Momentum, f, ∇f, x)
    α, β, v, g = M.α, M.β, M.v, ∇f(x)
    @debug "Gradient: $g"
    @debug M
    @debug "Parameters: ($α, $β, $v, $g)"
    v[:]  = β * v .- α * g
    return x + v
end

mutable struct BFGS <: DescentMethod
    Q
end

BFGS(n::Int64) = BFGS(Matrix(1.0I, n, n))

function strong_backtracking(f, ∇, x, d; α=1.0, β=1e-4, σ=0.1)
    y0, g0 = f(x), ∇(x) ⋅ d
    y_prev, α_prev  = NaN, 0.0
    αlo, αhi = NaN, NaN
  # bracket phase
    # @debug "Before first inf loop"
    while true
        y = f(x + α * d)
        if y > y0 + β * α * g0 || (!isnan(y_prev) && y ≥ y_prev)
            αlo, αhi = α_prev, α
            break
        end
        g = ∇(x + α * d) ⋅ d
        if abs(g) ≤ -σ * g0
            return α
        elseif g ≥ 0
      αlo, αhi  = α, α_prev
      break
        end
        y_prev, α_prev, α  = y, α, 2α
    end
  # zoom phase
    ylo = f(x + αlo * d)
    @debug "Before second inf loop"
    α_old = NaN
    while true
        @debug "Start loop"
        α = (αlo + αhi) / 2
        if α == α_old
            return α
        else
            α_old = α
        end
        @debug "α: $α"
        y = f(x + α * d)
        @debug "y: $y"
        @debug "y > y0 + β * α * g0 $(y > y0 + β * α * g0)"
        @debug "y ≥ ylo $(y ≥ ylo)"
        if y > y0 + β * α * g0 || y ≥ ylo
            @debug "first case"
            @debug "ylo $ylo"
            αhi  = α
            @debug "ahi $αhi"
        else
            @debug "second case"
            g  = ∇(x + α * d) ⋅ d
            @debug "g: $g"
            @debug "abs(g): $(abs(g))"
            @debug "-σ * g0 : $(-σ * g0)"
            if abs(g) ≤ -σ * g0
                @debug "second case inner first case"
                return α
            elseif g * (αhi - αlo) ≥ 0
                @debug "second case inner second case"
                αhi = αlo
            end
            αlo = α
            @debug "α: $α"
            @debug "αlo: $αlo"
            @debug "nether case"
        end
    end
end

    function step!(M::BFGS, f, ∇f, x)
    if f(x) ≈ 0.0
        return x
    end

    Q, g = M.Q, ∇f(x)
    # @debug "Q: $Q"
    # @debug "g: $g"
    α = strong_backtracking(f, ∇f, x, -Q * g)
    # @debug "α: $α"
    x′ = x + α * (-Q * g)
    # @debug "x': $x′"
    g′ = ∇f(x′)
    # @debug "g: $g′"
    δ = x′ - x
    # @debug "δ: $δ"
    γ = g′ - g
    # @debug "γ: $γ"
    Q[:] = Q - (δ * γ' * Q + Q * γ * δ') / (δ' * γ) + (1 + (γ' * Q * γ) / (δ' * γ))[1] * (δ * δ') / (δ' * γ)
    # @debug "new Q: $Q"
    return x′
end

    mutable struct LBFGS
    m::Int64
    δs
    γs
    qs
    LBFGS() = new()
end

    function init!(M::LBFGS, m)::LBFGS
    M.m = m
    M.δs = [] 
    M.γs = [] 
    M.qs = []
    return M
end

    function step!(M::LBFGS, f, ∇f, θ)
    δs, γs, qs = M.δs, M.γs, M.qs 
    m::Int64, g = length(δs), ∇f(θ)
    d = -g # kierunek
    # if isnan(g)
        # there is no dericative at θ
        # we can't progress any further
        # posibli move in random direction
        # return θ || (!isnan(y_prev) && y ≥ y_prev)
    if m > 0 
        q = g
        for i in m:-1:1
            qs[i] = copy(q)
            q -= (δs[i] ⋅ q) / (γs[i] ⋅ δs[i]) * γs[i]
        end
        z = (γs[m] .* δs[m] .* q) / (γs[m] ⋅ γs[m]) 
        for i in 1:+1:m
            z += δs[i] * (δs[i] ⋅ qs[i] - γs[i] ⋅ z) / (γs[i] ⋅ δs[i]) 
        end
        d = -z; # rekonstrukcja kierunku
    end
    φ = α -> f(θ + α * d)
    φ′ = α -> ∇f(θ + α * d) ⋅ d 
    α = line_search(φ, φ′, d)
    @debug "Point: $θ"
    @debug "line_search: $α"
    θ′ = θ + α * d
    @debug "New Point: $θ′"
    g′ = ∇f(θ′) # nowy wektor
    δ = θ′ - θ
    γ = g′ - g
    push!(δs, δ);
    push!(γs, γ);
    push!(qs, zero(θ)) 
   
    while length(δs) > M.m
        popfirst!(δs); popfirst!(γs); popfirst!(qs) 
    end
    return θ′ 
end

    function zoom(φ, φ′, αlo, αhi, c1=1e-4, c2=0.1, jmax=1000)
    φ′0 = φ′(0.0) 
    for j = 1:jmax
        αj = 0.5(αlo + αhi) # bisection 
        φαj = φ(αj)
        if φαj > φ(0.0) + c1 * αj * φ′0 || φαj ≥ φ(αlo)
            αhi = αj 
        else
            φ′αj = φ′(αj)
            if abs(φ′αj) ≤ -c2 * φ′0
                return αj
            end
            if φ′αj * (αhi - αlo) ≥ 0.0 
                αhi = αlo
            end
            αlo = αj 
        end
    end
    return 0.5(αlo + αhi) 
end

    function line_search(φ, φ′, d, c1=1e-4, c2=0.1, ρ=0.1, αmax=100., jmax=1000)
    αi, αj = 0.0, 1.0
    φαi, φ0, φ′0 = φ(αi), φ(0.0), φ′(0.0) 
    for j = 1:jmax
        φαj = φ(αj)
        if φαj > φ0 + c1 * αj * φ′0 || (φαj ≥ φαi && j > 1)
            return zoom(φ, φ′, αi, αj)
        end
        φ′αj = φ′(αj)
        if abs(φ′αj) ≤ -c2 * φ′0
            return αj 
        end
        if φ′αj ≥ 0.0
            return zoom(φ, φ′, αj, αi)
        end
        αi, αj = αj, ρ * αj + (1.0 - ρ) * αmax
        φαi = φαj 
    end
    return αj 
end

    @exportAll
end
