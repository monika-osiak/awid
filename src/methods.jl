module Methods

using LinearAlgebra

abstract type DescentMethod end

mutable struct Momentum <: DescentMethod
    α# ::Array{Float64} # learning rate
    β# ::Array{Float64} # momentum decay
    v# ::Float64# momentum
end


Momentum(α, β, n::Integer) = Momentum(α, β, zeros(n))

function step!(M::Momentum, f, ∇f, x) 
    α, β, v, g = M.α, M.β, M.v, ∇f(x)
    @debug ("Gradient: ", g)
    @debug "$(M)"
    v[:]  = β * v .- α * g
    return x + v
end

mutable struct BFGS <: DescentMethod
    Q
end

BFGS(n::Integer) = BFGS(Matrix(1.0I, n, n))

function strong_backtracking(f, ∇, x, d; α=1, β=1e-4, σ=0.1)
    y0, g0, y_prev, α_prev = f(x), ∇(x) ⋅ d, NaN, 0
    αlo, αhi = NaN, NaN
  # bracket phase
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
      αlo, αhi = α, α_prev
      break
        end
        y_prev, α_prev, α = y, α, 2α
    end
  # zoom phase
    ylo = f(x + αlo * d)
    while true
        α = (αlo + αhi) / 2
        y = f(x + α * d)
        if y > y0 + β * α * g0 || y ≥ ylo
            αhi = α
        else
            g = ∇(x + α * d) ⋅ d
            if abs(g) ≤ -σ * g0
                return α
            elseif g * (αhi - αlo) ≥ 0
        αhi = αlo
            end
            αlo = α
        end
    end
end

function step!(M::BFGS, f, ∇f, x)
    if f(x) ≈ 0.0
        return x
    end

    Q, g = M.Q, ∇f(x)
    α = strong_backtracking(f, ∇f, x, -Q * g)
    x′ = x + α * (-Q * g)
    g′ = ∇f(x′)
    δ = x′ - x
    γ = g′ - g
    Q[:] = Q - (δ * γ' * Q + Q * γ * δ') / (δ' * γ) + (1 + (γ' * Q * γ) / (δ' * γ))[1] * (δ * δ') / (δ' * γ)
    return x′
end

mutable struct LBFGS
    m::Float64
    δs# ::Array{Float64}
    γs# ::Array{Float64}
    qs# ::Array{Float64}
    LBFGS() = new()
end

function init!(M::LBFGS, m) 
    M.m = m
    M.δs = [] 
    M.γs = [] 
    M.qs = []
    return M
end

function step!(M::LBFGS, f, ∇f, θ) 
    δs, γs, qs = M.δs, M.γs, M.qs 
    m, g = length(δs), ∇f(θ)
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
    φ = α -> f(θ + α * d); φ′ = α -> ∇f(θ + α * d) ⋅ d 
    α = line_search(φ, φ′, d)
    θ′ = θ + α * d; g′ = ∇f(θ′) # nowy wektor
    δ = θ′ - θ;γ = g′ - g
    push!(δs, δ);
    push!(γs, γ);
    push!(qs, zero(θ)) 
    while length(δs) > M.m
        popfirst!(δs); popfirst!(γs); popfirst!(qs) 
    end
    return θ′ 
end
export Momentum, BFGS, LBFGS, step!, init!, DescentMethod

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

end
