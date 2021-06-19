module Methods
using Base:Integer, Float64, Int64
using FastClosures
using LinearAlgebra
using ExportAll
using OptimizationMethods.Wrapped

export Momentum, BFGS, LBFGS, step!, init!, DescentMethod, optimalize, GradientDescent

function optimalize(f, ∇f, x₀::Vector{Float64}, opt, e::Float64, i::Int64)::Tuple{Vector{Vector{Float64}},Vector{Float64},Int64}
    pts::Vector{Vector{Float64}} = [x₀] # kolejne wektory x
    err::Vector{Float64} = Vector{Float64}(undef, i) # kolejne wartości f. straty
    # minimalizacja realokacji
    p = 0
    while true
        p += 1
        prev::Float64 = f(pts[end])
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
    α::Float64 # learning rate
end

function step!(M::GradientDescent, f, ∇f, θ::Vector{Float64})::Vector{Float64}
    # α::Float64, g::Vector{Float64} = M.α, ∇f(θ)
    return θ .- M.α .* ∇f(θ)
end

struct Momentum <: DescentMethod
    α::Float64 # learning rate
    β::Float64 # momentum decay
    v::Vector{Float64}# momentum
end


Momentum(α, β, n::Int64) = Momentum(α, β, zeros(n))
# Momentum(α, β, n::Vector{Float64}) = Momentum(α, β, n)


function step!(M::Momentum, f, ∇f, x::Vector{Float64})::Vector{Float64}
    α, β, v, g = M.α, M.β, M.v, ∇f(x)
    @debug "Gradient: $g"
    @debug M
    @debug "Parameters: ($α, $β, $v, $g)"
    v .= β .* v .- α .* g
    return x .+ v
end

struct BFGS <: DescentMethod
    Q::Matrix{Float64}
    x′::Vector{Float64}
    g′::Vector{Float64}
    δ::Vector{Float64}
    γ::Vector{Float64}
end

function BFGS(n::T) where T <: Integer
    return BFGS(Matrix(1.0I, n, n), zeros(n), zeros(n), zeros(n), zeros(n))
end
function strong_backtracking(f, ∇, x::Vector{Float64}, d; α=1.0, β=1e-4, σ=0.1)::Float64
    y0::Float64, g0 = f(x), ∇(x) ⋅ d
    y_prev::Float64, α_prev::Float64  = NaN, 0.0
    αlo::Float64, αhi::Float64 = NaN, NaN
  # bracket phase
    # @debug "Before first inf loop"
    while true
        y::Float64 = f(x + α * d)
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
    ylo::Float64 = f(x + αlo * d)
    # @debug "Before second inf loop"
    α_old = NaN
    while true
        # @debug "Start loop"
        α = (αlo + αhi) / 2
        if α == α_old
            return α
        else
            α_old = α
        end
        # @debug "α: $α"
        y = f(x + α * d)
        # @debug "y: $y"
        # @debug "y > y0 + β * α * g0 $(y > y0 + β * α * g0)"
        # @debug "y ≥ ylo $(y ≥ ylo)"
        if y > y0 + β * α * g0 || y ≥ ylo
            # @debug "first case"
            # @debug "ylo $ylo"
            αhi  = α
            # @debug "ahi $αhi"
        else
            # @debug "second case"
            g  = ∇(x + α * d) ⋅ d
            # @debug "g: $g"
            # @debug "abs(g): $(abs(g))"
            # @debug "-σ * g0 : $(-σ * g0)"
            if abs(g) ≤ -σ * g0
                # @debug "second case inner first case"
                return α
            elseif g * (αhi - αlo) ≥ 0
                # @debug "second case inner second case"
                αhi = αlo
            end
            αlo = α
            # @debug "α: $α"
            # @debug "αlo: $αlo"
            # @debug "nether case"
        end
    end
end

    function step!(M::BFGS, f, ∇f, x::Vector{Float64})::Vector{Float64}
    if f(x) ≈ 0.0
        return x
    end

    Q::Matrix{Float64}, g::Vector{Float64} = M.Q, ∇f(x)
    # @debug "Q: $Q"
    # @debug "g: $g"
    α = strong_backtracking(f, ∇f, x, -Q * g)
    # @debug "α: $α"
    x′, g′, δ, γ = M.x′, M.g′, M.δ, M.γ

    x′ = x .+ α .* (-Q * g)
    @debug "x': $x′"
    @debug "x .+ α .* (-Q * g): $( x .+ α .* (-Q * g))"
    @debug "x: $x"
    g′ .= ∇f(x′)
    @debug "g: $g′"
    δ .= x′ .- x
    @debug "δ: $δ"
    @debug "x′ .- x: $(x′ .- x)"
    γ .= g′ .- g
    @debug "γ: $γ"
    tmp = δ' * γ
    tmp2 = Q * γ
    Q .= Q .- (δ * γ' * Q + tmp2 * δ') / tmp .+ (1 + (γ' * tmp2) / tmp)[1] * (δ * δ') / tmp
    @debug "new Q: $Q"
    return x′
end

    mutable struct LBFGS
    m::Int64
    δs::Vector{Vector{Float64}}
    γs::Vector{Vector{Float64}}
    qs::Vector{Vector{Float64}}
    g::Vector{Float64}
    z::Vector{Float64}
    d::Vector{Float64}
    current::Int64
    csize::Int64
    wraps::WrappedIndex{Int64}
    LBFGS() = new()
end

function init!(M::LBFGS, m::Int64, n::Int64)::LBFGS
    M.m = m
    M.δs = Vector{Vector{Float64}}(undef, m) # 
    M.γs =  Vector{Vector{Float64}}(undef, m) # 
    M.qs =  Vector{Vector{Float64}}(undef, m) # 
    for i in 1:m
        M.δs[i] = Vector{Float64}(undef, n) # 
        M.γs[i] = Vector{Float64}(undef, n) # 
        M.qs[i] = Vector{Float64}(undef, n) # 
    end
    M.g = Vector{Float64}(undef, n) # 
    M.z = Vector{Float64}(undef, n) # 
    M.d = Vector{Float64}(undef, n) # 
    M.current = 1
    M.csize = 0
    M.wraps = WrappedIndex(M.current, M.csize, m)
    return M
end

    function step!(M::LBFGS, f, ∇f, θ::Vector{Float64})::Vector{Float64}
    δs, γs, qs = M.δs, M.γs, M.qs 
    g = M.g
    g .= ∇f(θ)
    d = M.d
    z = M.z
    d = -g # kierunek
    wraps = M.wraps
    csize = M.wraps.size
    # if isnan(g)
        # there is no dericative at θ
        # we can't progress any further
        # posibli move in random direction
        # return θ || (!isnan(y_prev) && y ≥ y_prev)
    if csize > 0 
        q::Vector{Float64} = g
        last = before_mark(wraps)
        for i in  Iterators.reverse(wraps)
            qs[i] .= q
            q -= (δs[i] ⋅ q) / (γs[i] ⋅ δs[i]) .* γs[i]
        end
        z = (γs[last] .* δs[last] .* q) / (γs[last] ⋅ γs[last]) 
        for i in wraps
            z += δs[i] * (δs[i] ⋅ qs[i] - γs[i] ⋅ z) / (γs[i] ⋅ δs[i]) 
        end
        d = -z; # rekonstrukcja kierunku
    end
    φ = @closure α -> f(θ .+ α .* d)
    φ′ = @closure α -> ∇f(θ .+ α .* d) ⋅ d 
    α = line_search(φ, φ′, d)
    windx = write_index(wraps)
    # @debug "Point: $θ"
    # @debug "line_search: $α"
    θ′ = θ .+ α .* d
    # @debug "New Point: $θ′"
    δs[windx] .= θ′ .- θ
    γs[windx] .= ∇f(θ′) .- g

    commit_one(wraps)
    return θ′ 
end

function zoom(φ, φ′, αlo::Float64, αhi::Float64, c1=1e-4, c2=0.1, jmax=1000)::Float64
    φ′0 = φ′(0.0) 
    for j = 1:jmax
        αj::Float64 = 0.5(αlo + αhi) # bisection 
        φαj::Float64 = φ(αj)
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

    function line_search(φ, φ′, d, c1=1e-4, c2=0.1, ρ=0.1, αmax=100., jmax=1000)::Float64
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
