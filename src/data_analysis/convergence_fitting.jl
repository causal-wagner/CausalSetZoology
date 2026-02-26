"""
    compute_sigma_evolution(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
    )::Tuple{Vector{Int}, Matrix{Float64}}

Compute σ_k(N) for increasing sample size N and (optionally) averaged bins.

# Arguments
- `X`: Matrix of size `(Nsamples, nbins)`, typically output of `densify_hists`.
- `batchsize`: Step size in sample number `N`.
- `bin_average`: Number of original bins merged into one (`>= 1`).

# Returns
- `Ns::Vector{Int}`: Sample sizes used (N values).
- `σ::Matrix{Float64}`: `σ[j, k]` is the std of bin `k` at sample size `Ns[j]`.

# Keyword Arguments
- `batchsize`: Numeric control parameter for fitting/sampling resolution.
- `bin_average`: Bin selection or binning control parameter.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function compute_sigma_evolution(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
)::Tuple{Vector{Int}, Matrix{Float64}}

    @assert batchsize ≥ 1
    @assert bin_average ≥ 1

    Nsamples, nbins = size(X)

    # ---- bin averaging -----------------------------------------------------
    nbins_eff = cld(nbins, bin_average)

    Xb = if bin_average == 1
        X
    else
        Xavg = zeros(Float64, Nsamples, nbins_eff)
        for k in 1:nbins_eff
            lo = (k-1)*bin_average + 1
            hi = min(k*bin_average, nbins)
            Xavg[:, k] .= Statistics.mean(@view X[:, lo:hi]; dims=2)
        end
        Xavg
    end

    # ---- sample sizes ------------------------------------------------------
    Ns = collect(batchsize:batchsize:Nsamples)
    nsteps = length(Ns)

    # ---- cumulative statistics --------------------------------------------
    csum  = cumsum(Xb; dims=1)
    csum2 = cumsum(Xb.^2; dims=1)

    σ = zeros(Float64, nsteps, nbins_eff)

    for (j, N) in enumerate(Ns)
        μ   = view(csum, N, :) ./ N
        var = max.(view(csum2, N, :) ./ N .- μ.^2, 0.0)
        σ[j, :] .= sqrt.(var)
    end

    return Ns, σ
end

"""
    compute_mu_evolution(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
    )::Tuple{Vector{Int}, Matrix{Float64}}

Compute μₖ(N) (cumulative means) over sample size N.

# Returns
- `Ns::Vector{Int}`
- `μ::Matrix{Float64}` where `μ[j, k]` is the mean of bin `k` at sample size
  `Ns[j]`.

# Arguments
- `X`: Input coordinate/data values used in the computation.

# Keyword Arguments
- `batchsize`: Numeric control parameter for fitting/sampling resolution.
- `bin_average`: Bin selection or binning control parameter.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function compute_mu_evolution(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
)::Tuple{Vector{Int}, Matrix{Float64}}

    @assert batchsize ≥ 1
    @assert bin_average ≥ 1

    Nsamples, nbins = size(X)

    # ---- bin averaging -----------------------------------------------------
    nbins_eff = cld(nbins, bin_average)

    Xb = if bin_average == 1
        X
    else
        Xavg = zeros(Float64, Nsamples, nbins_eff)
        for k in 1:nbins_eff
            lo = (k-1)*bin_average + 1
            hi = min(k*bin_average, nbins)
            Xavg[:, k] .= Statistics.mean(@view X[:, lo:hi]; dims=2)
        end
        Xavg
    end

    # ---- sample sizes ------------------------------------------------------
    Ns = collect(batchsize:batchsize:Nsamples)
    nsteps = length(Ns)

    # ---- cumulative means --------------------------------------------------
    csum = cumsum(Xb; dims=1)
    μ = zeros(Float64, nsteps, nbins_eff)

    for (j, N) in enumerate(Ns)
        μ[j, :] .= view(csum, N, :) ./ N
    end

    return Ns, μ
end

"""
    fit_sigma_infty_alpha(
        σn::AbstractVector{<:Real},
        ns::AbstractVector{<:Real};
        σinf_init::Union{Nothing,Real}=nothing,
        α_init::Real=0.5,
        bounds_σinf::Union{Nothing,Tuple{Real,Real}}=nothing,
        bounds_α::Tuple{Real,Real}=(1e-3, 5.0),
        fix_sigma_inf::Bool=true,
    )::NamedTuple

Fit the convergence model
`σ(n) ≈ σ∞ + A n^{-α}`
to a single bin trajectory.

The objective is reduced by solving for `A` in closed form for each candidate
(`σ∞`, `α`) pair. If `fix_sigma_inf=true`, only `α` is optimized and `σ∞` is
held fixed (after optional clamping to `bounds_σinf`).

Returns `(σinf, α, A, objective)`.

# Arguments
- `ns`: Sample-size/count values used along convergence or fitting curves.

# Keyword Arguments
- `fix_sigma_inf`: Keyword option `fix_sigma_inf` controlling this method's behavior.

# Returns
- `result::NamedTuple`: Output of `fit_sigma_infty_alpha` with type annotation `NamedTuple`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function fit_sigma_infty_alpha(
    σn::AbstractVector{<:Real},
    ns::AbstractVector{<:Real};
    σinf_init::Union{Nothing,Real}=nothing,
    α_init::Real = 0.5,
    bounds_σinf::Union{Nothing,Tuple{Real,Real}}=nothing,
    bounds_α::Tuple{Real,Real} = (1e-3, 5.0),
    fix_sigma_inf::Bool = true,
)::NamedTuple
    # Initial guess

    if σinf_init === nothing
        σinf_init = σn[end]
    end

    if bounds_σinf === nothing
        bounds_σinf = (0.5 * σn[end], 1.5 * σn[end])
    end   

    if fix_sigma_inf
        σinf_fixed = σinf_init
        if σinf_fixed === nothing
            σinf_fixed = σn[end]
        end
        if bounds_σinf !== nothing
            σinf_fixed = clamp(σinf_fixed, bounds_σinf[1], bounds_σinf[2])
        end

        function obj_fixed(x)
            α = clamp(x[1], bounds_α[1], bounds_α[2])
            numer = sum(ns.^(-α) .* abs.(σn .- σinf_fixed))
            denom = sum(ns.^(-2α))
            A = numer / denom
            r = abs.(σn .- σinf_fixed) .- A .* ns.^(-α)
            return sum(r.^2)
        end

        result = Optim.optimize(obj_fixed, [α_init], Optim.NelderMead())
        α_hat = Optim.minimizer(result)[1]
        α_hat = clamp(α_hat, bounds_α[1], bounds_α[2])
        σinf_hat = σinf_fixed
        A_hat = sum(ns.^(-α_hat) .* abs.(σn .- σinf_hat)) / sum(ns.^(-2α_hat))
        fmin = Optim.minimum(result)
    else
        x0 = [σinf_init, α_init]

        # Objective function (reduced, as in boxed equation)
        function obj_free(x)
            σinf = length(x) == 2 ? x[1] : (σinf_init === nothing ? σn[end] : σinf_init)
            α = length(x) == 2 ? x[2] : x[1]
            # Clamp α to bounds
            α = clamp(α, bounds_α[1], bounds_α[2])
            # Clamp σinf to bounds
            σinf = clamp(σinf, bounds_σinf[1], bounds_σinf[2])
            # Compute A (reduced least squares)
            numer = sum(ns.^(-α) .* abs.(σn .- σinf))
            denom = sum(ns.^(-2α))
            A = numer / denom
            r = abs.(σn .- σinf) .- A .* ns.^(-α)
            return sum(r.^2)
        end
        
        # Run optimization (Nelder-Mead, no gradients)
        result = Optim.optimize(obj_free, x0, Optim.NelderMead())
        xopt = Optim.minimizer(result)
        σinf_hat, α_hat = xopt
        # Clamp to bounds for output
        α_hat = clamp(α_hat, bounds_α[1], bounds_α[2])
        if bounds_σinf !== nothing
            σinf_hat = clamp(σinf_hat, bounds_σinf[1], bounds_σinf[2])
        end
        # Compute A at optimum
        A_hat = sum(ns.^(-α_hat) .* abs.(σn .- σinf_hat)) / sum(ns.^(-2α_hat))
        fmin = Optim.minimum(result)
    end
    return (σinf = σinf_hat, α = α_hat, A = A_hat, objective = fmin)
end

"""
    fit_mu_infty_beta(
        μn::AbstractVector{<:Real},
        ns::AbstractVector{<:Real};
        μinf_init::Union{Nothing,Real}=nothing,
        β_init::Real=0.5,
        bounds_μinf::Union{Nothing,Tuple{Real,Real}}=nothing,
        bounds_β::Tuple{Real,Real}=(1e-3, 5.0),
        fix_sigma_inf::Bool=true,
    )::NamedTuple

Fit the convergence model
`μ(n) ≈ μ∞ + B n^{-β}`
to a single bin trajectory.

The objective is reduced by solving for `B` in closed form for each candidate
(`μ∞`, `β`) pair. If `fix_sigma_inf=true`, only `β` is optimized and `μ∞` is
held fixed (after clamping to bounds).

Returns `(μinf, β, B, objective)`.

# Arguments
- `ns`: Sample-size/count values used along convergence or fitting curves.

# Keyword Arguments
- `fix_sigma_inf`: Keyword option `fix_sigma_inf` controlling this method's behavior.

# Returns
- `result::NamedTuple`: Output of `fit_mu_infty_beta` with type annotation `NamedTuple`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function fit_mu_infty_beta(
    μn::AbstractVector{<:Real},
    ns::AbstractVector{<:Real};
    μinf_init::Union{Nothing,Real}=nothing,
    β_init::Real = 0.5,
    bounds_μinf::Union{Nothing,Tuple{Real,Real}}=nothing,
    bounds_β::Tuple{Real,Real} = (1e-3, 5.0),
    fix_sigma_inf::Bool = true,
)::NamedTuple
    if μinf_init === nothing
        μinf_init = μn[end]
    end

    if bounds_μinf === nothing
        δ = abs(μn[end])
        δ == 0 && (δ = 1.0)
        bounds_μinf = (μn[end] - δ, μn[end] + δ)
    end

    if fix_sigma_inf
        μinf_fixed = μinf_init
        if μinf_fixed === nothing
            μinf_fixed = μn[end]
        end
        μinf_fixed = clamp(μinf_fixed, bounds_μinf[1], bounds_μinf[2])

        function obj_fixed(x)
            β = clamp(x[1], bounds_β[1], bounds_β[2])
            numer = sum(ns.^(-β) .* abs.(μn .- μinf_fixed))
            denom = sum(ns.^(-2β))
            B = numer / denom
            r = abs.(μn .- μinf_fixed) .- B .* ns.^(-β)
            return sum(r.^2)
        end

        result = Optim.optimize(obj_fixed, [β_init], Optim.NelderMead())
        β_hat = Optim.minimizer(result)[1]
        β_hat = clamp(β_hat, bounds_β[1], bounds_β[2])
        μinf_hat = μinf_fixed
        B_hat = sum(ns.^(-β_hat) .* abs.(μn .- μinf_hat)) / sum(ns.^(-2β_hat))
        fmin = Optim.minimum(result)
    else
        x0 = [μinf_init, β_init]

        function obj_free(x)
            μinf = length(x) == 2 ? x[1] : (μinf_init === nothing ? μn[end] : μinf_init)
            β = length(x) == 2 ? x[2] : x[1]
            β = clamp(β, bounds_β[1], bounds_β[2])
            μinf = clamp(μinf, bounds_μinf[1], bounds_μinf[2])
            numer = sum(ns.^(-β) .* abs.(μn .- μinf))
            denom = sum(ns.^(-2β))
            B = numer / denom
            r = abs.(μn .- μinf) .- B .* ns.^(-β)
            return sum(r.^2)
        end

        result = Optim.optimize(obj_free, x0, Optim.NelderMead())
        xopt = Optim.minimizer(result)
        μinf_hat, β_hat = xopt
        β_hat = clamp(β_hat, bounds_β[1], bounds_β[2])
        μinf_hat = clamp(μinf_hat, bounds_μinf[1], bounds_μinf[2])
        B_hat = sum(ns.^(-β_hat) .* abs.(μn .- μinf_hat)) / sum(ns.^(-2β_hat))
        fmin = Optim.minimum(result)
    end
    return (μinf = μinf_hat, β = β_hat, B = B_hat, objective = fmin)
end

"""
    fit_sigma_convergence(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
        bounds_σinf::Union{Nothing,Tuple{Real,Real}} = nothing,
        bounds_α::Tuple{Real,Real} = (1e-3, 5.0),
        σinf_init::Union{Nothing,Real} = nothing,
        α_init::Real = 0.5,
        fix_sigma_inf::Bool = true,
    )

Compute σₖ(n) via `compute_sigma_evolution`, then fit
    σₖ(n) ≈ σ∞ + A n^{-α}
for:
1. every (possibly averaged) histogram bin k
2. the bin-averaged mean ⟨σ(n)⟩

Returns a NamedTuple with fields:
- Ns               :: Vector{Int}
- σ                :: Matrix{Float64}   (σ[j, k])
- bin_fits         :: Vector{Union{NamedTuple,Missing}} (may contain `missing` if not enough data in a bin)
- mean_fit         :: NamedTuple

# Arguments
- `X`: Input coordinate/data values used in the computation.

# Keyword Arguments
- `batchsize`: Numeric control parameter for fitting/sampling resolution.
- `bin_average`: Bin selection or binning control parameter.
- `fix_sigma_inf`: Keyword option `fix_sigma_inf` controlling this method's behavior.

# Returns
- `result`: Output of `fit_sigma_convergence` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function fit_sigma_convergence(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
    bounds_σinf::Union{Nothing,Tuple{Real,Real}} = nothing,
    bounds_α::Tuple{Real,Real} = (1e-3, 5.0),
    σinf_init::Union{Nothing,Real} = nothing,
    α_init::Real = 0.5,
    fix_sigma_inf::Bool = true,
)

    # ---- compute σ evolution ---------------------------------------------
    Ns, σ = compute_sigma_evolution(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
    )

    ns = Float64.(Ns)
    nsteps, nbins = size(σ)

    # ---- per-bin fits -----------------------------------------------------
    bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

    for k in 1:nbins
        σn = view(σ, :, k)

        # include zeros; only drop non-finite values
        mask = isfinite.(σn)
        ns_k = ns[mask]
        σn_k = σn[mask]

        if length(σn_k) < 3
            bin_fits[k] = missing
        else
            bin_fits[k] = fit_sigma_infty_alpha(
                σn_k,
                ns_k;
                σinf_init = σinf_init,
                bounds_σinf = bounds_σinf,
                bounds_α = bounds_α,
                α_init = α_init,
                fix_sigma_inf = fix_sigma_inf,
            )
        end
    end

    # ---- mean σ(n) fit ----------------------------------------------------
    σmean = vec(Statistics.mean(σ; dims=2))
    mask = isfinite.(σmean)
    ns_m = ns[mask]
    σm   = σmean[mask]

    mean_fit = fit_sigma_infty_alpha(
        σm,
        ns_m;
        σinf_init = σm[end],
        bounds_σinf = bounds_σinf,
        bounds_α = bounds_α,
        α_init = α_init,
        fix_sigma_inf = fix_sigma_inf,
    )

    return (
        Ns = Ns,
        σ = σ,
        bin_fits = bin_fits,
        mean_fit = mean_fit,
    )
end

"""
    fit_mu_convergence(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
        bounds_μinf::Union{Nothing,Tuple{Real,Real}} = nothing,
        bounds_β::Tuple{Real,Real} = (1e-3, 5.0),
        μinf_init::Union{Nothing,Real} = nothing,
        β_init::Real = 0.5,
        fix_sigma_inf::Bool = true,
    )

Compute μₖ(n) via `compute_mu_evolution`, then fit
    μₖ(n) ≈ μ∞ + B n^{-β}
for:
1. every (possibly averaged) histogram bin k
2. the bin-averaged mean ⟨μ(n)⟩

Returns a NamedTuple with fields:
- Ns               :: Vector{Int}
- μ                :: Matrix{Float64}   (μ[j, k])
- bin_fits         :: Vector{Union{NamedTuple,Missing}}
- mean_fit         :: NamedTuple

# Arguments
- `X`: Input coordinate/data values used in the computation.

# Keyword Arguments
- `batchsize`: Numeric control parameter for fitting/sampling resolution.
- `bin_average`: Bin selection or binning control parameter.
- `fix_sigma_inf`: Keyword option `fix_sigma_inf` controlling this method's behavior.

# Returns
- `result`: Output of `fit_mu_convergence` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function fit_mu_convergence(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
    bounds_μinf::Union{Nothing,Tuple{Real,Real}} = nothing,
    bounds_β::Tuple{Real,Real} = (1e-3, 5.0),
    μinf_init::Union{Nothing,Real} = nothing,
    β_init::Real = 0.5,
    fix_sigma_inf::Bool = true,
)
    Ns, μ = compute_mu_evolution(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
    )

    ns = Float64.(Ns)
    nsteps, nbins = size(μ)

    bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

    for k in 1:nbins
        μn = view(μ, :, k)
        mask = isfinite.(μn)
        ns_k = ns[mask]
        μn_k = μn[mask]

        if length(μn_k) < 3
            bin_fits[k] = missing
        else
            bin_fits[k] = fit_mu_infty_beta(
                μn_k,
                ns_k;
                μinf_init = μinf_init,
                bounds_μinf = bounds_μinf,
                bounds_β = bounds_β,
                β_init = β_init,
                fix_sigma_inf = fix_sigma_inf,
            )
        end
    end

    μmean = vec(Statistics.mean(μ; dims=2))
    mask = isfinite.(μmean)
    ns_m = ns[mask]
    μm   = μmean[mask]

    mean_fit = fit_mu_infty_beta(
        μm,
        ns_m;
        μinf_init = μm[end],
        bounds_μinf = bounds_μinf,
        bounds_β = bounds_β,
        β_init = β_init,
        fix_sigma_inf = fix_sigma_inf,
    )

    return (
        Ns = Ns,
        μ = μ,
        bin_fits = bin_fits,
        mean_fit = mean_fit,
    )
end
