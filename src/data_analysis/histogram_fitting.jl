"""
    fit_curve(
        y_values::Vector{Float64},
        f::Function,
        param_syms::Tuple{Vararg{Symbol}},
        ;
        x_values::Union{Nothing,Vector{<:Real}} = nothing,
        stds::Union{Nothing,Vector{Float64}} = nothing,
        minimize_χ²::Bool = false,
        init::Union{Nothing,NamedTuple} = nothing,
        bounds::Union{Nothing,Tuple{NamedTuple,NamedTuple}} = nothing,
        goodness_of_fit::Bool = false,
        ϵ::Real = 1e-3,
        multistart::Int = 1,
        rng::Union{Nothing,Random.AbstractRNG} = nothing,
        optim_options = nothing,
        method = Optim.NelderMead(),
        autodiff = nothing,
        verbose::Bool = false,
        std_fn::Union{Nothing,Function} = nothing,
        verbose_step::Union{Nothing,Int} = nothing,
        return_cov::Bool = false,
        bootstrap_errorbars::Bool = false,
        n_boot::Int = 200,
    )

Fit model values `f.(x, params)` to `y_values`.

`param_syms` defines parameter order in returned NamedTuples. If `stds` are
provided and `minimize_χ²=true`, weighted residuals are used (optionally via
`std_fn`). Multi-start optimization is supported via `multistart`.

# Returns
- `params::NamedTuple` by default,
- optionally diagnostics (`rel_residuals`, `χ²`),
- optionally covariance/error estimates (`cov`, `stderr`) when `return_cov=true`.

# Arguments
- `y_values`: Observed response values to fit or analyze.
- `f`: Model/function handle used by the method.
- `param_syms`: Parameter symbols/values defining model parameterization.

# Keyword Arguments
- `x_values`: X-axis/sample-coordinate values aligned with the modeled/observed data.
- `stds`: Keyword option `stds` controlling this method's behavior.
- `init`: Keyword option `init` controlling this method's behavior.
- `bounds`: Keyword option `bounds` controlling this method's behavior.
- `goodness_of_fit`: Keyword option `goodness_of_fit` controlling this method's behavior.
- `multistart`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.
- `optim_options`: Keyword option `optim_options` controlling this method's behavior.
- `method`: Keyword option `method` controlling this method's behavior.
- `autodiff`: Keyword option `autodiff` controlling this method's behavior.
- `verbose`: Boolean toggle controlling output or execution behavior.
- `std_fn`: Callable used to evaluate the model or compute transformed uncertainties.
- `verbose_step`: Keyword option `verbose_step` controlling this method's behavior.
- `return_cov`: Keyword option `return_cov` controlling this method's behavior.
- `bootstrap_errorbars`: Keyword option `bootstrap_errorbars` controlling this method's behavior.
- `n_boot`: Numeric control parameter for fitting, binning, or Monte Carlo/permutation resolution.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function fit_curve(
    y_values::Vector{Float64},
    f::Function,
    param_syms::Tuple{Vararg{Symbol}};
    x_values::Union{Nothing,Vector{<:Real}} = nothing,
    stds::Union{Nothing,Vector{Float64}} = nothing,
    minimize_χ²::Bool = false,
    init::Union{Nothing,NamedTuple} = nothing,
    bounds::Union{Nothing,Tuple{NamedTuple,NamedTuple}} = nothing,
    goodness_of_fit::Bool = false,
    ϵ::Real = 1e-3,
    multistart::Int = 1,
    rng::Union{Nothing,Random.AbstractRNG} = nothing,
    optim_options = nothing,
    method = Optim.NelderMead(),
    autodiff = nothing,
    verbose::Bool = false,
    std_fn::Union{Nothing,Function} = nothing,
    verbose_step::Union{Nothing,Int} = nothing,
    return_cov::Bool = false,
    bootstrap_errorbars::Bool = false,
    n_boot::Int = 200,
)
    if std_fn !== nothing && stds === nothing
        error("std_fn requires stds to be provided")
    end

    if !isnothing(stds)
        if !(length(stds) == length(y_values))
            throw(ArgumentError("stds length must match y_values length"))
        end
        stds = replace_zeros(stds; ϵ = ϵ)
        bad = (.!isfinite.(stds)) .| (stds .<= 0)
        if any(bad)
            nz = stds[isfinite.(stds) .& (stds .> 0)]
            fillval = isempty(nz) ? ϵ : minimum(nz) * ϵ
            @warn "stds contain non-finite or non-positive values; adjusting to eps*min nonzero std to avoid infinite chi-squared." eps = ϵ
            stds = copy(stds)
            stds[bad] .= fillval
        end
    else
        if !(!minimize_χ²)
            throw(ArgumentError("minimize_χ² requires stds to be provided"))
        end
    end

    p = length(param_syms)
    if !(multistart ≥ 1)
        throw(ArgumentError("multistart must be ≥ 1"))
    end
    to_vec(nt::NamedTuple) = [getfield(nt, s) for s in param_syms]
    to_nt(v::AbstractVector) = NamedTuple{param_syms}(Tuple(v))

    init_vec = if init === nothing
        ones(p)
    else
        to_vec(init)
    end

    bounds_vec = nothing
    if bounds !== nothing
        if bounds isa Tuple && length(bounds) == 2
            lower, upper = bounds
            lower_vec = to_vec(lower)
            upper_vec = to_vec(upper)
            if !(length(lower_vec) == p && length(upper_vec) == p)
                throw(ArgumentError("bounds length must match param_syms"))
            end
            bounds_vec = (lower_vec, upper_vec)
        else
            error("bounds must be a tuple (lower, upper)")
        end
    end

    xs = x_values === nothing ? collect(1:length(y_values)) : x_values
    ys = y_values

    function make_obj(ys_local)
        return function (x)
            v = bounds_vec === nothing ? x : clamp.(x, bounds_vec[1], bounds_vec[2])
            params = to_nt(v)
            preds = f.(xs, Ref(params))
            if minimize_χ²
                σ = std_fn === nothing ? stds : std_fn(ys_local, preds, stds, params)
                r = (ys_local .- preds) ./ σ
            else
                r = ys_local .- preds
            end
            return sum(r .^ 2)
        end
    end

    function solve(x0, ys_local)
        obj = make_obj(ys_local)
        if isnothing(optim_options)
            result = autodiff === nothing ?
                Optim.optimize(obj, x0, method) :
                Optim.optimize(obj, x0, method; autodiff = autodiff)
        else
            result = autodiff === nothing ?
                Optim.optimize(obj, x0, method, optim_options) :
                Optim.optimize(obj, x0, method, optim_options; autodiff = autodiff)
        end
        xopt = Optim.minimizer(result)
        fmin = Optim.minimum(result)
        return xopt, fmin
    end

    function score_for(xopt)
        x = bounds_vec === nothing ? xopt : clamp.(xopt, bounds_vec[1], bounds_vec[2])
        params = to_nt(x)
        preds = f.(xs, Ref(params))
        residuals = ys .- preds
        if !isnothing(stds)
            σ = std_fn === nothing ? stds : std_fn(ys, preds, stds, params)
            dof = length(ys) - p
            return dof > 0 ? sum((residuals ./ σ) .^ 2) / dof : NaN
        else
            denom = similar(ys)
            @inbounds for i in eachindex(ys)
                denom[i] = ys[i] == 0 ? eps() : ys[i]
            end
            rel = residuals ./ denom
            return sqrt(Statistics.mean(rel .^ 2))
        end
    end

    best_x, best_f = solve(init_vec, ys)
    best_score = score_for(best_x)
    label = isnothing(stds) ? "rel_rms" : "χ²"
    step = if verbose
        verbose_step === nothing ? max(1, round(Int, multistart * 0.1)) : max(1, verbose_step)
    else
        0
    end
    if verbose
        println("multistart 1: ", label, " = ", best_score, " (best = ", best_score, ")")
        flush(stdout)
    end

    if multistart > 1
        rng = rng === nothing ? Random.GLOBAL_RNG : rng
        scale = bounds_vec === nothing ? [init === 0 ? 1. : abs(init) for init in init_vec] : bounds_vec[2] .- bounds_vec[1]
        for i in 2:multistart
            x0 = if bounds_vec === nothing
                init_vec .+ (2 .* rand(rng, p) .- 1) .* scale
            else
                lower, upper = bounds_vec
                x = similar(init_vec)
                for j in 1:p
                    lo = lower[j]
                    hi = upper[j]
                    if isfinite(lo) && isfinite(hi)
                        x[j] = lo + rand(rng) * (hi - lo)
                    else
                        s = init_vec[j] == 0 ? 1.0 : abs(init_vec[j])
                        x[j] = init_vec[j] + (2 * rand(rng) - 1) * s
                        if isfinite(lo)
                            x[j] = max(x[j], lo)
                        end
                        if isfinite(hi)
                            x[j] = min(x[j], hi)
                        end
                    end
                end
                x
            end
            xopt, fmin = solve(x0, ys)
            score = score_for(xopt)
            if fmin < best_f
                best_x, best_f = xopt, fmin
                best_score = score
            end
            if verbose && (i % step == 0 || i == multistart)
                println("multistart ", i, ": ", label, " = ", score, " (best = ", best_score, ")")
                flush(stdout)
            end
        end
    end

    if verbose
        println("multistart done: best ", label, " = ", best_score)
        flush(stdout)
    end

    xopt = bounds_vec === nothing ? best_x : clamp.(best_x, bounds_vec[1], bounds_vec[2])
    params = to_nt(xopt)

    function jacobian_fd(xs, params; eps = 1e-6)
        p = length(param_syms)
        base_preds = f.(xs, Ref(params))
        J = Matrix{Float64}(undef, length(xs), p)
        for (j, sym) in enumerate(param_syms)
            v = getfield(params, sym)
            step = (v == 0 ? 1.0 : abs(v)) * eps
            vpert = v + step
            pvec = [getfield(params, s) for s in param_syms]
            pvec[j] = vpert
            ppert = NamedTuple{param_syms}(Tuple(pvec))
            preds = f.(xs, Ref(ppert))
            J[:, j] = (preds .- base_preds) ./ step
        end
        return J
    end

    function cov_and_stderr(xs, ys, params)
        if bootstrap_errorbars
            if stds === nothing
                error("bootstrap_errorbars requires stds to be provided")
            end
            rng_local = rng === nothing ? Random.GLOBAL_RNG : rng
            n_boot = n_boot
            p = length(param_syms)
            samples = Matrix{Float64}(undef, n_boot, p)
            base_preds = f.(xs, Ref(params))
            σ_base = std_fn === nothing ? stds : std_fn(ys, base_preds, stds, params)
            for i in 1:n_boot
                ys_boot = ys .+ randn(rng_local, length(ys)) .* σ_base
                xopt_boot, _ = solve(xopt, ys_boot)
                xopt_boot = bounds_vec === nothing ? xopt_boot : clamp.(xopt_boot, bounds_vec[1], bounds_vec[2])
                samples[i, :] = xopt_boot
            end
            cov = Statistics.cov(samples)
            stderr = sqrt.(abs.(diag(cov)))
            stderr_nt = NamedTuple{param_syms}(Tuple(stderr))
            return cov, stderr_nt
        end
        J = jacobian_fd(xs, params)
        if !isnothing(stds)
            σ = std_fn === nothing ? stds : std_fn(ys, f.(xs, Ref(params)), stds, params)
            W = LinearAlgebra.Diagonal(1.0 ./ (σ .^ 2))
            JT_W_J = J' * W * J
            dof = length(ys) - length(param_syms)
            s2 = dof > 0 ? sum(((ys .- f.(xs, Ref(params))) ./ σ) .^ 2) / dof : 1.0
            cov = try
                inv(JT_W_J) * s2
            catch
                LinearAlgebra.pinv(JT_W_J) * s2
            end
        else
            JT_J = J' * J
            dof = length(ys) - length(param_syms)
            s2 = dof > 0 ? sum((ys .- f.(xs, Ref(params))) .^ 2) / dof : 1.0
            cov = try
                inv(JT_J) * s2
            catch
                LinearAlgebra.pinv(JT_J) * s2
            end
        end
        stderr = sqrt.(abs.(diag(cov)))
        stderr_nt = NamedTuple{param_syms}(Tuple(stderr))
        return cov, stderr_nt
    end

    if goodness_of_fit
        preds = f.(xs, Ref(params))
        residuals = ys .- preds
        if !isnothing(stds)
            σ = std_fn === nothing ? stds : std_fn(ys, preds, stds, params)
            dof = length(ys) - p
            if dof <= 0
                @warn "chi-squared undefined: degrees of freedom <= 0" dof = dof
                χ² = NaN
            else
                χ² = sum((residuals ./ σ) .^ 2) / dof
            end
            if return_cov
                cov, stderr_nt = cov_and_stderr(xs, ys, params)
                return (params = params, rel_residuals = residuals ./ ys, χ² = χ², cov = cov, stderr = stderr_nt)
            end
            return (params = params, rel_residuals = residuals ./ ys, χ² = χ²)
        end
        if return_cov
            cov, stderr_nt = cov_and_stderr(xs, ys, params)
            return (params = params, rel_residuals = residuals ./ ys, cov = cov, stderr = stderr_nt)
        end
        return (params = params, rel_residuals = residuals ./ ys)
    end

    if return_cov
        cov, stderr_nt = cov_and_stderr(xs, ys, params)
        return (params = params, cov = cov, stderr = stderr_nt)
    end
    return params
end

"""
    fit_histogram_bins(y_values, f, param_syms, bin_lo, bin_hi; kwargs...)

See [`fit_curve`](@ref).

This convenience wrapper slices `y_values` (and matching `stds` / `x_values`)
to the bin interval `[bin_lo, bin_hi]`, then forwards all fitting options to
`fit_curve`.

# Arguments
- `y_values`: Observed response values to fit or analyze.
- `f`: Model/function handle used by the method.
- `param_syms`: Parameter symbols/values defining model parameterization.
- `bin_lo`: Bin selection or binning control parameter.
- `bin_hi`: Bin selection or binning control parameter.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: Output of `fit_histogram_bins` as described in the summary above.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function fit_histogram_bins(
    y_values::Vector{Float64},
    f::Function,
    param_syms::Tuple{Vararg{Symbol}},
    bin_lo::Int,
    bin_hi::Int;
    stds::Union{Nothing,Vector{Float64}} = nothing,
    minimize_χ²::Bool = false,
    x_values::Union{Nothing,Vector{<:Real}} = nothing,
    init::Union{Nothing,NamedTuple} = nothing,
    bounds::Union{Nothing,Tuple{NamedTuple,NamedTuple}} = nothing,
    goodness_of_fit::Bool = false,
    ϵ::Real = 1e-3,
    multistart::Int = 1,
    rng::Union{Nothing,Random.AbstractRNG} = nothing,
    optim_options = nothing,
    method = Optim.NelderMead(),
    autodiff = nothing,
    verbose::Bool = false,
    std_fn::Union{Nothing,Function} = nothing,
    verbose_step::Union{Nothing,Int} = nothing,
    return_cov::Bool = false,
)
    if !(1 <= bin_lo <= bin_hi <= length(y_values))
        throw(ArgumentError("bin range out of bounds"))
    end
    if !isnothing(stds)
        if !(length(stds) == length(y_values))
            throw(ArgumentError("stds length must match y_values length"))
        end
        stds_slice = stds[bin_lo:bin_hi]
    else
        stds_slice = nothing
    end

    xs = if x_values !== nothing
        if !(length(x_values) >= bin_hi)
            throw(ArgumentError("x_values length must be larger than bin_hi"))
        end
        x_values[bin_lo:bin_hi]
    else
        collect(bin_lo:bin_hi)
    end
    ys = y_values[bin_lo:bin_hi]

    return fit_curve(
        ys,
        f,
        param_syms;
        x_values = xs,
        stds = stds_slice,
        minimize_χ² = minimize_χ²,
        init = init,
        bounds = bounds,
        goodness_of_fit = goodness_of_fit,
        ϵ = ϵ,
        multistart = multistart,
        rng = rng,
        optim_options = optim_options,
        method = method,
        autodiff = autodiff,
        verbose = verbose,
        std_fn = std_fn,
        verbose_step = verbose_step,
        return_cov = return_cov,
    )
end
