"""
    _fit_curve_to_vec(param_syms, nt)

Convert a parameter `NamedTuple` to a vector in `param_syms` order.

# Arguments
- `param_syms`: Ordered parameter symbols.
- `nt`: Parameter values as a `NamedTuple`.

# Returns
- `Vector{Any}`: Parameter values in the order given by `param_syms`."""
_fit_curve_to_vec(param_syms::Tuple{Vararg{Symbol}}, nt::NamedTuple) =
    begin
        if !all(s -> hasproperty(nt, s), param_syms)
            throw(ArgumentError("named tuple must contain all symbols in param_syms"))
        end
        [getfield(nt, s) for s in param_syms]
    end

"""
    _fit_curve_to_nt(param_syms, v)

Convert a parameter vector to a `NamedTuple` in `param_syms` order.

# Arguments
- `param_syms`: Ordered parameter symbols.
- `v`: Parameter vector.

# Returns
- `NamedTuple`: Parameter values mapped to `param_syms`."""
function _fit_curve_to_nt(param_syms::Tuple{Vararg{Symbol}}, v::AbstractVector)
    if length(v) != length(param_syms)
        throw(ArgumentError("parameter vector length must match param_syms length"))
    end
    return NamedTuple{param_syms}(Tuple(v))
end

"""Internal bounds-mode marker type for `fit_curve` helpers."""
abstract type _FitBounds end
"""Bounds mode indicating no parameter clamping is applied."""
struct _NoBounds <: _FitBounds end
"""Bounds mode storing lower/upper box constraints in parameter order."""
struct _BoxBounds <: _FitBounds
    lower::Vector{Float64}
    upper::Vector{Float64}
end

"""Internal weighting-mode marker type for `fit_curve` helpers."""
abstract type _FitWeightingMode end

"""Weighting mode for ordinary least squares (unweighted residuals)."""
struct _UnweightedMode <: _FitWeightingMode end

"""Weighting mode for chi-squared style weighted residuals."""
struct _WeightedMode <: _FitWeightingMode
    stds::Vector{Float64}
    std_fn::Union{Nothing,Function}
end

"""Internal normalized fit configuration built by `_fit_curve_prepare_inputs`."""
struct _FitConfig
    param_syms::Tuple{Vararg{Symbol}}
    init_vec::Vector{Float64}
    bounds_mode::_FitBounds
    objective_mode::_FitWeightingMode
    diagnostics_mode::_FitWeightingMode
    multistart::Int
    rng::Random.AbstractRNG
    optim_options
    method
    autodiff
    verbose::Bool
    verbose_step::Union{Nothing,Int}
    return_cov::Bool
    goodness_of_fit::Bool
    bootstrap_errorbars::Bool
    n_boot::Int
end

"""Internal summary of the best multistart candidate."""
struct _FitRunSummary
    best_x::Vector{Float64}
    best_f::Float64
    best_score::Float64
end

"""Internal typed fit result before conversion to the public return shape."""
struct _FitResult
    params
    residuals::Vector{Float64}
    χ²::Union{Nothing,Float64}
    cov::Union{Nothing,Matrix{Float64}}
    stderr
end

"""
    _fit_curve_apply_bounds(x, bounds_mode)

Apply internal bounds mode to a parameter vector.

# Arguments
- `x`: Parameter vector.
- `bounds_mode`: Internal bounds mode.

# Returns
- `AbstractVector`: Original or clamped parameter vector.

# Throws
- `ArgumentError`: Raised when `x` and bounds lengths do not match."""
_fit_curve_apply_bounds(x, ::_NoBounds) = x
function _fit_curve_apply_bounds(x, bounds::_BoxBounds)
    if length(x) != length(bounds.lower) || length(x) != length(bounds.upper)
        throw(ArgumentError("x length must match bounds length"))
    end
    return clamp.(x, bounds.lower, bounds.upper)
end

"""
    _fit_curve_make_bounds_mode(bounds_vec)

Convert optional raw bounds vectors to an internal bounds mode.

# Arguments
- `bounds_vec`: Either `nothing` or `(lower, upper)` vectors.

# Returns
- `_FitBounds`: Internal bounds representation.

# Throws
- `ArgumentError`: Raised when lower/upper vectors have mismatched lengths or invalid ordering."""
_fit_curve_make_bounds_mode(::Nothing) = _NoBounds()
function _fit_curve_make_bounds_mode(bounds_vec::Tuple{<:AbstractVector,<:AbstractVector})
    lower = Vector{Float64}(bounds_vec[1])
    upper = Vector{Float64}(bounds_vec[2])
    if length(lower) != length(upper)
        throw(ArgumentError("lower and upper bounds must have equal length"))
    end
    if any(lower .> upper)
        throw(ArgumentError("bounds must satisfy lower <= upper component-wise"))
    end
    return _BoxBounds(lower, upper)
end

"""
    _fit_curve_make_objective_mode(stds, std_fn, minimize_χ²)

Build the residual-weighting mode used by the optimization objective.

# Arguments
- `stds`: Optional standard deviations.
- `std_fn`: Optional standard-deviation callback.
- `minimize_χ²`: Whether weighted residual optimization is requested.

# Returns
- `_FitWeightingMode`: Objective weighting mode.

# Throws
- `ArgumentError`: Raised when weighted optimization is requested without `stds`."""
function _fit_curve_make_objective_mode(::Nothing, ::Union{Nothing,Function}, minimize_χ²::Bool)
    if minimize_χ²
        throw(ArgumentError("minimize_χ² requires stds to be provided"))
    end
    return _UnweightedMode()
end
function _fit_curve_make_objective_mode(stds::Vector{Float64}, std_fn::Union{Nothing,Function}, minimize_χ²::Bool)
    return minimize_χ² ? _WeightedMode(stds, std_fn) : _UnweightedMode()
end

"""
    _fit_curve_make_diagnostics_mode(stds, std_fn)

Build the residual-weighting mode used by diagnostics and covariance steps.

# Arguments
- `stds`: Optional standard deviations.
- `std_fn`: Optional standard-deviation callback.

# Returns
- `_FitWeightingMode`: Diagnostics weighting mode."""
_fit_curve_make_diagnostics_mode(::Nothing, ::Union{Nothing,Function}) = _UnweightedMode()
_fit_curve_make_diagnostics_mode(stds::Vector{Float64}, std_fn::Union{Nothing,Function}) = _WeightedMode(stds, std_fn)

"""
    _fit_curve_sigma(mode, ys, preds, params)

Resolve effective standard deviations for the current weighting mode.

# Arguments
- `mode`: Internal weighting mode.
- `ys`: Observed values.
- `preds`: Model predictions.
- `params`: Current parameter `NamedTuple`.

# Returns
- `nothing` for unweighted mode, otherwise a vector of effective standard deviations.

# Throws
- `ArgumentError`: Raised when weighted standard deviations are invalid or length-mismatched."""
_fit_curve_sigma(:: _UnweightedMode, ys, preds, params) = nothing
function _fit_curve_sigma(mode::_WeightedMode, ys, preds, params)
    σ = mode.std_fn === nothing ? mode.stds : mode.std_fn(ys, preds, mode.stds, params)
    if length(σ) != length(ys)
        throw(ArgumentError("effective stds length must match ys length"))
    end
    if any((.!isfinite.(σ)) .| (σ .<= 0))
        throw(ArgumentError("effective stds must be finite and positive"))
    end
    return σ
end

"""Return stored standard deviations for the given weighting mode."""
_fit_curve_stds(:: _UnweightedMode) = nothing
_fit_curve_stds(mode::_WeightedMode) = mode.stds

"""Return stored standard-deviation callback for the given weighting mode."""
_fit_curve_std_fn(:: _UnweightedMode) = nothing
_fit_curve_std_fn(mode::_WeightedMode) = mode.std_fn

"""Predicate indicating whether weighting mode carries standard deviations."""
_fit_curve_has_stds(mode::_FitWeightingMode) = mode isa _WeightedMode

"""Human-readable label used for progress output for a weighting mode."""
_fit_curve_label(mode::_FitWeightingMode) = _fit_curve_has_stds(mode) ? "χ²" : "rel_rms"

"""Convert internal bounds mode back to optional tuple representation."""
_fit_curve_bounds_tuple(:: _NoBounds) = nothing
_fit_curve_bounds_tuple(bounds::_BoxBounds) = (bounds.lower, bounds.upper)

"""
    _fit_curve_residual_bundle(x, ys, f, xs, param_syms, bounds_mode, weighting_mode)

Compute bounded parameters, predictions, residuals, and effective standard deviations.

# Arguments
- `x`: Parameter vector candidate.
- `ys`: Observed values.
- `f`: Model function `f(x, params)`.
- `xs`: Input coordinates.
- `param_syms`: Ordered parameter symbols.
- `bounds_mode`: Internal bounds mode.
- `weighting_mode`: Internal weighting mode.

# Returns
- `NamedTuple`: `(x, params, preds, residuals, σ)`.

# Throws
- `ArgumentError`: Raised when input lengths are inconsistent."""
function _fit_curve_residual_bundle(
    x,
    ys,
    f::Function,
    xs,
    param_syms::Tuple{Vararg{Symbol}},
    bounds_mode::_FitBounds,
    weighting_mode::_FitWeightingMode,
)
    if length(xs) != length(ys)
        throw(ArgumentError("xs length must match ys length"))
    end
    if length(x) != length(param_syms)
        throw(ArgumentError("parameter vector length must match param_syms length"))
    end
    x_eff = _fit_curve_apply_bounds(x, bounds_mode)
    params = _fit_curve_to_nt(param_syms, x_eff)
    preds = f.(xs, Ref(params))
    residuals = ys .- preds
    σ = _fit_curve_sigma(weighting_mode, ys, preds, params)
    return (x = x_eff, params = params, preds = preds, residuals = residuals, σ = σ)
end

"""
    _fit_curve_objective_value(residuals, σ)

Compute sum-of-squares objective value from residuals and optional weights.

# Arguments
- `residuals`: Residual vector.
- `σ`: Optional standard deviations.

# Returns
- `Float64`: Objective value.

# Throws
- `ArgumentError`: Raised when `σ` is provided with mismatched length."""
function _fit_curve_objective_value(residuals, σ)
    if σ === nothing
        return sum(residuals .^ 2)
    end
    if length(σ) != length(residuals)
        throw(ArgumentError("stds length must match residuals length"))
    end
    return sum((residuals ./ σ) .^ 2)
end

"""
    _fit_curve_score_value(residuals, ys, σ, p)

Compute the scalar fit score from residuals.

Returns reduced `χ²` when `σ` is provided, otherwise relative RMS residual.

# Arguments
- `residuals`: Residual vector.
- `ys`: Observed values.
- `σ`: Optional standard deviations.
- `p`: Number of fitted parameters.

# Returns
- `Float64`: Score value.

# Throws
- `ArgumentError`: Raised when lengths are inconsistent or `p < 0`."""
function _fit_curve_score_value(residuals, ys, σ, p::Int)
    if p < 0
        throw(ArgumentError("number of parameters p must be non-negative"))
    end
    if length(residuals) != length(ys)
        throw(ArgumentError("residuals length must match ys length"))
    end
    if σ !== nothing
        if length(σ) != length(ys)
            throw(ArgumentError("stds length must match ys length"))
        end
        dof = length(ys) - p
        return dof > 0 ? sum((residuals ./ σ) .^ 2) / dof : NaN
    end
    denom = similar(residuals, Float64)
    @inbounds for i in eachindex(ys)
        denom[i] = ys[i] == 0 ? eps() : ys[i]
    end
    rel = residuals ./ denom
    return sqrt(Statistics.mean(rel .^ 2))
end

"""
    _fit_curve_require_param_keys(nt, param_syms, name)

Validate that a `NamedTuple` contains all required parameter symbols.

# Arguments
- `nt`: Candidate `NamedTuple`.
- `param_syms`: Required parameter symbols.
- `name`: Human-readable input name for error messages.

# Throws
- `ArgumentError`: Raised when required symbols are missing."""
function _fit_curve_require_param_keys(
    nt::NamedTuple,
    param_syms::Tuple{Vararg{Symbol}},
    name::AbstractString,
)
    if !all(s -> hasproperty(nt, s), param_syms)
        throw(ArgumentError("$(name) must contain all symbols in param_syms"))
    end
    return nothing
end

"""
    _fit_curve_prepare_inputs(y_values, param_syms; kwargs...)

Normalize and validate all `fit_curve` inputs into a single internal config.

# Arguments
- `y_values`: Observed response values.
- `param_syms`: Ordered parameter symbols.

# Keyword Arguments
- Matches the `fit_curve` keyword interface.

# Returns
- `(xs, ys, cfg)`: Prepared x-values, y-values, and internal `_FitConfig`.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `DomainError`: Raised for invalid numeric options.
- `DimensionMismatch`: Raised when provided arrays have inconsistent lengths."""
function _fit_curve_prepare_inputs(
    y_values::Vector{Float64},
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
        throw(ArgumentError("std_fn requires stds to be provided"))
    end
    if !(ϵ > 0)
        throw(DomainError(ϵ, "ϵ must be positive"))
    end
    if !(multistart >= 1)
        throw(ArgumentError("multistart must be ≥ 1"))
    end
    if !all(isfinite, y_values)
        throw(ArgumentError("y_values must be finite"))
    end

    stds_clean = if isnothing(stds)
        if minimize_χ²
            throw(ArgumentError("minimize_χ² requires stds to be provided"))
        end
        nothing
    else
        if length(stds) != length(y_values)
            throw(ArgumentError("stds length must match y_values length"))
        end
        local s = replace_zeros(stds; ϵ = ϵ)
        bad = (.!isfinite.(s)) .| (s .<= 0)
        if any(bad)
            nz = s[isfinite.(s) .& (s .> 0)]
            fillval = isempty(nz) ? ϵ : minimum(nz) * ϵ
            @warn "stds contain non-finite or non-positive values; adjusting to eps*min nonzero std to avoid infinite chi-squared." eps = ϵ
            s = copy(s)
            s[bad] .= fillval
        end
        s
    end

    p = length(param_syms)
    init_vec = if init === nothing
        ones(Float64, p)
    else
        _fit_curve_require_param_keys(init, param_syms, "init")
        Float64.(_fit_curve_to_vec(param_syms, init))
    end

    bounds_vec = nothing
    if bounds !== nothing
        lower, upper = bounds
        _fit_curve_require_param_keys(lower, param_syms, "lower bounds")
        _fit_curve_require_param_keys(upper, param_syms, "upper bounds")
        lower_vec = Float64.(_fit_curve_to_vec(param_syms, lower))
        upper_vec = Float64.(_fit_curve_to_vec(param_syms, upper))
        if any(lower_vec .> upper_vec)
            throw(ArgumentError("bounds must satisfy lower <= upper component-wise"))
        end
        bounds_vec = (lower_vec, upper_vec)
    end

    xs = x_values === nothing ? collect(1:length(y_values)) : x_values
    if length(xs) != length(y_values)
        throw(DimensionMismatch("x_values length must match y_values length"))
    end

    bounds_mode = _fit_curve_make_bounds_mode(bounds_vec)
    objective_mode = _fit_curve_make_objective_mode(stds_clean, std_fn, minimize_χ²)
    diagnostics_mode = _fit_curve_make_diagnostics_mode(stds_clean, std_fn)
    cfg = _FitConfig(
        param_syms,
        init_vec,
        bounds_mode,
        objective_mode,
        diagnostics_mode,
        multistart,
        rng === nothing ? Random.GLOBAL_RNG : rng,
        optim_options,
        method,
        autodiff,
        verbose,
        verbose_step,
        return_cov,
        goodness_of_fit,
        bootstrap_errorbars,
        n_boot,
    )
    return xs, y_values, cfg
end

"""
    _fit_curve_objective(f, xs, ys_local, param_syms; kwargs...)

Build the least-squares objective used by `fit_curve`.

# Arguments
- `f`: Model function `f(x, params)`.
- `xs`: Input coordinates.
- `ys_local`: Target values for the objective evaluation.
- `param_syms`: Ordered parameter symbols.

# Keyword Arguments
- `bounds_vec`: Optional bound vectors `(lower, upper)` applied by clamping.
- `minimize_χ²`: Whether to use weighted residuals.
- `stds`: Standard deviations used for weighted residuals.
- `std_fn`: Optional custom standard-deviation callback.

# Returns
- `Function`: Objective function that maps a parameter vector to a scalar loss."""
function _fit_curve_objective(
    f::Function,
    xs,
    ys_local,
    param_syms::Tuple{Vararg{Symbol}};
    bounds_vec = nothing,
    minimize_χ²::Bool = false,
    stds::Union{Nothing,AbstractVector{<:Real}} = nothing,
    std_fn::Union{Nothing,Function} = nothing,
)
    bounds_mode = _fit_curve_make_bounds_mode(bounds_vec)
    weighting_mode = _fit_curve_make_objective_mode(stds, std_fn, minimize_χ²)
    return function (x)
        bundle = _fit_curve_residual_bundle(
            x,
            ys_local,
            f,
            xs,
            param_syms,
            bounds_mode,
            weighting_mode,
        )
        return _fit_curve_objective_value(bundle.residuals, bundle.σ)
    end
end

"""
    _fit_curve_solve(x0, ys_local, f, xs, param_syms; kwargs...)

Run one optimization solve for the curve-fitting objective.

# Arguments
- `x0`: Initial parameter vector.
- `ys_local`: Target values used in this solve.
- `f`: Model function `f(x, params)`.
- `xs`: Input coordinates.
- `param_syms`: Ordered parameter symbols.

# Keyword Arguments
- `bounds_vec`: Optional bound vectors `(lower, upper)` applied by clamping.
- `minimize_χ²`: Whether to use weighted residuals.
- `stds`: Standard deviations used for weighted residuals.
- `std_fn`: Optional custom standard-deviation callback.
- `optim_options`: Options object passed to `Optim.optimize`.
- `method`: Optimizer method.
- `autodiff`: Optional autodiff mode forwarded to `Optim.optimize`.

# Returns
- `(xopt, fmin)`: Minimizer vector and objective minimum value."""
function _fit_curve_solve(
    x0,
    ys_local,
    f::Function,
    xs,
    param_syms::Tuple{Vararg{Symbol}};
    bounds_vec = nothing,
    minimize_χ²::Bool = false,
    stds::Union{Nothing,AbstractVector{<:Real}} = nothing,
    std_fn::Union{Nothing,Function} = nothing,
    optim_options = nothing,
    method = Optim.NelderMead(),
    autodiff = nothing,
)
    bounds_mode = _fit_curve_make_bounds_mode(bounds_vec)
    weighting_mode = _fit_curve_make_objective_mode(stds, std_fn, minimize_χ²)
    obj = _fit_curve_objective(
        f,
        xs,
        ys_local,
        param_syms;
        bounds_vec = bounds_mode isa _NoBounds ? nothing : (bounds_mode.lower, bounds_mode.upper),
        minimize_χ² = weighting_mode isa _WeightedMode,
        stds = weighting_mode isa _WeightedMode ? weighting_mode.stds : nothing,
        std_fn = weighting_mode isa _WeightedMode ? weighting_mode.std_fn : nothing,
    )
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

"""
    _fit_curve_score(xopt, ys, f, xs, param_syms; kwargs...)

Compute the model score for one parameter vector.

Returns reduced `χ²` when `stds` are provided, otherwise relative RMS residual.

# Arguments
- `xopt`: Candidate parameter vector.
- `ys`: Target values.
- `f`: Model function `f(x, params)`.
- `xs`: Input coordinates.
- `param_syms`: Ordered parameter symbols.

# Keyword Arguments
- `bounds_vec`: Optional bound vectors `(lower, upper)` applied by clamping.
- `stds`: Standard deviations used for weighted residuals.
- `std_fn`: Optional custom standard-deviation callback.

# Returns
- `Real`: Score value (`χ²` or relative RMS depending on inputs)."""
function _fit_curve_score(
    xopt,
    ys::AbstractVector{<:Real},
    f::Function,
    xs,
    param_syms::Tuple{Vararg{Symbol}};
    bounds_vec = nothing,
    stds::Union{Nothing,AbstractVector{<:Real}} = nothing,
    std_fn::Union{Nothing,Function} = nothing,
)
    bounds_mode = _fit_curve_make_bounds_mode(bounds_vec)
    weighting_mode = _fit_curve_make_diagnostics_mode(stds, std_fn)
    bundle = _fit_curve_residual_bundle(
        xopt,
        ys,
        f,
        xs,
        param_syms,
        bounds_mode,
        weighting_mode,
    )
    p = length(param_syms)
    return _fit_curve_score_value(bundle.residuals, ys, bundle.σ, p)
end

"""
    _fit_curve_multistart_candidate(init_vec, bounds_vec, rng)

Sample one multistart initialization vector.

# Arguments
- `init_vec`: Base parameter vector.
- `bounds_vec`: Optional bound vectors `(lower, upper)`; when present, candidates are sampled within/clamped to bounds.
- `rng`: Random number generator.

# Returns
- `Vector{Float64}`: Candidate starting vector for an optimizer run."""
function _fit_curve_multistart_candidate(
    init_vec::AbstractVector{<:Real},
    bounds_vec,
    rng::Random.AbstractRNG,
)
    return _fit_curve_multistart_candidate(init_vec, _fit_curve_make_bounds_mode(bounds_vec), rng)
end

function _fit_curve_multistart_candidate(
    init_vec::AbstractVector{<:Real},
    bounds_mode::_FitBounds,
    rng::Random.AbstractRNG,
)
    p = length(init_vec)
    if bounds_mode isa _NoBounds
        scale = [x == 0 ? 1.0 : abs(x) for x in init_vec]
        return init_vec .+ (2 .* rand(rng, p) .- 1) .* scale
    end
    lower = bounds_mode.lower
    upper = bounds_mode.upper
    x = similar(init_vec, Float64)
    @inbounds for j in 1:p
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
    return x
end

"""
    _fit_curve_jacobian_fd(xs, params, f, param_syms; eps=1e-6)

Approximate the model Jacobian by forward finite differences.

# Arguments
- `xs`: Input coordinates.
- `params`: Parameter values as a `NamedTuple`.
- `f`: Model function `f(x, params)`.
- `param_syms`: Ordered parameter symbols.

# Keyword Arguments
- `eps`: Relative finite-difference step factor.

# Returns
- `Matrix{Float64}`: Jacobian matrix with shape `(length(xs), length(param_syms))`."""
function _fit_curve_jacobian_fd(
    xs,
    params::NamedTuple,
    f::Function,
    param_syms::Tuple{Vararg{Symbol}};
    eps::Real = 1e-6,
)
    p = length(param_syms)
    base_preds = f.(xs, Ref(params))
    J = Matrix{Float64}(undef, length(xs), p)
    for (j, sym) in enumerate(param_syms)
        v = getfield(params, sym)
        step = (v == 0 ? 1.0 : abs(v)) * eps
        pvec = _fit_curve_to_vec(param_syms, params)
        pvec[j] = v + step
        ppert = _fit_curve_to_nt(param_syms, pvec)
        preds = f.(xs, Ref(ppert))
        J[:, j] = (preds .- base_preds) ./ step
    end
    return J
end

"""
    _fit_curve_cov_and_stderr(xs, ys, params, xopt, solve_fn, f, param_syms; kwargs...)

Estimate parameter covariance and standard errors for a fitted model.

Uses either bootstrap resampling or local linearization via finite-difference Jacobian.

# Arguments
- `xs`: Input coordinates.
- `ys`: Target values.
- `params`: Best-fit parameters as `NamedTuple`.
- `xopt`: Best-fit parameter vector.
- `solve_fn`: Solver callback `(x0, ys_local) -> (xopt, fmin)` used for bootstrap refits.
- `f`: Model function `f(x, params)`.
- `param_syms`: Ordered parameter symbols.

# Keyword Arguments
- `stds`: Standard deviations used for weighted fits/bootstrap noise.
- `std_fn`: Optional custom standard-deviation callback.
- `bootstrap_errorbars`: Whether to estimate covariance by bootstrap.
- `n_boot`: Number of bootstrap resamples.
- `rng`: Random number generator for bootstrap sampling.
- `bounds_vec`: Optional bound vectors `(lower, upper)` applied to bootstrap refits.

# Returns
- `(cov, stderr)`: Covariance matrix and `NamedTuple` of parameter standard errors.

# Throws
- `ArgumentError`: Raised when `bootstrap_errorbars=true` and `stds` is not provided.
- `DomainError`: Raised for invalid bootstrap configuration (`n_boot < 2`)."""
function _fit_curve_cov_and_stderr(
    xs,
    ys,
    params::NamedTuple,
    xopt,
    solve_fn::Function,
    f::Function,
    param_syms::Tuple{Vararg{Symbol}};
    stds::Union{Nothing,AbstractVector{<:Real}} = nothing,
    std_fn::Union{Nothing,Function} = nothing,
    bootstrap_errorbars::Bool = false,
    n_boot::Int = 200,
    rng::Union{Nothing,Random.AbstractRNG} = nothing,
    bounds_vec = nothing,
)
    bounds_mode = _fit_curve_make_bounds_mode(bounds_vec)
    diagnostics_mode = _fit_curve_make_diagnostics_mode(stds, std_fn)
    weighted = diagnostics_mode isa _WeightedMode
    if bootstrap_errorbars
        if !weighted
            throw(ArgumentError("bootstrap_errorbars requires stds to be provided"))
        end
        if !(n_boot >= 2)
            throw(DomainError(n_boot, "n_boot must be >= 2 when bootstrap_errorbars=true"))
        end
        rng_local = rng === nothing ? Random.GLOBAL_RNG : rng
        p = length(param_syms)
        samples = Matrix{Float64}(undef, n_boot, p)
        base_bundle = _fit_curve_residual_bundle(
            xopt,
            ys,
            f,
            xs,
            param_syms,
            bounds_mode,
            diagnostics_mode,
        )
        σ_base = base_bundle.σ
        for i in 1:n_boot
            ys_boot = ys .+ randn(rng_local, length(ys)) .* σ_base
            xopt_boot, _ = solve_fn(xopt, ys_boot)
            xopt_boot = _fit_curve_apply_bounds(xopt_boot, bounds_mode)
            samples[i, :] = xopt_boot
        end
        cov = Statistics.cov(samples)
        stderr = sqrt.(abs.(LinearAlgebra.diag(cov)))
        return cov, _fit_curve_to_nt(param_syms, stderr)
    end

    J = _fit_curve_jacobian_fd(xs, params, f, param_syms)
    if weighted
        bundle = _fit_curve_residual_bundle(
            xopt,
            ys,
            f,
            xs,
            param_syms,
            bounds_mode,
            diagnostics_mode,
        )
        preds = bundle.preds
        σ = bundle.σ
        W = LinearAlgebra.Diagonal(1.0 ./ (σ .^ 2))
        JT_W_J = J' * W * J
        dof = length(ys) - length(param_syms)
        s2 = dof > 0 ? sum(((ys .- preds) ./ σ) .^ 2) / dof : 1.0
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
    stderr = sqrt.(abs.(LinearAlgebra.diag(cov)))
    return cov, _fit_curve_to_nt(param_syms, stderr)
end

"""
    _fit_curve_run_multistart(ys, f, xs, cfg)

Run the multistart optimization phase for curve fitting.

# Arguments
- `ys`: Observed response values.
- `f`: Model function `f(x, params)`.
- `xs`: Input coordinates.
- `cfg`: Prepared fit configuration.

# Returns
- `(run, solve_local)`: Best-run summary and the solver callback reused by post-processing."""
function _fit_curve_run_multistart(ys, f::Function, xs, cfg::_FitConfig)
    if cfg.multistart < 1
        throw(ArgumentError("multistart must be >= 1"))
    end
    if length(cfg.init_vec) != length(cfg.param_syms)
        throw(ArgumentError("init vector length must match param_syms length"))
    end
    bounds_vec = _fit_curve_bounds_tuple(cfg.bounds_mode)
    solve_local(x0, ys_local) = _fit_curve_solve(
        x0,
        ys_local,
        f,
        xs,
        cfg.param_syms;
        bounds_vec = bounds_vec,
        minimize_χ² = _fit_curve_has_stds(cfg.objective_mode),
        stds = _fit_curve_stds(cfg.objective_mode),
        std_fn = _fit_curve_std_fn(cfg.objective_mode),
        optim_options = cfg.optim_options,
        method = cfg.method,
        autodiff = cfg.autodiff,
    )

    best_x, best_f = solve_local(cfg.init_vec, ys)
    best_score = _fit_curve_score(
        best_x,
        ys,
        f,
        xs,
        cfg.param_syms;
        bounds_vec = bounds_vec,
        stds = _fit_curve_stds(cfg.diagnostics_mode),
        std_fn = _fit_curve_std_fn(cfg.diagnostics_mode),
    )
    label = _fit_curve_label(cfg.diagnostics_mode)
    step = if cfg.verbose
        cfg.verbose_step === nothing ? max(1, round(Int, cfg.multistart * 0.1)) : max(1, cfg.verbose_step)
    else
        0
    end
    if cfg.verbose
        println("multistart 1: ", label, " = ", best_score, " (best = ", best_score, ")")
        flush(stdout)
    end

    for i in 2:cfg.multistart
        x0 = _fit_curve_multistart_candidate(cfg.init_vec, cfg.bounds_mode, cfg.rng)
        xopt, fmin = solve_local(x0, ys)
        score = _fit_curve_score(
            xopt,
            ys,
            f,
            xs,
            cfg.param_syms;
            bounds_vec = bounds_vec,
            stds = _fit_curve_stds(cfg.diagnostics_mode),
            std_fn = _fit_curve_std_fn(cfg.diagnostics_mode),
        )
        if fmin < best_f
            best_x, best_f = xopt, fmin
            best_score = score
        end
        if cfg.verbose && (i % step == 0 || i == cfg.multistart)
            println("multistart ", i, ": ", label, " = ", score, " (best = ", best_score, ")")
            flush(stdout)
        end
    end

    if cfg.verbose
        println("multistart done: best ", label, " = ", best_score)
        flush(stdout)
    end
    return _FitRunSummary(Vector{Float64}(best_x), best_f, best_score), solve_local
end

"""
    _fit_curve_finalize_result(ys, f, xs, cfg, run, solve_local)

Build the internal typed result after optimization.

# Arguments
- `ys`: Observed response values.
- `f`: Model function `f(x, params)`.
- `xs`: Input coordinates.
- `cfg`: Prepared fit configuration.
- `run`: Output from `_fit_curve_run_multistart`.
- `solve_local`: Solver callback used for optional bootstrap covariance.

# Returns
- `_FitResult`: Internal result container with fitted parameters and optional diagnostics."""
function _fit_curve_finalize_result(
    ys,
    f::Function,
    xs,
    cfg::_FitConfig,
    run::_FitRunSummary,
    solve_local::Function,
)
    if length(run.best_x) != length(cfg.param_syms)
        throw(ArgumentError("best parameter vector length must match param_syms length"))
    end
    xopt = _fit_curve_apply_bounds(run.best_x, cfg.bounds_mode)
    bundle = _fit_curve_residual_bundle(
        xopt,
        ys,
        f,
        xs,
        cfg.param_syms,
        cfg.bounds_mode,
        cfg.diagnostics_mode,
    )
    params = bundle.params
    residuals = bundle.residuals

    χ² = nothing
    if _fit_curve_has_stds(cfg.diagnostics_mode) && cfg.goodness_of_fit
        dof = length(ys) - length(cfg.param_syms)
        if dof <= 0
            @warn "chi-squared undefined: degrees of freedom <= 0" dof = dof
            χ² = NaN
        else
            χ² = sum((residuals ./ bundle.σ) .^ 2) / dof
        end
    end

    cov = nothing
    stderr_nt = nothing
    if cfg.return_cov
        cov, stderr_nt = _fit_curve_cov_and_stderr(
            xs,
            ys,
            params,
            xopt,
            solve_local,
            f,
            cfg.param_syms;
            stds = _fit_curve_stds(cfg.diagnostics_mode),
            std_fn = _fit_curve_std_fn(cfg.diagnostics_mode),
            bootstrap_errorbars = cfg.bootstrap_errorbars,
            n_boot = cfg.n_boot,
            rng = cfg.rng,
            bounds_vec = _fit_curve_bounds_tuple(cfg.bounds_mode),
        )
    end
    return _FitResult(params, residuals, χ², cov, stderr_nt)
end

"""
    _fit_curve_public_output(result, ys, cfg)

Convert the internal typed result into the backward-compatible public return shape.

# Arguments
- `result`: Internal fit result.
- `ys`: Observed response values.
- `cfg`: Prepared fit configuration.

# Returns
- `NamedTuple`: Public result matching `fit_curve` output conventions."""
function _fit_curve_public_output(result::_FitResult, ys, cfg::_FitConfig)
    if length(result.residuals) != length(ys)
        throw(ArgumentError("residuals length must match ys length"))
    end
    if cfg.goodness_of_fit
        if _fit_curve_has_stds(cfg.diagnostics_mode)
            if cfg.return_cov
                return (
                    params = result.params,
                    rel_residuals = result.residuals ./ ys,
                    χ² = result.χ²,
                    cov = result.cov,
                    stderr = result.stderr,
                )
            end
            return (params = result.params, rel_residuals = result.residuals ./ ys, χ² = result.χ²)
        end
        if cfg.return_cov
            return (
                params = result.params,
                rel_residuals = result.residuals ./ ys,
                cov = result.cov,
                stderr = result.stderr,
            )
        end
        return (params = result.params, rel_residuals = result.residuals ./ ys)
    end
    if cfg.return_cov
        return (params = result.params, cov = result.cov, stderr = result.stderr)
    end
    return result.params
end

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
- `x_values`: Optional x coordinates aligned with `y_values`.
- `stds`: Optional standard deviations for weighted diagnostics and uncertainty estimation.
- `minimize_χ²`: If true, optimize weighted residuals (requires `stds`).
- `init`: Optional initial parameters as `NamedTuple`.
- `bounds`: Optional bounds as `(lower::NamedTuple, upper::NamedTuple)`.
- `goodness_of_fit`: If true, include residual diagnostics (`rel_residuals` and optionally `χ²`).
- `ϵ`: Positive floor used when replacing zero or invalid standard deviations.
- `multistart`: Number of multistart optimization runs (`>= 1`).
- `rng`: Random number generator used for stochastic steps.
- `optim_options`: Optional `Optim.Options` passed to `Optim.optimize`.
- `method`: Optimization method passed to `Optim.optimize`.
- `autodiff`: Optional autodiff mode for supported optimizers.
- `verbose`: If true, print multistart progress.
- `std_fn`: Optional callback `(ys, preds, stds, params) -> σ` for effective standard deviations.
- `verbose_step`: Optional progress print stride.
- `return_cov`: If true, include covariance and standard-error estimates.
- `bootstrap_errorbars`: If true, estimate covariance by bootstrap instead of local linearization.
- `n_boot`: Number of bootstrap resamples (must be `>= 2` when `bootstrap_errorbars=true`).

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `DomainError`: Raised for invalid numeric option values.
- `DimensionMismatch`: Raised when provided arrays have inconsistent lengths."""
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
    xs, ys, cfg = _fit_curve_prepare_inputs(
        y_values,
        param_syms;
        x_values = x_values,
        stds = stds,
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
        bootstrap_errorbars = bootstrap_errorbars,
        n_boot = n_boot,
    )
    run, solve_local = _fit_curve_run_multistart(ys, f, xs, cfg)
    result = _fit_curve_finalize_result(ys, f, xs, cfg, run, solve_local)
    return _fit_curve_public_output(result, ys, cfg)
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
- `bin_lo`: First bin index (inclusive) used for fitting.
- `bin_hi`: Last bin index (inclusive) used for fitting.

# Keyword Arguments
- `stds`: Optional standard deviations for weighted fitting.
- `minimize_χ²`: If true, fit by weighted residuals (requires `stds`).
- `x_values`: Optional x coordinates aligned with `y_values`.
- `init`: Optional initial parameter `NamedTuple`.
- `bounds`: Optional parameter bounds as `(lower::NamedTuple, upper::NamedTuple)`.
- `goodness_of_fit`: If true, include fit diagnostics in the return value.
- `ϵ`: Positive floor used when replacing zero standard deviations.
- `multistart`: Number of multistart optimization runs (`>= 1`).
- `rng`: RNG used for multistart sampling.
- `optim_options`: Optional `Optim.Options` passed to `Optim.optimize`.
- `method`: Optimization method passed to `Optim.optimize`.
- `autodiff`: Optional autodiff mode for supported optimizers.
- `verbose`: If true, print multistart progress.
- `std_fn`: Optional callback to compute effective standard deviations.
- `verbose_step`: Optional progress print stride.
- `return_cov`: If true, include covariance and standard-error estimates.

# Returns
- `result`: Return value from `fit_curve` on the selected bin slice.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `DomainError`: Raised for invalid numeric option values.
- `DimensionMismatch`: Raised when provided arrays have inconsistent lengths."""
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
