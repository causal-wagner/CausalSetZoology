"""
    relative_change(a::Real, b::Real)::Float64

Relative change between two positive scalars: |a-b|/(a+b).

# Arguments
- `a`: First positive scalar.
- `b`: Second positive scalar.

# Returns
- `result::Float64`: Output of `relative_change` with type annotation `Float64`.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
"""
function relative_change(a::Real, b::Real)::Float64
    if !(a > 0 && b > 0)
        throw(DomainError((a, b), "relative_change requires positive scalars"))
    end
    return abs(a - b) / (a + b)
end

"""
    bin_scalar_pairs(pairs::Vector{<:Tuple{Any,Real}}, num_bins::Union{Nothing,Int}, bin_edges::Union{Nothing,Vector{<:Real}})

Group `(value, scalar)` pairs into bins. Returns a vector of `(bin_key, values)` pairs.
If `num_bins === nothing`, uses exact scalar values.
If `bin_edges` is provided, uses those edges; otherwise computes edges from the scalars.
If `num_bins` equals the number of distinct scalar labels, preserves exact scalar
labels instead of midpoint bin centers.

# Arguments
- `pairs`: Input (value, scalar) pairs to group.
- `num_bins`: Bin selection or binning control parameter.
- `bin_edges`: Bin selection or binning control parameter.

# Returns
- `result`: Vector of (bin_center, values) tuples sorted by bin center.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
"""
function bin_scalar_pairs(
    pairs::AbstractVector{<:Tuple{T,S}},
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,Vector{<:Real}} = nothing,
) where {T,S<:Real}
    isempty(pairs) && return Vector{Tuple{Real,Vector{T}}}()

    scalars = [s for (_, s) in pairs]
    if num_bins !== nothing
        if !(num_bins ≥ 1)
            throw(DomainError(num_bins, "num_bins must be >= 1"))
        end
    end
    if num_bins !== nothing && bin_edges === nothing
        vmin, vmax = minimum(scalars), maximum(scalars)
        if vmin == vmax
            bin_edges = [vmin, vmax + 1e-12]
        else
            bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
        end
    end
    use_exact_labels = num_bins !== nothing && length(unique(scalars)) == num_bins

    groups = Dict{Real,Vector{T}}()
    for (v, s) in pairs
        key = if num_bins === nothing
            s
        elseif use_exact_labels
            s
        else
            idx = searchsortedlast(bin_edges, s)
            idx = clamp(idx, 1, length(bin_edges) - 1)
            (bin_edges[idx] + bin_edges[idx + 1]) / 2
        end
        get!(groups, key, T[])
        push!(groups[key], v)
    end

    keys_sorted = sort(collect(keys(groups)))
    return [(k, groups[k]) for k in keys_sorted]
end

struct _ScalarBinContext{T,S<:Real}
    pairs::Vector{Tuple{T,Float64}}
    bins::Vector{Tuple{S,Vector{T}}}
end

"""
    _typed_scalar_pairs(pairs_raw)

Validate and materialize `(value, scalar)` pairs in a concrete internal format.

This helper enforces type `Vector{Tuple{T,Float64}}` for pairs, where `T` is 
inferred from the first value.

# Arguments
- `pairs_raw`: Raw scalar-paired dataset entries.

# Returns
- `result`: Concrete typed `(value, Float64)` pairs.

# Throws
- `ArgumentError`: Raised when an entry is not pair-like (missing first/second slot).
- `TypeError`: Raised when scalar slot is non-`Real` or value types are inconsistent.
"""
function _typed_scalar_pairs(pairs_raw::AbstractVector)
    p1 = pairs_raw[1]
    v1 = try
        p1[1]
    catch
        throw(ArgumentError("expected pair-like entries `(value, scalar)`"))
    end
    s1 = try
        p1[2]
    catch
        throw(ArgumentError("expected pair-like entries `(value, scalar)`"))
    end
    if !(s1 isa Real)
        throw(TypeError(:distinguishability, "scalar", Real, s1))
    end

    Tval = typeof(v1)
    pairs = Vector{Tuple{Tval,Float64}}(undef, length(pairs_raw))
    pairs[1] = (v1, Float64(s1))

    for i in 2:length(pairs_raw)
        p = pairs_raw[i]
        v = try
            p[1]
        catch
            throw(ArgumentError("expected pair-like entries `(value, scalar)`"))
        end
        s = try
            p[2]
        catch
            throw(ArgumentError("expected pair-like entries `(value, scalar)`"))
        end
        if !(v isa Tval)
            throw(TypeError(:distinguishability, "value", Tval, v))
        end
        if !(s isa Real)
            throw(TypeError(:distinguishability, "scalar", Real, s))
        end
        pairs[i] = (v, Float64(s))
    end
    return pairs
end

"""
    _prepare_scalar_bin_context(data, caller; num_bins=nothing, ref=nothing)

Validate scalar-paired input and build the shared scalar-bin context used by
scalar-bin distinguishability routines.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `caller`: Caller name for validation error messages.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `ref`: Optional reference dataset used only for validation checks.

# Returns
- `result`: Internal `_ScalarBinContext` containing normalized pairs and binned groups.

# Throws
- `DimensionMismatch`: Raised when the top-level dataset count is not one.
- `ArgumentError`: Raised for empty dataset or empty reference set.
- `DomainError`: Raised when `num_bins` is provided but invalid.
"""
function _prepare_scalar_bin_context(
    data::AbstractVector{<:AbstractVector},
    caller::AbstractString;
    num_bins::Union{Nothing,Int} = nothing,
    ref::Union{Nothing,AbstractVector} = nothing,
)
    if !(length(data) == 1)
        throw(DimensionMismatch("$caller expects one dataset (one path), got length(data)=$(length(data))"))
    end
    pairs_raw = data[1]
    if isempty(pairs_raw)
        throw(ArgumentError("dataset must be non-empty"))
    end
    if ref !== nothing && isempty(ref)
        throw(ArgumentError("reference set must be non-empty"))
    end
    if num_bins !== nothing && !(num_bins >= 1)
        throw(DomainError(num_bins, "num_bins must be >= 1"))
    end

    pairs = _typed_scalar_pairs(pairs_raw)
    scalars = [s for (_, s) in pairs]
    bin_edges = nothing
    if num_bins !== nothing
        vmin, vmax = minimum(scalars), maximum(scalars)
        if vmin == vmax
            bin_edges = [vmin, vmax + 1e-12]
        else
            bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
        end
    end
    bins = bin_scalar_pairs(pairs, num_bins, bin_edges)
    return _ScalarBinContext(pairs, bins)
end

"""
    _map_scalar_bin_pairs(compute, bins)

Apply `compute(s1, vals1, s2, vals2)` to every unordered pair of scalar bins.

# Arguments
- `compute`: Mapping function for one bin pair.
- `bins`: Scalar-bin collection.

# Returns
- `result`: Vector of outputs returned by `compute` for all bin pairs.
"""
function _map_scalar_bin_pairs(compute::F, bins::AbstractVector) where {F<:Function}
    out = Vector{NamedTuple}()
    for i in 1:length(bins)
        s1, vals1 = bins[i]
        for j in (i + 1):length(bins)
            s2, vals2 = bins[j]
            push!(out, compute(s1, vals1, s2, vals2))
        end
    end
    return out
end

"""
    _map_scalar_bin_reference(compute, bins)

Apply `compute(s, vals)` independently to each scalar bin.

# Arguments
- `compute`: Mapping function for one bin.
- `bins`: Scalar-bin collection.

# Returns
- `result`: Vector of outputs returned by `compute` for each bin.
"""
function _map_scalar_bin_reference(compute::F, bins::AbstractVector) where {F<:Function}
    out = Vector{NamedTuple}()
    for (s, vals) in bins
        push!(out, compute(s, vals))
    end
    return out
end

"""
    scalar_bin_distinguishability(data::Vector{Vector{Tuple{T,Real}}}; num_bins=nothing)

Given one dataset (top-level vector must have length 1) of `(value, scalar)` pairs
from `load_histograms_from_paths(..., scalar)` or `load_field_with_scalar`, bin by
scalar (if `num_bins` is set) and compute distinguishability for all bin pairs.

Returns a vector of `(s1, s2, rel_change, D)`.

# Arguments
- `data`: Input dataset(s) consumed by this method.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.

# Returns
- `result`: Vector of named tuples with distinguishability metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_distinguishability(
    data::Vector{Vector{Tuple{T,Real}}};
    num_bins::Union{Nothing,Int} = nothing,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
) where {T}
    ctx = _prepare_scalar_bin_context(data, "scalar_bin_distinguishability"; num_bins = num_bins)
    bins = ctx.bins
    n_bins = length(bins)
    total = n_bins * (n_bins - 1) ÷ 2
    tasks = Vector{Tuple{Int,Int}}(undef, total)
    t = 1
    for i in 1:n_bins
        for j in (i + 1):n_bins
            tasks[t] = (i, j)
            t += 1
        end
    end
    out = Vector{NamedTuple}(undef, total)
    compute_task = function (k::Int)
        i, j = tasks[k]
        s1, vals1 = bins[i]
        s2, vals2 = bins[j]
        res = energy_based_histogram_distinguishability(vals1, vals2; distance = distance, verbose = verbose)
        return (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D = res.D)
    end
    n_threads = Threads.maxthreadid()
    if n_threads > 1 && total > 1
        Threads.@threads for tid in 1:n_threads
            for k in tid:n_threads:total
                @inbounds out[k] = compute_task(k)
            end
        end
    else
        for k in 1:total
            @inbounds out[k] = compute_task(k)
        end
    end
    return out
end

"""
    scalar_bin_distinguishability(data::Vector{Vector{Tuple{T,Real}}}, num_draws::Int; num_bins=nothing, rng=...)

See `scalar_bin_distinguishability(data; num_bins=...)`.

This overload uses Monte Carlo distinguishability with `num_draws` samples and
adds uncertainty output: each result has fields
`(s1, s2, rel_change, D, std)`.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Vector of named tuples with distinguishability metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_distinguishability(
    data::Vector{Vector{Tuple{T,Real}}},
    num_draws::Int;
    num_bins::Union{Nothing,Int} = nothing,
    rng = Random.default_rng(),
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
) where {T}
    ctx = _prepare_scalar_bin_context(data, "scalar_bin_distinguishability"; num_bins = num_bins)
    bins = ctx.bins
    n_bins = length(bins)
    total = n_bins * (n_bins - 1) ÷ 2
    tasks = Vector{Tuple{Int,Int}}(undef, total)
    t = 1
    for i in 1:n_bins
        for j in (i + 1):n_bins
            tasks[t] = (i, j)
            t += 1
        end
    end
    seeds = rand(rng, UInt64, total)
    out = Vector{NamedTuple}(undef, total)
    compute_task = function (k::Int)
        i, j = tasks[k]
        s1, vals1 = bins[i]
        s2, vals2 = bins[j]
        rng_k = Random.Xoshiro(seeds[k])
        res = energy_based_histogram_distinguishability(vals1, vals2, num_draws; rng = rng_k, distance = distance, verbose = verbose)
        return (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D = res.D, std = res.std)
    end
    n_threads = Threads.maxthreadid()
    if n_threads > 1 && total > 1
        Threads.@threads for tid in 1:n_threads
            for k in tid:n_threads:total
                @inbounds out[k] = compute_task(k)
            end
        end
    else
        for k in 1:total
            @inbounds out[k] = compute_task(k)
        end
    end
    return out
end

"""
    scalar_bin_distinguishability(data::Vector{Vector{Tuple{T,Real}}}, ref::AbstractVector; num_bins=nothing)

See `scalar_bin_distinguishability(data; num_bins=...)`.

This overload compares each scalar bin to a fixed reference sample `ref` and
returns `(scalar, D)` entries.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `ref`: Reference sample used for bin-wise comparison.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.

# Returns
- `result`: Vector of named tuples with distinguishability metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_distinguishability(
    data::Vector{Vector{Tuple{T,Real}}},
    ref::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
) where {T}
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_distinguishability";
        num_bins = num_bins,
        ref = ref,
    )
    bins = ctx.bins
    total = length(bins)
    out = Vector{NamedTuple}(undef, total)
    compute_idx = function (k::Int)
        s, vals = bins[k]
        res = energy_based_histogram_distinguishability(vals, ref; distance = distance, verbose = verbose)
        return (scalar = s, D = res.D)
    end
    n_threads = Threads.maxthreadid()
    if n_threads > 1 && total > 1
        Threads.@threads for tid in 1:n_threads
            for k in tid:n_threads:total
                @inbounds out[k] = compute_idx(k)
            end
        end
    else
        for k in 1:total
            @inbounds out[k] = compute_idx(k)
        end
    end
    return out
end

"""
    scalar_bin_distinguishability(data::Vector{Vector{Tuple{T,Real}}}, ref::AbstractVector, num_draws::Int; num_bins=nothing, rng=...)

See `scalar_bin_distinguishability(data, ref; num_bins=...)`.

This overload uses Monte Carlo distinguishability with `num_draws` and returns
`(scalar, D, std)` entries.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `ref`: Reference sample used for bin-wise comparison.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Vector of named tuples with distinguishability metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_distinguishability(
    data::Vector{Vector{Tuple{T,Real}}},
    ref::AbstractVector,
    num_draws::Int;
    num_bins::Union{Nothing,Int} = nothing,
    rng = Random.default_rng(),
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
) where {T}
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_distinguishability";
        num_bins = num_bins,
        ref = ref,
    )
    bins = ctx.bins
    total = length(bins)
    seeds = rand(rng, UInt64, total)
    out = Vector{NamedTuple}(undef, total)
    compute_idx = function (k::Int)
        s, vals = bins[k]
        rng_k = Random.Xoshiro(seeds[k])
        res = energy_based_histogram_distinguishability(vals, ref, num_draws; rng = rng_k, distance = distance, verbose = verbose)
        return (scalar = s, D = res.D, std = res.std)
    end
    n_threads = Threads.maxthreadid()
    if n_threads > 1 && total > 1
        Threads.@threads for tid in 1:n_threads
            for k in tid:n_threads:total
                @inbounds out[k] = compute_idx(k)
            end
        end
    else
        for k in 1:total
            @inbounds out[k] = compute_idx(k)
        end
    end
    return out
end

"""
    scalar_bin_distinguishability_permutation(data::Vector{Vector{Tuple{T,Real}}}; num_bins=nothing, n_perm=1000, rng=...)

Permutation-test version of `scalar_bin_distinguishability`. Returns a vector of
`(s1, s2, rel_change, D_obs, p_value, z_emp, z_coll, std_Ts)` for each bin pair.

# Arguments
- `data`: Input dataset(s) consumed by this method.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `n_perm`: Number of permutations.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Vector of named tuples with permutation-test metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_distinguishability_permutation(
    data::Vector{Vector{Tuple{T,Real}}};
    num_bins::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    rng = Random.default_rng(),
    progress::Bool = false,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
) where {T}
    ctx = _prepare_scalar_bin_context(data, "scalar_bin_distinguishability_permutation"; num_bins = num_bins)
    n_bins = length(ctx.bins)
    total = n_bins * (n_bins - 1) ÷ 2
    out = Vector{NamedTuple}(undef, total)
    k = 1
    pm = progress ? ProgressMeter.Progress(total; desc = "perm scalar bins") : nothing
    for i in 1:n_bins
        s1, vals1 = ctx.bins[i]
        for j in (i + 1):n_bins
            s2, vals2 = ctx.bins[j]
            res = histogram_distinguishability_permutation(vals1, vals2; n_perm = n_perm, rng = rng, progress = false, distance = distance, verbose = verbose)
            out[k] = (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
            k += 1
            progress && ProgressMeter.next!(pm)
        end
    end
    return out
end

"""
    scalar_bin_distinguishability_permutation(data::Vector{Vector{Tuple{T,Real}}}, num_draws::Int; num_bins=nothing, n_perm=1000, rng=...)

See `scalar_bin_distinguishability_permutation(data; ...)`.

This overload uses Monte Carlo pair sampling (`num_draws`) inside the
permutation test and returns the same fields.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `n_perm`: Number of permutations.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Vector of named tuples with permutation-test metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_distinguishability_permutation(
    data::Vector{Vector{Tuple{T,Real}}},
    num_draws::Int;
    num_bins::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    rng = Random.default_rng(),
    progress::Bool = false,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
) where {T}
    ctx = _prepare_scalar_bin_context(data, "scalar_bin_distinguishability_permutation"; num_bins = num_bins)
    n_bins = length(ctx.bins)
    total = n_bins * (n_bins - 1) ÷ 2
    out = Vector{NamedTuple}(undef, total)
    k = 1
    pm = progress ? ProgressMeter.Progress(total; desc = "perm scalar bins") : nothing
    for i in 1:n_bins
        s1, vals1 = ctx.bins[i]
        for j in (i + 1):n_bins
            s2, vals2 = ctx.bins[j]
            res = histogram_distinguishability_permutation(vals1, vals2, num_draws; n_perm = n_perm, rng = rng, progress = false, distance = distance, verbose = verbose)
            out[k] = (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
            k += 1
            progress && ProgressMeter.next!(pm)
        end
    end
    return out
end

"""
    scalar_bin_distinguishability_permutation(data::Vector{Vector{Tuple{T,Real}}}, ref::AbstractVector; num_bins=nothing, n_perm=1000, rng=...)

See `scalar_bin_distinguishability_permutation(data; ...)`.

This overload compares each scalar bin to reference sample `ref` and returns
`(scalar, D_obs, p_value, z_emp, z_coll, std_Ts)` entries.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `ref`: Reference sample used for bin-wise comparison.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `n_perm`: Number of permutations.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Vector of named tuples with permutation-test metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_distinguishability_permutation(
    data::Vector{Vector{Tuple{T,Real}}},
    ref::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    rng = Random.default_rng(),
    progress::Bool = false,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
) where {T}
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_distinguishability_permutation";
        num_bins = num_bins,
        ref = ref,
    )
    out = Vector{NamedTuple}(undef, length(ctx.bins))
    pm = progress ? ProgressMeter.Progress(length(ctx.bins); desc = "perm scalar bins") : nothing
    for (k, (s, vals)) in enumerate(ctx.bins)
        res = histogram_distinguishability_permutation(vals, ref; n_perm = n_perm, rng = rng, progress = false, distance = distance, verbose = verbose)
        out[k] = (scalar = s, D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
        progress && ProgressMeter.next!(pm)
    end
    return out
end

"""
    scalar_bin_distinguishability_permutation(data::Vector{Vector{Tuple{T,Real}}}, ref::AbstractVector, num_draws::Int; num_bins=nothing, n_perm=1000, rng=...)

See `scalar_bin_distinguishability_permutation(data, ref; ...)`.

This overload uses Monte Carlo pair sampling (`num_draws`) and returns the same
per-bin fields.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `ref`: Reference sample used for bin-wise comparison.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `n_perm`: Number of permutations.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Vector of named tuples with permutation-test metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_distinguishability_permutation(
    data::Vector{Vector{Tuple{T,Real}}},
    ref::AbstractVector,
    num_draws::Int;
    num_bins::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    rng = Random.default_rng(),
    progress::Bool = false,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
) where {T}
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_distinguishability_permutation";
        num_bins = num_bins,
        ref = ref,
    )
    out = Vector{NamedTuple}(undef, length(ctx.bins))
    pm = progress ? ProgressMeter.Progress(length(ctx.bins); desc = "perm scalar bins") : nothing
    for (k, (s, vals)) in enumerate(ctx.bins)
        res = histogram_distinguishability_permutation(vals, ref, num_draws; n_perm = n_perm, rng = rng, progress = false, distance = distance, verbose = verbose)
        out[k] = (scalar = s, D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
        progress && ProgressMeter.next!(pm)
    end
    return out
end

"""
    hellinger_distance(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64

Compute the Hellinger distance between two probability vectors.
Assumes `p` and `q` are nonnegative and have equal length.

# Arguments
- `p`: First probability vector.
- `q`: Second probability vector.

# Returns
- `result::Float64`: Output of `hellinger_distance` with type annotation `Float64`.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
"""
function hellinger_distance(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    if !(length(p) == length(q))
        throw(DimensionMismatch("Hellinger distance requires equal-length vectors"))
    end
    sp = Vector{Float64}(undef, length(p))
    sq = Vector{Float64}(undef, length(q))
    @inbounds for i in eachindex(p, q)
        sp[i] = _sqrt_with_tolerance(Float64(p[i]); name = "probability")
        sq[i] = _sqrt_with_tolerance(Float64(q[i]); name = "probability")
    end
    spp = LinearAlgebra.dot(sp, sp)
    sqq = LinearAlgebra.dot(sq, sq)
    spq = LinearAlgebra.dot(sp, sq)
    d2 = (spp + sqq - 2 * spq) / 2
    return sqrt(max(d2, 0.0))
end

@inline function total_variation_distance(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    if !(length(p) == length(q))
        throw(DimensionMismatch("Total variation distance requires equal-length vectors"))
    end
    s = 0.0
    @inbounds @simd for i in eachindex(p, q)
        s += abs(Float64(p[i]) - Float64(q[i]))
    end
    return 0.5 * s
end

@inline function _normalize_energy_distance(distance::Symbol)::Symbol
    if distance in (:Hellinger, :hellinger)
        return :hellinger
    elseif distance in (:TV, :tv, :TotalVariation, :total_variation)
        return :tv
    end
    throw(DomainError(distance, "distance must be one of :Hellinger or :TV"))
end

@inline function _sqrt_with_tolerance(x::Float64; tol::Float64 = 1e-12, name::AbstractString = "value")::Float64
    if x < -tol
        throw(DomainError(x, "sqrt received significantly negative $name (tol=$tol)"))
    end
    return sqrt(max(x, 0.0))
end

@inline function _hellinger_from_sqrt(
    sp::AbstractVector{<:Real},
    sq::AbstractVector{<:Real},
    spp::Float64,
    sqq::Float64,
)::Float64
    d2 = (spp + sqq - 2 * LinearAlgebra.dot(sp, sq)) / 2
    return sqrt(max(d2, 0.0))
end

function _sqrt_vectors_and_norms(vecs::Vector{<:AbstractVector{<:Real}})
    n = length(vecs)
    sq = Vector{Vector{Float64}}(undef, n)
    norms = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        v = vecs[i]
        s = Vector{Float64}(undef, length(v))
        for j in eachindex(v)
            s[j] = _sqrt_with_tolerance(Float64(v[j]); name = "probability")
        end
        sq[i] = s
        norms[i] = LinearAlgebra.dot(s, s)
    end
    return sq, norms
end

"""
    _prepare_vectors_for_distance(vecs_a, vecs_b)

Internal preprocessing for Hellinger-based distance computations.

Pads all vectors in both datasets to a common length, then trims both datasets
to the maximal nonzero coordinate observed across either set.

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.

# Returns
- `result`: Tuple `(A, B)` of aligned `Vector{Vector{Float64}}` inputs ready for distance computation.
"""
function _prepare_vectors_for_distance(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
)
    maxlen = maximum(length.(vcat(vecs_a, vecs_b)))
    pad_to(v, n) = length(v) == n ? collect(v) : vcat(collect(v), zeros(Float64, n - length(v)))
    A = [pad_to(v, maxlen) for v in vecs_a]
    B = [pad_to(v, maxlen) for v in vecs_b]

    max_nonzero = 0
    for v in A
        idx = findlast(>(0), v)
        if idx !== nothing && idx > max_nonzero
            max_nonzero = idx
        end
    end
    for v in B
        idx = findlast(>(0), v)
        if idx !== nothing && idx > max_nonzero
            max_nonzero = idx
        end
    end
    max_nonzero == 0 && (max_nonzero = maxlen)
    A = [v[1:max_nonzero] for v in A]
    B = [v[1:max_nonzero] for v in B]
    return A, B
end

@inline function _log_projection_info(
    method::AbstractString,
    projected_out::Integer,
    total::Integer,
    cutoff_rtol::Real,
)::Nothing
    @info "Distinguishability projection ($(method)): projected out $(projected_out) / $(total) directions (cutoff_rtol=$(Float64(cutoff_rtol)) * median_eig)"
    return nothing
end

function _project_vectors_pooled_covariance_for_energy(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}};
    covariance_cutoff_rel_median::Float64 = 1e-6,
    method::Symbol = :energy,
    verbose::Bool = false,
)
    if !(covariance_cutoff_rel_median >= 0.0)
        throw(DomainError(covariance_cutoff_rel_median, "covariance_cutoff_rel_median must be nonnegative"))
    end
    A, B = _prepare_vectors_for_distance(vecs_a, vecs_b)
    X = reduce(vcat, permutedims.(vcat(A, B)))
    μ = vec(Statistics.mean(X; dims = 1))
    Xc = X .- transpose(μ)

    Σ = Statistics.cov(Xc; dims = 1)
    eig = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Matrix{Float64}(Σ)))
    λ = eig.values
    U = eig.vectors
    λmed = Statistics.median(λ)
    keep = findall(λi -> λi > covariance_cutoff_rel_median * λmed, λ)
    if isempty(keep)
        keep = [argmax(λ)]
    end
    if verbose
        _log_projection_info(String(method), length(λ) - length(keep), length(λ), covariance_cutoff_rel_median)
    end
    Uk = U[:, keep]
    Z = Xc * Uk

    zmin = vec(minimum(Z; dims = 1))
    shift = max.(-zmin, 0.0) .+ 1e-12

    function row_to_nonnegative(zrow)
        u = Vector{Float64}(zrow) .+ shift
        if !(sum(u) > 0.0)
            u .= 0.0
            u[1] = 1.0
            return u
        end
        return u
    end

    n1 = length(A)
    n2 = length(B)
    Aproj = [row_to_nonnegative(@view Z[i, :]) for i in 1:n1]
    Bproj = [row_to_nonnegative(@view Z[n1 + j, :]) for j in 1:n2]
    return Aproj, Bproj
end

"""
    _distance_matrix_exact(vecs)

Build the exact pairwise Hellinger distance matrix for a vector dataset.

# Arguments
- `vecs`: Vector-valued input data.

# Returns
- `result`: Dense symmetric matrix `D` with `D[i, j] = hellinger_distance(vecs[i], vecs[j])`.
"""
function _distance_matrix_exact(
    vecs::Vector{<:AbstractVector{<:Real}};
    distance::Symbol = :Hellinger,
)
    dist = _normalize_energy_distance(distance)
    n = length(vecs)
    D = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        D[i, i] = 0.0
    end
    if dist == :hellinger
        sq, norms = _sqrt_vectors_and_norms(vecs)
        Threads.@threads for i in 1:(n - 1)
            @inbounds for j in (i + 1):n
                d = _hellinger_from_sqrt(sq[i], sq[j], norms[i], norms[j])
                D[i, j] = d
                D[j, i] = d
            end
        end
    else
        Threads.@threads for i in 1:(n - 1)
            @inbounds for j in (i + 1):n
                d = total_variation_distance(vecs[i], vecs[j])
                D[i, j] = d
                D[j, i] = d
            end
        end
    end
    return D
end

function _pairwise_hellinger_sum_sqrt(
    sqA::Vector{<:AbstractVector{<:Real}},
    normA::Vector{Float64},
    sqB::Vector{<:AbstractVector{<:Real}},
    normB::Vector{Float64},
)::Float64
    n1 = length(sqA)
    n2 = length(sqB)
    partial = zeros(Float64, Threads.maxthreadid())
    Threads.@threads for i in 1:n1
        tid = Threads.threadid()
        ai = sqA[i]
        ai2 = normA[i]
        s = 0.0
        @inbounds for j in 1:n2
            s += _hellinger_from_sqrt(ai, sqB[j], ai2, normB[j])
        end
        partial[tid] += s
    end
    return sum(partial)
end

function _pairwise_total_variation_sum(
    A::Vector{<:AbstractVector{<:Real}},
    B::Vector{<:AbstractVector{<:Real}},
)::Float64
    n1 = length(A)
    n2 = length(B)
    partial = zeros(Float64, Threads.maxthreadid())
    Threads.@threads for i in 1:n1
        tid = Threads.threadid()
        ai = A[i]
        s = 0.0
        @inbounds for j in 1:n2
            s += total_variation_distance(ai, B[j])
        end
        partial[tid] += s
    end
    return sum(partial)
end

@inline function _sample_var_from_sums(sumv::Float64, sumsq::Float64, n::Int)::Float64
    if n <= 1
        return NaN
    end
    meanv = sumv / n
    return max((sumsq - n * meanv * meanv) / (n - 1), 0.0)
end

"""
    _thread_seeds(rng, nt::Int)

Generate `nt` unique `UInt64` seeds from `rng` for thread-local RNG streams.

# Arguments
- `rng`: Base random number generator.
- `nt`: Number of seeds to generate.

# Returns
- `Vector{UInt64}`: Unique per-thread seeds.

# Throws
- `DomainError`: Raised when `nt < 1`.
"""
function _thread_seeds(rng, nt::Int)
    if !(nt >= 1)
        throw(DomainError(nt, "nt must be >= 1"))
    end
    seeds = Vector{UInt64}(undef, nt)
    seen = Set{UInt64}()
    i = 1
    while i <= nt
        s = rand(rng, UInt64)
        if !(s in seen)
            seeds[i] = s
            push!(seen, s)
            i += 1
        end
    end
    return seeds
end

"""
    energy_based_histogram_distinguishability(hists_a, hists_b)

Compute the normalized energy-distance distinguishability D ∈ [0,1] between
two histogram/vector samples. Inputs can be:
- `Vector{<:AbstractDict}` histograms (will be normalized to probabilities),
- `Vector{<:AbstractVector{<:Real}}` already-normalized vectors.

Returns a named tuple `(D = value, E = value)`, where `E` is the unnormalized
energy distance and `D` is the normalized paper-style distinguishability.
For histogram inputs, both sets are normalized with `normalize_hists(..., :probability)`
and then trimmed to the maximal nonzero bin of the union.

# Arguments
- `hists_a`: Histogram input data.
- `hists_b`: Histogram input data.

# Returns
- `result`: Named tuple containing distinguishability `D`, raw energy distance `E`,
  and, for Monte Carlo overloads, `std`.

# Throws
- `DomainError`: Propagated for invalid `num_draws`.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function energy_based_histogram_distinguishability(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
    ;
    covariance_cutoff_rel_median::Float64 = 1e-6,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    if !(!isempty(hists_a) && !isempty(hists_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    norm_a = normalize_hists([hists_a]; normalization = :probability)[1]
    norm_b = normalize_hists([hists_b]; normalization = :probability)[1]

    all_dense = densify_hists(vcat(norm_a, norm_b))
    n1 = length(norm_a)
    n2 = length(norm_b)
    A = all_dense[1:n1, :]
    B = all_dense[n1+1:n1+n2, :]

    vecs_a = [Vector{Float64}(A[i, :]) for i in 1:size(A, 1)]
    vecs_b = [Vector{Float64}(B[i, :]) for i in 1:size(B, 1)]

    return energy_based_histogram_distinguishability(
        vecs_a,
        vecs_b;
        covariance_cutoff_rel_median = covariance_cutoff_rel_median,
        distance = distance,
        verbose = verbose,
    )
end

"""
    total_histogram_distinguishability(hists...)

Compute total distinguishability after sample-wise concatenation of multiple
observables shaped as `[class_a_samples, class_b_samples]`.

Each observable may contain histogram dictionaries or vectors; alignment and
concatenation are handled by `concatenate_hists`.

# Arguments
- `hists`: One or more observables, each shaped as `[class_a_samples, class_b_samples]`.

# Returns
- `result`: Named tuple `(D = value)`.

# Throws
- `ArgumentError`: Propagated for empty input or invalid observable/sample types.
- `DimensionMismatch`: Propagated when class sample counts do not align across observables.
- `DomainError`: Propagated for downstream numeric-domain violations.
"""
function total_histogram_distinguishability(
    hists...;
    covariance_cutoff_rel_median::Float64 = 1e-6,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    vecs_a, vecs_b = concatenate_hists(hists...)
    return energy_based_histogram_distinguishability(
        vecs_a,
        vecs_b;
        covariance_cutoff_rel_median = covariance_cutoff_rel_median,
        distance = distance,
        verbose = verbose,
    )
end

"""
    energy_based_histogram_distinguishability(hists_a, hists_b, num_draws; rng=...)

See `energy_based_histogram_distinguishability(hists_a, hists_b)`.

This overload uses Monte Carlo estimation with `num_draws` and returns
`(D, std)` instead of only `(D,)`.

# Arguments
- `hists_a`: Histogram input data.
- `hists_b`: Histogram input data.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Named tuple containing distinguishability `D`, raw energy distance `E`,
  and, for Monte Carlo overloads, `std`.

# Throws
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function energy_based_histogram_distinguishability(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
    num_draws::Int;
    rng = Random.default_rng(),
    covariance_cutoff_rel_median::Float64 = 1e-6,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    if !(!isempty(hists_a) && !isempty(hists_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    norm_a = normalize_hists([hists_a]; normalization = :probability)[1]
    norm_b = normalize_hists([hists_b]; normalization = :probability)[1]

    all_dense = densify_hists(vcat(norm_a, norm_b))
    n1 = length(norm_a)
    n2 = length(norm_b)
    A = all_dense[1:n1, :]
    B = all_dense[n1+1:n1+n2, :]

    vecs_a = [Vector{Float64}(A[i, :]) for i in 1:size(A, 1)]
    vecs_b = [Vector{Float64}(B[i, :]) for i in 1:size(B, 1)]

    return energy_based_histogram_distinguishability(
        vecs_a,
        vecs_b,
        num_draws;
        rng = rng,
        covariance_cutoff_rel_median = covariance_cutoff_rel_median,
        distance = distance,
        verbose = verbose,
    )
end

"""
    energy_based_histogram_distinguishability(vecs_a, vecs_b)

See `energy_based_histogram_distinguishability(hists_a, hists_b)`.

This is the exact vector-input implementation (no dict normalization stage).

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.

# Returns
- `result`: Named tuple containing distinguishability `D` and, for Monte Carlo overloads, `std`.

# Throws
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function energy_based_histogram_distinguishability(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
    ;
    covariance_cutoff_rel_median::Float64 = 1e-6,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    A, B = _project_vectors_pooled_covariance_for_energy(
        vecs_a,
        vecs_b;
        covariance_cutoff_rel_median = covariance_cutoff_rel_median,
        verbose = verbose,
    )
    dist = _normalize_energy_distance(distance)
    n1 = length(A)
    n2 = length(B)
    exy = 0.0
    exx = 0.0
    eyy = 0.0
    if dist == :hellinger
        sqA, normA = _sqrt_vectors_and_norms(A)
        sqB, normB = _sqrt_vectors_and_norms(B)
        exy = _pairwise_hellinger_sum_sqrt(sqA, normA, sqB, normB) / (n1 * n2)
        exx = _pairwise_hellinger_sum_sqrt(sqA, normA, sqA, normA) / (n1 * n1)
        eyy = _pairwise_hellinger_sum_sqrt(sqB, normB, sqB, normB) / (n2 * n2)
    else
        exy = _pairwise_total_variation_sum(A, B) / (n1 * n2)
        exx = _pairwise_total_variation_sum(A, A) / (n1 * n1)
        eyy = _pairwise_total_variation_sum(B, B) / (n2 * n2)
    end

    E = 2 * exy - exx - eyy
    denom = 2 * exy
    D = denom == 0 ? 0.0 : E / denom
    return (D = D, E = E)
end

"""
    energy_based_histogram_distinguishability(vecs_a, vecs_b, num_draws; rng=...)

See `energy_based_histogram_distinguishability(vecs_a, vecs_b)`.

Monte Carlo variant using `num_draws`; returns `(D, E, std)`.

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Named tuple containing distinguishability `D`, raw energy distance `E`,
  and, for Monte Carlo overloads, `std`.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function energy_based_histogram_distinguishability(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
    num_draws::Int;
    rng = Random.default_rng(),
    covariance_cutoff_rel_median::Float64 = 1e-6,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    A, B = _project_vectors_pooled_covariance_for_energy(
        vecs_a,
        vecs_b;
        covariance_cutoff_rel_median = covariance_cutoff_rel_median,
        verbose = verbose,
    )
    n1 = length(A)
    n2 = length(B)
    Dmat = _distance_matrix_exact(vcat(A, B); distance = distance)

    if !(num_draws > 0)
        throw(DomainError(num_draws, "num_draws must be positive"))
    end
    m = num_draws

    nt = Threads.maxthreadid()
    sum_xy_t = zeros(Float64, nt)
    sum_xx_t = zeros(Float64, nt)
    sum_yy_t = zeros(Float64, nt)
    sumsq_xy_t = zeros(Float64, nt)
    sumsq_xx_t = zeros(Float64, nt)
    sumsq_yy_t = zeros(Float64, nt)
    seeds = _thread_seeds(rng, nt)
    rngs = [Random.Xoshiro(seed) for seed in seeds]

    Threads.@threads for t in 1:m
        tid = Threads.threadid()
        rng_t = rngs[tid]
        i = rand(rng_t, 1:n1); j = rand(rng_t, 1:n2)
        vxy = Dmat[i, n1 + j]
        i1 = rand(rng_t, 1:n1); i2 = rand(rng_t, 1:n1)
        vxx = Dmat[i1, i2]
        j1 = rand(rng_t, 1:n2); j2 = rand(rng_t, 1:n2)
        vyy = Dmat[n1 + j1, n1 + j2]

        sum_xy_t[tid] += vxy
        sum_xx_t[tid] += vxx
        sum_yy_t[tid] += vyy
        sumsq_xy_t[tid] += vxy * vxy
        sumsq_xx_t[tid] += vxx * vxx
        sumsq_yy_t[tid] += vyy * vyy
    end

    sum_xy = sum(sum_xy_t)
    sum_xx = sum(sum_xx_t)
    sum_yy = sum(sum_yy_t)
    sumsq_xy = sum(sumsq_xy_t)
    sumsq_xx = sum(sumsq_xx_t)
    sumsq_yy = sum(sumsq_yy_t)

    exy = sum_xy / m
    exx = sum_xx / m
    eyy = sum_yy / m

    var_exy = _sample_var_from_sums(sum_xy, sumsq_xy, m) / m
    var_exx = _sample_var_from_sums(sum_xx, sumsq_xx, m) / m
    var_eyy = _sample_var_from_sums(sum_yy, sumsq_yy, m) / m

    E = 2 * exy - exx - eyy
    denom = 2 * exy
    if denom == 0
        return (D = 0.0, E = E, std = 0.0)
    end

    var_E = 4 * var_exy + var_exx + var_eyy
    var_denom = 4 * var_exy
    cov_Edenom = 4 * var_exy
    var_D = (var_E / denom^2) + (E^2 * var_denom / denom^4) - (2 * E * cov_Edenom / denom^3)
    std_D = sqrt(max(var_D, 0.0))
    return (D = E / denom, E = E, std = std_D)
end

@inline function _normalize_probability_vector(v::AbstractVector{<:Real}, out::Vector{Float64})::Nothing
    s = 0.0
    @inbounds for i in eachindex(v)
        x = Float64(v[i])
        if x < -1e-12
            throw(DomainError(x, "histogram/bin value must be nonnegative"))
        end
        xi = max(x, 0.0)
        out[i] = xi
        s += xi
    end
    if !(s > 0.0)
        throw(DomainError(s, "histogram/vector sum must be positive for probability normalization"))
    end
    invs = inv(s)
    @inbounds for i in eachindex(v)
        out[i] *= invs
    end
    return nothing
end

function _subsample_class_indices(n::Int, max_per_class::Union{Nothing,Int}, rng)
    if max_per_class === nothing || n <= max_per_class
        return collect(1:n)
    end
    return sort!(Random.randperm(rng, n)[1:max_per_class])
end

function _prepare_mi_embedding_from_projected(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}};
    rng = Random.default_rng(),
    max_per_class::Union{Nothing,Int} = nothing,
)
    idx_a = _subsample_class_indices(length(vecs_a), max_per_class, rng)
    idx_b = _subsample_class_indices(length(vecs_b), max_per_class, rng)
    sel_a = vecs_a[idx_a]
    sel_b = vecs_b[idx_b]

    d = length(sel_a[1])
    Xa = Matrix{Float64}(undef, length(sel_a), d)
    Xb = Matrix{Float64}(undef, length(sel_b), d)
    @inbounds for i in 1:size(Xa, 1), j in 1:d
        Xa[i, j] = Float64(sel_a[i][j])
    end
    @inbounds for i in 1:size(Xb, 1), j in 1:d
        Xb[i, j] = Float64(sel_b[i][j])
    end
    return Xa, Xb
end

function _pca_project(
    X::Matrix{Float64};
    pca_mode::Symbol = :cutoff,
    pca_dim::Int = 32,
    explained_variance::Real = 0.99,
    eigenvalue_rtol::Real = 1e-6,
    verbose::Bool = false,
)
    _validate_mi_pca_parameters(pca_mode, pca_dim, explained_variance, eigenvalue_rtol)
    μ = vec(Statistics.mean(X; dims = 1))
    Xc = X .- transpose(μ)
    n, d = size(Xc)
    if n <= 2 || d == 0
        verbose && _log_projection_info("mutual_information_additional", 0, d, eigenvalue_rtol)
        return Xc
    end
    F = LinearAlgebra.svd(Xc; full = false)
    s2 = F.S .^ 2
    if isempty(s2)
        verbose && _log_projection_info("mutual_information_additional", 0, d, eigenvalue_rtol)
        return Xc
    end
    λmed = Statistics.median(s2)
    keep = findall(λ -> λ > λmed * Float64(eigenvalue_rtol), s2)
    r_cut = isempty(keep) ? 1 : last(keep)
    if pca_mode == :cutoff
        r = r_cut
    elseif pca_mode == :dim
        r = min(pca_dim, r_cut)
    else
        total = sum(@view s2[1:r_cut])
        if total <= 0.0
            r = 1
        else
            cfrac = cumsum(@view s2[1:r_cut]) ./ total
            r = searchsortedfirst(cfrac, Float64(explained_variance))
            r = clamp(r, 1, r_cut)
        end
    end
    verbose && _log_projection_info("mutual_information_additional", d - r, d, eigenvalue_rtol)
    return F.U[:, 1:r] * LinearAlgebra.Diagonal(F.S[1:r])
end

@inline function _validate_mi_pca_parameters(
    pca_mode::Symbol,
    pca_dim::Int,
    explained_variance::Real,
    eigenvalue_rtol::Real,
)
    if !(pca_mode in (:dim, :variance, :cutoff))
        throw(DomainError(pca_mode, "pca_mode must be one of :dim, :variance, :cutoff"))
    end
    if !(pca_dim >= 1)
        throw(DomainError(pca_dim, "pca_dim must be >= 1"))
    end
    if !(0 < explained_variance <= 1)
        throw(DomainError(explained_variance, "explained_variance must satisfy 0 < explained_variance <= 1"))
    end
    if !(0 <= eigenvalue_rtol < 1)
        throw(DomainError(eigenvalue_rtol, "eigenvalue_rtol must satisfy 0 <= eigenvalue_rtol < 1"))
    end
    return nothing
end

@inline function _sqeuclidean_row(X::Matrix{Float64}, i::Int, j::Int)::Float64
    s = 0.0
    @inbounds @simd for k in axes(X, 2)
        d = X[i, k] - X[j, k]
        s += d * d
    end
    return s
end

function _mi_knn_binary_projected(X::Matrix{Float64}, labels::Vector{Bool}, k::Int)
    n = size(X, 1)
    n_a = count(labels)
    n_b = n - n_a
    k_eff = min(k, n_a - 1, n_b - 1)
    if !(k_eff >= 1)
        throw(DomainError(k, "k must be <= min(n_a-1, n_b-1) and both classes must have at least 2 samples"))
    end

    m_counts = Vector{Int}(undef, n)
    nt = Threads.maxthreadid()
    dist_bufs = [Vector{Float64}(undef, n) for _ in 1:nt]
    same_bufs = [Vector{Float64}(undef, max(n_a, n_b)) for _ in 1:nt]
    tol = 1e-14
    tie_eps = 1e-12

    Threads.@threads for i in 1:n
        tid = Threads.threadid()
        d_all = dist_bufs[tid]
        d_same = same_bufs[tid]
        yi = labels[i]
        nsame = 0
        @inbounds for j in 1:n
            if i == j
                d_all[j] = Inf
                continue
            end
            dij = _sqeuclidean_row(X, i, j)
            # Deterministically break exact ties (common with repeated histograms).
            if dij == 0.0
                dij = tie_eps * j
            end
            d_all[j] = dij
            if labels[j] == yi
                nsame += 1
                d_same[nsame] = dij
            end
        end

        eps2 = partialsort!(view(d_same, 1:nsame), k_eff) + tol
        m = 0
        @inbounds for j in 1:n
            if d_all[j] <= eps2
                m += 1
            end
        end
        m_counts[i] = m
    end

    p_a = n_a / n
    p_b = n_b / n
    H_y = -(p_a * log(p_a) + p_b * log(p_b))
    if H_y <= 0.0
        return 0.0
    end

    mean_dig_ny = (n_a * SpecialFunctions.digamma(n_a) + n_b * SpecialFunctions.digamma(n_b)) / n
    mean_dig_m = Statistics.mean(SpecialFunctions.digamma.(m_counts))
    I = SpecialFunctions.digamma(n) - mean_dig_ny + SpecialFunctions.digamma(k_eff) - mean_dig_m
    return clamp(I / H_y, 0.0, 1.0)
end

function _mi_knn_binary(
    Xa::Matrix{Float64},
    Xb::Matrix{Float64};
    k::Int = 5,
    pca_mode::Symbol = :cutoff,
    pca_dim::Int = 32,
    explained_variance::Real = 0.99,
    eigenvalue_rtol::Real = 1e-6,
    verbose::Bool = false,
)
    _validate_mi_pca_parameters(pca_mode, pca_dim, explained_variance, eigenvalue_rtol)
    X = vcat(Xa, Xb)
    Xp = if pca_mode == :cutoff
        X
    else
        _pca_project(
            X;
            pca_mode = pca_mode,
            pca_dim = pca_dim,
            explained_variance = explained_variance,
            eigenvalue_rtol = eigenvalue_rtol,
            verbose = verbose,
        )
    end
    labels = vcat(fill(true, size(Xa, 1)), fill(false, size(Xb, 1)))
    return _mi_knn_binary_projected(Xp, labels, k)
end

"""
    distinguishability_mutual_information(hists_a, hists_b)

Compute normalized mutual-information distinguishability `D_mi ∈ [0,1]` between
two histogram/vector sample sets.

For vector inputs, each sample vector is interpreted as a histogram and normalized
to a probability vector internally. For dictionary inputs, histograms are normalized
with `normalize_hists(...; normalization=:probability)` first.

The estimator uses:
1. Hellinger embedding `u = sqrt(p)` of normalized histogram vectors.
2. Pooled-cutoff projection (`pca_mode = :cutoff`, default), with optional extra PCA reduction.
3. Binary-label kNN MI estimation with normalization by class entropy `H(Y)`.

# Arguments
- `hists_a`: Histogram/vector samples for class A.
- `hists_b`: Histogram/vector samples for class B.

# Keyword Arguments
- `k`: Neighbor count for kNN MI estimator.
- `pca_mode`: PCA selection mode: `:cutoff` (default), `:dim`, or `:variance`.
- `pca_dim`: PCA embedding dimension upper bound (used when `pca_mode = :dim`).
- `explained_variance`: Target retained variance fraction (used when `pca_mode = :variance`).
- `eigenvalue_rtol`: Relative eigenvalue cutoff for near-null components.
- `max_per_class`: Optional random class subsampling cap before estimation.
- `rng`: Random number generator used for stochastic subsampling.

# Returns
- `result`: Named tuple `(D_mi = value)`.

# Throws
- `ArgumentError`: Raised for empty inputs.
- `DomainError`: Raised for invalid estimator parameters or histogram/vector normalization.
"""
function distinguishability_mutual_information(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
    ;
    k::Int = 5,
    pca_mode::Symbol = :cutoff,
    pca_dim::Int = 32,
    explained_variance::Real = 0.99,
    eigenvalue_rtol::Real = 1e-6,
    max_per_class::Union{Nothing,Int} = nothing,
    rng = Random.default_rng(),
    verbose::Bool = false,
)
    if !(!isempty(hists_a) && !isempty(hists_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    norm_a = normalize_hists([hists_a]; normalization = :probability)[1]
    norm_b = normalize_hists([hists_b]; normalization = :probability)[1]

    all_dense = densify_hists(vcat(norm_a, norm_b))
    n1 = length(norm_a)
    n2 = length(norm_b)
    A = all_dense[1:n1, :]
    B = all_dense[n1+1:n1+n2, :]

    vecs_a = [Vector{Float64}(A[i, :]) for i in 1:size(A, 1)]
    vecs_b = [Vector{Float64}(B[i, :]) for i in 1:size(B, 1)]
    return distinguishability_mutual_information(
        vecs_a,
        vecs_b;
        k = k,
        pca_mode = pca_mode,
        pca_dim = pca_dim,
        explained_variance = explained_variance,
        eigenvalue_rtol = eigenvalue_rtol,
        max_per_class = max_per_class,
        rng = rng,
        verbose = verbose,
    )
end

"""
    distinguishability_mutual_information(vecs_a, vecs_b)

Vector-input implementation of `distinguishability_mutual_information(hists_a, hists_b)`.

# Arguments
- `vecs_a`: Vector-valued histogram samples for class A.
- `vecs_b`: Vector-valued histogram samples for class B.

# Returns
- `result`: Named tuple `(D_mi = value)`.
"""
function distinguishability_mutual_information(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
    ;
    k::Int = 5,
    pca_mode::Symbol = :cutoff,
    pca_dim::Int = 32,
    explained_variance::Real = 0.99,
    eigenvalue_rtol::Real = 1e-6,
    max_per_class::Union{Nothing,Int} = nothing,
    rng = Random.default_rng(),
    verbose::Bool = false,
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    _validate_mi_pca_parameters(pca_mode, pca_dim, explained_variance, eigenvalue_rtol)
    if max_per_class !== nothing && !(max_per_class >= 2)
        throw(DomainError(max_per_class, "max_per_class must be nothing or >= 2"))
    end
    Aproj, Bproj = _project_vectors_pooled_covariance_for_energy(
        vecs_a,
        vecs_b;
        covariance_cutoff_rel_median = eigenvalue_rtol,
        method = :mutual_information,
        verbose = verbose,
    )
    Xa, Xb = _prepare_mi_embedding_from_projected(Aproj, Bproj; rng = rng, max_per_class = max_per_class)
    return (D_mi = _mi_knn_binary(
        Xa,
        Xb;
        k = k,
        pca_mode = pca_mode,
        pca_dim = pca_dim,
        explained_variance = explained_variance,
        eigenvalue_rtol = eigenvalue_rtol,
        verbose = verbose,
    ),)
end

"""
    distinguishability_mutual_information(hists_a, hists_b, num_draws; rng=...)

Bootstrap-resampled variant of `distinguishability_mutual_information(hists_a, hists_b)`.
Returns `(D_mi, std)` where `D_mi` is mean bootstrap normalized MI and `std`
its sample standard deviation across draws.
"""
function distinguishability_mutual_information(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
    num_draws::Int;
    rng = Random.default_rng(),
    k::Int = 5,
    pca_mode::Symbol = :cutoff,
    pca_dim::Int = 32,
    explained_variance::Real = 0.99,
    eigenvalue_rtol::Real = 1e-6,
    max_per_class::Union{Nothing,Int} = nothing,
    verbose::Bool = false,
)
    if !(!isempty(hists_a) && !isempty(hists_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    norm_a = normalize_hists([hists_a]; normalization = :probability)[1]
    norm_b = normalize_hists([hists_b]; normalization = :probability)[1]

    all_dense = densify_hists(vcat(norm_a, norm_b))
    n1 = length(norm_a)
    n2 = length(norm_b)
    A = all_dense[1:n1, :]
    B = all_dense[n1+1:n1+n2, :]

    vecs_a = [Vector{Float64}(A[i, :]) for i in 1:size(A, 1)]
    vecs_b = [Vector{Float64}(B[i, :]) for i in 1:size(B, 1)]
    return distinguishability_mutual_information(
        vecs_a,
        vecs_b,
        num_draws;
        rng = rng,
        k = k,
        pca_mode = pca_mode,
        pca_dim = pca_dim,
        explained_variance = explained_variance,
        eigenvalue_rtol = eigenvalue_rtol,
        max_per_class = max_per_class,
        verbose = verbose,
    )
end

"""
    distinguishability_mutual_information(vecs_a, vecs_b, num_draws; rng=...)

Bootstrap-resampled vector-input variant of `distinguishability_mutual_information`.
"""
function distinguishability_mutual_information(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
    num_draws::Int;
    rng = Random.default_rng(),
    k::Int = 5,
    pca_mode::Symbol = :cutoff,
    pca_dim::Int = 32,
    explained_variance::Real = 0.99,
    eigenvalue_rtol::Real = 1e-6,
    max_per_class::Union{Nothing,Int} = nothing,
    verbose::Bool = false,
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    _validate_mi_pca_parameters(pca_mode, pca_dim, explained_variance, eigenvalue_rtol)
    if !(num_draws > 0)
        throw(DomainError(num_draws, "num_draws must be positive"))
    end
    if max_per_class !== nothing && !(max_per_class >= 2)
        throw(DomainError(max_per_class, "max_per_class must be nothing or >= 2"))
    end

    Aproj, Bproj = _project_vectors_pooled_covariance_for_energy(
        vecs_a,
        vecs_b;
        covariance_cutoff_rel_median = eigenvalue_rtol,
        method = :mutual_information,
        verbose = verbose,
    )
    Xa, Xb = _prepare_mi_embedding_from_projected(Aproj, Bproj; rng = rng, max_per_class = max_per_class)
    na = size(Xa, 1)
    nb = size(Xb, 1)
    if na < 2 || nb < 2
        throw(DomainError((na, nb), "each class must contain at least 2 samples after preprocessing"))
    end

    X = vcat(Xa, Xb)
    Xp = if pca_mode == :cutoff
        X
    else
        _pca_project(
            X;
            pca_mode = pca_mode,
            pca_dim = pca_dim,
            explained_variance = explained_variance,
            eigenvalue_rtol = eigenvalue_rtol,
            verbose = verbose,
        )
    end
    Xpa = @view Xp[1:na, :]
    Xpb = @view Xp[(na + 1):(na + nb), :]

    Ds = Vector{Float64}(undef, num_draws)
    draw_seeds = rand(rng, UInt64, num_draws)

    Threads.@threads for t in 1:num_draws
        rng_t = Random.Xoshiro(draw_seeds[t])
        idx_a = rand(rng_t, 1:na, na)
        idx_b = rand(rng_t, 1:nb, nb)
        Xd = vcat(Xpa[idx_a, :], Xpb[idx_b, :])
        labels = vcat(fill(true, na), fill(false, nb))
        Ds[t] = _mi_knn_binary_projected(Xd, labels, k)
    end

    return (D_mi = Statistics.mean(Ds), std = Statistics.std(Ds))
end

"""
    total_histogram_mutual_information_distinguishability(hists...; num_draws=nothing, rng=...)

Compute normalized mutual-information distinguishability after sample-wise
concatenation of one or more observables shaped as `[class_a_samples, class_b_samples]`.

If `num_draws === nothing`, returns exact empirical `D_mi`. If `num_draws` is
provided, returns bootstrap mean/std as `(D_mi, std)`.
"""
function total_histogram_mutual_information_distinguishability(
    hists...;
    num_draws::Union{Nothing,Int} = nothing,
    rng = Random.default_rng(),
    k::Int = 5,
    pca_mode::Symbol = :cutoff,
    pca_dim::Int = 32,
    explained_variance::Real = 0.99,
    eigenvalue_rtol::Real = 1e-6,
    max_per_class::Union{Nothing,Int} = nothing,
    verbose::Bool = false,
)
    vecs_a, vecs_b = concatenate_hists(hists...)
    if num_draws === nothing
        return distinguishability_mutual_information(
            vecs_a,
            vecs_b;
            k = k,
            pca_mode = pca_mode,
            pca_dim = pca_dim,
            explained_variance = explained_variance,
            eigenvalue_rtol = eigenvalue_rtol,
            max_per_class = max_per_class,
            rng = rng,
            verbose = verbose,
        )
    end
    return distinguishability_mutual_information(
        vecs_a,
        vecs_b,
        num_draws;
        rng = rng,
        k = k,
        pca_mode = pca_mode,
        pca_dim = pca_dim,
        explained_variance = explained_variance,
        eigenvalue_rtol = eigenvalue_rtol,
        max_per_class = max_per_class,
        verbose = verbose,
    )
end

@inline function _tv_state_key(v::AbstractVector{Float64}, quant_scale::Float64)::String
    io = IOBuffer()
    @inbounds for i in eachindex(v)
        i > 1 && write(io, UInt8(','))
        q = round(Int64, max(v[i], 0.0) * quant_scale)
        print(io, q)
    end
    return String(take!(io))
end

function _tv_counts(
    vecs::Vector{<:AbstractVector{Float64}},
    quant_scale::Float64,
)::Dict{String,Int}
    counts = Dict{String,Int}()
    @inbounds for v in vecs
        key = _tv_state_key(v, quant_scale)
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

function _tv_from_counts(
    counts_a::Dict{String,Int},
    n_a::Int,
    counts_b::Dict{String,Int},
    n_b::Int,
)::Float64
    keys_union = union(keys(counts_a), keys(counts_b))
    s = 0.0
    @inbounds for k in keys_union
        p_a = get(counts_a, k, 0) / n_a
        p_b = get(counts_b, k, 0) / n_b
        s += abs(p_a - p_b)
    end
    return 0.5 * s
end

function _normalize_hist_vectors_for_tv(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
)::Tuple{Vector{Vector{Float64}},Vector{Vector{Float64}}}
    A, B = _prepare_vectors_for_distance(vecs_a, vecs_b)
    norm_a = Vector{Vector{Float64}}(undef, length(A))
    norm_b = Vector{Vector{Float64}}(undef, length(B))
    @inbounds for i in eachindex(A)
        out = Vector{Float64}(undef, length(A[i]))
        _normalize_probability_vector(A[i], out)
        norm_a[i] = out
    end
    @inbounds for i in eachindex(B)
        out = Vector{Float64}(undef, length(B[i]))
        _normalize_probability_vector(B[i], out)
        norm_b[i] = out
    end
    return norm_a, norm_b
end

function _tv_core(
    vecs_a::Vector{<:AbstractVector{Float64}},
    vecs_b::Vector{<:AbstractVector{Float64}},
    tv_quantization_digits::Int,
)::Float64
    quant_scale = 10.0^tv_quantization_digits
    counts_a = _tv_counts(vecs_a, quant_scale)
    counts_b = _tv_counts(vecs_b, quant_scale)
    return _tv_from_counts(counts_a, length(vecs_a), counts_b, length(vecs_b))
end

function _tv_bias_split_half(
    vecs::Vector{<:AbstractVector{Float64}},
    tv_quantization_digits::Int,
    bias_num_splits::Int,
    rng,
)::Vector{Float64}
    n = length(vecs)
    n2 = n ÷ 2
    if n2 < 1
        throw(DomainError(n, "need at least 2 samples in each class for TV bias check"))
    end
    out = Vector{Float64}(undef, bias_num_splits)
    @inbounds for r in 1:bias_num_splits
        perm = Random.randperm(rng, n)
        v1 = vecs[perm[1:n2]]
        v2 = vecs[perm[(n2 + 1):(2 * n2)]]
        out[r] = _tv_core(v1, v2, tv_quantization_digits)
    end
    return out
end

"""
    distinguishability_total_variation(vals_a, vals_b; tv_quantization_digits=8, check_bias=false, bias_num_splits=20, rng=..., verbose=false)

Compute empirical total-variation distinguishability between two sets of
discrete histogram samples and report corresponding Bayes accuracy estimate
for equal priors: `bayes_accuracy = (1 + D_tv) / 2`.

Inputs can be histogram dictionaries or vectors. Histogram vectors are first
probability-normalized and aligned to shared support. TV is then computed on
empirical distributions over full histogram states after quantization.

If `check_bias=true`, split-half within-class TV baselines are estimated for A
and B and summarized in the output.

# Returns
- Named tuple with `D_tv`, `bayes_accuracy`, and bias diagnostics.
"""
function distinguishability_total_variation(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict};
    tv_quantization_digits::Int = 8,
    check_bias::Bool = false,
    bias_num_splits::Int = 20,
    rng = Random.default_rng(),
    verbose::Bool = false,
)
    isempty(hists_a) && throw(ArgumentError("hists_a must be non-empty"))
    isempty(hists_b) && throw(ArgumentError("hists_b must be non-empty"))
    norm_a = normalize_hists([hists_a]; normalization = :probability)[1]
    norm_b = normalize_hists([hists_b]; normalization = :probability)[1]
    all_dense = densify_hists(vcat(norm_a, norm_b))
    n1 = length(norm_a)
    n2 = length(norm_b)
    vecs_a = [Vector{Float64}(all_dense[i, :]) for i in 1:n1]
    vecs_b = [Vector{Float64}(all_dense[n1 + i, :]) for i in 1:n2]
    return distinguishability_total_variation(
        vecs_a,
        vecs_b;
        tv_quantization_digits = tv_quantization_digits,
        check_bias = check_bias,
        bias_num_splits = bias_num_splits,
        rng = rng,
        verbose = verbose,
    )
end

function distinguishability_total_variation(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}};
    tv_quantization_digits::Int = 8,
    check_bias::Bool = false,
    bias_num_splits::Int = 20,
    rng = Random.default_rng(),
    verbose::Bool = false,
)
    isempty(vecs_a) && throw(ArgumentError("vecs_a must be non-empty"))
    isempty(vecs_b) && throw(ArgumentError("vecs_b must be non-empty"))
    if !(0 <= tv_quantization_digits <= 12)
        throw(DomainError(tv_quantization_digits, "tv_quantization_digits must be in [0, 12]"))
    end
    if !(bias_num_splits > 0)
        throw(DomainError(bias_num_splits, "bias_num_splits must be positive"))
    end

    norm_a, norm_b = _normalize_hist_vectors_for_tv(vecs_a, vecs_b)
    D_tv = _tv_core(norm_a, norm_b, tv_quantization_digits)
    bayes_accuracy = 0.5 * (1 + D_tv)

    if !check_bias
        verbose && @info "TV distinguishability: D_tv=$(D_tv), bayes_accuracy=$(bayes_accuracy), bias_check=false"
        return (
            D_tv = D_tv,
            bayes_accuracy = bayes_accuracy,
            tv_bias_mean = nothing,
            tv_bias_std = nothing,
            D_tv_debiased = nothing,
            bayes_accuracy_debiased = nothing,
        )
    end

    bias_a = _tv_bias_split_half(norm_a, tv_quantization_digits, bias_num_splits, rng)
    bias_b = _tv_bias_split_half(norm_b, tv_quantization_digits, bias_num_splits, rng)
    bias_all = vcat(bias_a, bias_b)
    tv_bias_mean = Statistics.mean(bias_all)
    tv_bias_std = Statistics.std(bias_all)
    D_tv_debiased = clamp(D_tv - tv_bias_mean, 0.0, 1.0)
    bayes_accuracy_debiased = 0.5 * (1 + D_tv_debiased)
    if verbose
        @info "TV distinguishability: D_tv=$(D_tv), bayes_accuracy=$(bayes_accuracy), tv_bias_mean=$(tv_bias_mean), tv_bias_std=$(tv_bias_std), D_tv_debiased=$(D_tv_debiased), bayes_accuracy_debiased=$(bayes_accuracy_debiased)"
    end
    return (
        D_tv = D_tv,
        bayes_accuracy = bayes_accuracy,
        tv_bias_mean = tv_bias_mean,
        tv_bias_std = tv_bias_std,
        D_tv_debiased = D_tv_debiased,
        bayes_accuracy_debiased = bayes_accuracy_debiased,
    )
end

function total_histogram_total_variation_distinguishability(
    hists...;
    tv_quantization_digits::Int = 8,
    check_bias::Bool = false,
    bias_num_splits::Int = 20,
    rng = Random.default_rng(),
    verbose::Bool = false,
)
    vecs_a, vecs_b = concatenate_hists(hists...)
    return distinguishability_total_variation(
        vecs_a,
        vecs_b;
        tv_quantization_digits = tv_quantization_digits,
        check_bias = check_bias,
        bias_num_splits = bias_num_splits,
        rng = rng,
        verbose = verbose,
    )
end

"""
    histogram_distinguishability_permutation(hists_a, hists_b; n_perm=1000, rng=...)

Permutation test for distinguishability. Returns a named tuple
`(D_obs, p_value, z_emp, z_coll, std_Ts)`, where `D_obs` is the observed
distinguishability and `z_coll` is the collider-style Z derived from `p_value`.

# Arguments
- `hists_a`: Histogram input data.
- `hists_b`: Histogram input data.

# Keyword Arguments
- `n_perm`: Number of permutations.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Named tuple containing `D_obs`, `p_value`, `z_emp`, `z_coll`, and `std_Ts`.

# Throws
- `DomainError`: Propagated for invalid `num_draws`.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function histogram_distinguishability_permutation(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict};
    n_perm::Int = 1000,
    rng = Random.default_rng(),
    progress::Bool = false,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    if !(!isempty(hists_a) && !isempty(hists_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    norm_a = normalize_hists([hists_a]; normalization = :probability)[1]
    norm_b = normalize_hists([hists_b]; normalization = :probability)[1]

    all_dense = densify_hists(vcat(norm_a, norm_b))
    n1 = length(norm_a)
    n2 = length(norm_b)
    A = all_dense[1:n1, :]
    B = all_dense[n1+1:n1+n2, :]

    vecs_a = [Vector{Float64}(A[i, :]) for i in 1:size(A, 1)]
    vecs_b = [Vector{Float64}(B[i, :]) for i in 1:size(B, 1)]

    return histogram_distinguishability_permutation(vecs_a, vecs_b; n_perm = n_perm, rng = rng, progress = progress, distance = distance, verbose = verbose)
end

"""
    histogram_distinguishability_permutation(hists_a, hists_b, num_draws; n_perm=1000, rng=...)

See `histogram_distinguishability_permutation(hists_a, hists_b; ...)`.

Monte Carlo variant that samples `num_draws` pooled pairs per permutation run;
returns the same fields.

# Arguments
- `hists_a`: Histogram input data.
- `hists_b`: Histogram input data.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `n_perm`: Number of permutations.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Named tuple containing `D_obs`, `p_value`, `z_emp`, `z_coll`, and `std_Ts`.

# Throws
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function histogram_distinguishability_permutation(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
    num_draws::Int;
    n_perm::Int = 1000,
    rng = Random.default_rng(),
    progress::Bool = false,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    if !(!isempty(hists_a) && !isempty(hists_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    norm_a = normalize_hists([hists_a]; normalization = :probability)[1]
    norm_b = normalize_hists([hists_b]; normalization = :probability)[1]

    all_dense = densify_hists(vcat(norm_a, norm_b))
    n1 = length(norm_a)
    n2 = length(norm_b)
    A = all_dense[1:n1, :]
    B = all_dense[n1+1:n1+n2, :]

    vecs_a = [Vector{Float64}(A[i, :]) for i in 1:size(A, 1)]
    vecs_b = [Vector{Float64}(B[i, :]) for i in 1:size(B, 1)]

    return histogram_distinguishability_permutation(vecs_a, vecs_b, num_draws; n_perm = n_perm, rng = rng, progress = progress, distance = distance, verbose = verbose)
end

"""
    histogram_distinguishability_permutation(vecs_a, vecs_b; n_perm=1000, rng=...)

See `histogram_distinguishability_permutation(hists_a, hists_b; ...)`.

This is the vector-input implementation (no dict normalization stage).

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.

# Keyword Arguments
- `n_perm`: Number of permutations.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Named tuple containing `D_obs`, `p_value`, `z_emp`, `z_coll`, and `std_Ts`.

# Throws
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function histogram_distinguishability_permutation(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}};
    n_perm::Int = 1000,
    rng = Random.default_rng(),
    progress::Bool = false,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    n1 = length(vecs_a)
    n2 = length(vecs_b)
    n = min(n1, n2)
    if n < 2
        return (D_obs = 0.0, p_value = 1.0, z_emp = 0.0, z_coll = 0.0, std_Ts = 0.0)
    end

    idx_a = n1 == n ? collect(1:n1) : sort!(Random.randperm(rng, n1)[1:n])
    idx_b = n2 == n ? collect(1:n2) : sort!(Random.randperm(rng, n2)[1:n])

    A = vecs_a[idx_a]
    B = vecs_b[idx_b]

    maxlen = maximum(length.(vcat(A, B)))
    pad_to(v, n) = length(v) == n ? collect(v) : vcat(collect(v), zeros(Float64, n - length(v)))
    Ap = [pad_to(v, maxlen) for v in A]
    Bp = [pad_to(v, maxlen) for v in B]

    max_nonzero = 0
    for v in Ap
        idx = findlast(>(0), v)
        if idx !== nothing && idx > max_nonzero
            max_nonzero = idx
        end
    end
    for v in Bp
        idx = findlast(>(0), v)
        if idx !== nothing && idx > max_nonzero
            max_nonzero = idx
        end
    end
    max_nonzero == 0 && (max_nonzero = maxlen)
    Ap = [v[1:max_nonzero] for v in Ap]
    Bp = [v[1:max_nonzero] for v in Bp]

    C = vcat(Ap, Bp)
    n_total = length(C)
    n_per = n

    Dmat = _distance_matrix_exact(C; distance = distance)
    pairs_u = Vector{Int}()
    pairs_v = Vector{Int}()
    dists = Float64[]
    for i in 1:(n_total - 1), j in (i + 1):n_total
        push!(pairs_u, i)
        push!(pairs_v, j)
        push!(dists, Dmat[i, j])
    end

    labels_obs = vcat(fill(true, n_per), fill(false, n_per))

    function compute_D(labels)
        sum_aa = 0.0
        sum_bb = 0.0
        sum_ab = 0.0
        cnt_aa = 0
        cnt_bb = 0
        cnt_ab = 0
        @inbounds for k in eachindex(dists)
            a = labels[pairs_u[k]]
            b = labels[pairs_v[k]]
            if a && b
                sum_aa += dists[k]; cnt_aa += 1
            elseif (!a) && (!b)
                sum_bb += dists[k]; cnt_bb += 1
            else
                sum_ab += dists[k]; cnt_ab += 1
            end
        end
        m_aa = cnt_aa == 0 ? 0.0 : sum_aa / cnt_aa
        m_bb = cnt_bb == 0 ? 0.0 : sum_bb / cnt_bb
        m_ab = cnt_ab == 0 ? 0.0 : sum_ab / cnt_ab
        E = 2 * m_ab - m_aa - m_bb
        denom = 2 * m_ab
        return denom == 0 ? 0.0 : E / denom
    end

    D_obs = compute_D(labels_obs)

    Ds = Vector{Float64}(undef, n_perm)
    pm = progress ? ProgressMeter.Progress(n_perm; desc = "permute D") : nothing
    for p in 1:n_perm
        perm = Random.randperm(rng, n_total)
        labels = falses(n_total)
        @inbounds for i in 1:n_per
            labels[perm[i]] = true
        end
        Ds[p] = compute_D(labels)
        progress && ProgressMeter.next!(pm)
    end

    p_value = (count(>=(D_obs), Ds) + 1) / (n_perm + 1)
    std_Ts = Statistics.std(Ds)
    z_emp = (D_obs - Statistics.mean(Ds)) / (std_Ts + eps())
    p_clamped = clamp(p_value, eps(), 1 - eps())
    z_coll = Distributions.quantile(Distributions.Normal(), 1 - p_clamped)
    return (D_obs = D_obs, p_value = p_value, z_emp = z_emp, z_coll = z_coll, std_Ts = std_Ts)
end

"""
    histogram_distinguishability_permutation(vecs_a, vecs_b, num_draws; n_perm=1000, rng=...)

See `histogram_distinguishability_permutation(vecs_a, vecs_b; ...)`.

Monte Carlo variant: samples `num_draws` pooled pairs and reuses them across
permutations.

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `n_perm`: Number of permutations.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Named tuple containing `D_obs`, `p_value`, `z_emp`, `z_coll`, and `std_Ts`.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function histogram_distinguishability_permutation(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
    num_draws::Int;
    n_perm::Int = 1000,
    rng = Random.default_rng(),
    progress::Bool = false,
    distance::Symbol = :Hellinger,
    verbose::Bool = false,
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    n1 = length(vecs_a)
    n2 = length(vecs_b)
    n = min(n1, n2)
    if n < 2
        return (D_obs = 0.0, p_value = 1.0, z_emp = 0.0, z_coll = 0.0, std_Ts = 0.0)
    end

    idx_a = n1 == n ? collect(1:n1) : sort!(Random.randperm(rng, n1)[1:n])
    idx_b = n2 == n ? collect(1:n2) : sort!(Random.randperm(rng, n2)[1:n])

    Ap, Bp = _prepare_vectors_for_distance(vecs_a[idx_a], vecs_b[idx_b])
    C = vcat(Ap, Bp)
    n_total = length(C)
    n_per = n
    Dmat = _distance_matrix_exact(C; distance = distance)

    if !(num_draws > 0)
        throw(DomainError(num_draws, "num_draws must be positive"))
    end
    pairs_u = Vector{Int}(undef, num_draws)
    pairs_v = Vector{Int}(undef, num_draws)
    dists = Vector{Float64}(undef, num_draws)
    _sample_pooled_pair_distances!(pairs_u, pairs_v, dists, Dmat, rng)

    labels_obs = vcat(fill(true, n_per), fill(false, n_per))

    function compute_D(labels)
        sum_aa = 0.0
        sum_bb = 0.0
        sum_ab = 0.0
        cnt_aa = 0
        cnt_bb = 0
        cnt_ab = 0
        @inbounds for k in eachindex(dists)
            a = labels[pairs_u[k]]
            b = labels[pairs_v[k]]
            if a && b
                sum_aa += dists[k]; cnt_aa += 1
            elseif (!a) && (!b)
                sum_bb += dists[k]; cnt_bb += 1
            else
                sum_ab += dists[k]; cnt_ab += 1
            end
        end
        m_aa = cnt_aa == 0 ? 0.0 : sum_aa / cnt_aa
        m_bb = cnt_bb == 0 ? 0.0 : sum_bb / cnt_bb
        m_ab = cnt_ab == 0 ? 0.0 : sum_ab / cnt_ab
        E = 2 * m_ab - m_aa - m_bb
        denom = 2 * m_ab
        return denom == 0 ? 0.0 : E / denom
    end

    D_obs = compute_D(labels_obs)

    Ds = Vector{Float64}(undef, n_perm)
    pm = progress ? ProgressMeter.Progress(n_perm; desc = "permute D") : nothing
    for p in 1:n_perm
        perm = Random.randperm(rng, n_total)
        labels = falses(n_total)
        @inbounds for i in 1:n_per
            labels[perm[i]] = true
        end
        Ds[p] = compute_D(labels)
        progress && ProgressMeter.next!(pm)
    end

    p_value = (count(>=(D_obs), Ds) + 1) / (n_perm + 1)
    std_Ts = Statistics.std(Ds)
    z_emp = (D_obs - Statistics.mean(Ds)) / (std_Ts + eps())
    p_clamped = clamp(p_value, eps(), 1 - eps())
    z_coll = Distributions.quantile(Distributions.Normal(), 1 - p_clamped)
    return (D_obs = D_obs, p_value = p_value, z_emp = z_emp, z_coll = z_coll, std_Ts = std_Ts)
end

"""
    mahalanobis_gap_distinguishability(A, B; regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, projection_tolerance=1e-6, to_regularize_rel=0.01, progress=false)

See `mahalanobis_gap_distinguishability(vecs_a, vecs_b; ...)`.

This overload accepts histogram dictionaries, normalizes/densifies them, and
delegates to the vector implementation.

# Arguments
- `A`: First histogram sample set.
- `B`: Second histogram sample set.

# Keyword Arguments
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `R`: Number of baseline resampling runs.
- `q`: Quantile parameter in [0, 1].
- `alpha`: Significance level used for thresholding (1 - alpha).
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: If true, also evaluate the reverse direction.
- `projection_tolerance`: Relative pooled/full eigenvalue cutoff for near-null projection/flooring.
- `to_regularize_rel`: Relative eigenvalue cutoff for split-based small-mode variance regularization.
- `progress`: If true, show progress for resampling stages where applicable.

# Returns
- `result`: Named tuple with Mahalanobis-gap statistic, threshold comparison, and optional symmetric outputs.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function mahalanobis_gap_distinguishability(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict};
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    projection_tolerance::Float64 = 1e-6,
    to_regularize_rel::Float64 = 0.01,
    progress::Bool = false,
    verbose::Bool = false,
)
    isempty(hists_a) && throw(ArgumentError("hists_a must be non-empty"))
    isempty(hists_b) && throw(ArgumentError("hists_b must be non-empty"))
    norm_a = normalize_hists([hists_a]; normalization = :probability)[1]
    norm_b = normalize_hists([hists_b]; normalization = :probability)[1]

    all_dense = densify_hists(vcat(norm_a, norm_b))
    n1 = length(norm_a)
    n2 = length(norm_b)
    A = all_dense[1:n1, :]
    B = all_dense[n1+1:n1+n2, :]

    vecs_a = [Vector{Float64}(A[i, :]) for i in 1:size(A, 1)]
    vecs_b = [Vector{Float64}(B[i, :]) for i in 1:size(B, 1)]

    return mahalanobis_gap_distinguishability(
        vecs_a,
        vecs_b;
        regulator = regulator,
        R = R,
        q = q,
        alpha = alpha,
        rng = rng,
        symmetric = symmetric,
        projection_tolerance = projection_tolerance,
        to_regularize_rel = to_regularize_rel,
        progress = progress,
        verbose = verbose,
    )
end

"""
    mahalanobis_gap_distinguishability(vals_a, vals_b; regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, projection_tolerance=1e-6, to_regularize_rel=0.01, progress=false)

See `mahalanobis_gap_distinguishability(vecs_a, vecs_b; ...)`.

Dispatch helper that forwards homogeneous vectors-of-dicts or
vectors-of-vectors to the corresponding concrete method.

# Arguments
- `vals_a`: First input sample collection.
- `vals_b`: Second input sample collection.

# Keyword Arguments
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `R`: Number of baseline resampling runs.
- `q`: Quantile parameter in [0, 1].
- `alpha`: Significance level used for thresholding (1 - alpha).
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: If true, also evaluate the reverse direction.
- `projection_tolerance`: Relative pooled/full eigenvalue cutoff for near-null projection/flooring.
- `to_regularize_rel`: Relative eigenvalue cutoff for split-based small-mode variance regularization.
- `progress`: If true, show progress for resampling stages where applicable.

# Returns
- `result`: Named tuple with Mahalanobis-gap statistic, threshold comparison, and optional symmetric outputs.

# Throws
- `TypeError`: Raised when vector-valued inputs contain non-`Real` elements.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function mahalanobis_gap_distinguishability(
    vals_a::AbstractVector,
    vals_b::AbstractVector;
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    projection_tolerance::Float64 = 1e-6,
    to_regularize_rel::Float64 = 0.01,
    progress::Bool = false,
    verbose::Bool = false,
)
    isempty(vals_a) && throw(ArgumentError("vals_a must be non-empty"))
    isempty(vals_b) && throw(ArgumentError("vals_b must be non-empty"))
    if all(v -> v isa AbstractDict, vals_a) && all(v -> v isa AbstractDict, vals_b)
        hists_a = AbstractDict[v for v in vals_a]
        hists_b = AbstractDict[v for v in vals_b]
        return mahalanobis_gap_distinguishability(
            hists_a,
            hists_b;
            regulator = regulator,
            R = R,
            q = q,
            alpha = alpha,
            rng = rng,
            symmetric = symmetric,
            projection_tolerance = projection_tolerance,
            to_regularize_rel = to_regularize_rel,
            progress = progress,
            verbose = verbose,
        )
    elseif all(v -> v isa AbstractVector, vals_a) && all(v -> v isa AbstractVector, vals_b)
        bad_a = findfirst(v -> any(x -> !(x isa Real), v), vals_a)
        if bad_a !== nothing
            bad_val = first(filter(x -> !(x isa Real), vals_a[bad_a]))
            throw(TypeError(:mahalanobis_gap_distinguishability, "vector element", Real, bad_val))
        end
        bad_b = findfirst(v -> any(x -> !(x isa Real), v), vals_b)
        if bad_b !== nothing
            bad_val = first(filter(x -> !(x isa Real), vals_b[bad_b]))
            throw(TypeError(:mahalanobis_gap_distinguishability, "vector element", Real, bad_val))
        end
        vecs_a = [Vector{Float64}(v) for v in vals_a]
        vecs_b = [Vector{Float64}(v) for v in vals_b]
        return mahalanobis_gap_distinguishability(
            vecs_a,
            vecs_b;
            regulator = regulator,
            R = R,
            q = q,
            alpha = alpha,
            rng = rng,
            symmetric = symmetric,
            projection_tolerance = projection_tolerance,
            to_regularize_rel = to_regularize_rel,
            progress = progress,
            verbose = verbose,
        )
    else
        throw(ArgumentError("mahalanobis_gap_distinguishability expects vectors of dicts or vectors of numeric vectors"))
    end
end

"""
    _prepare_vectors_for_mahalanobis(vecs_a, vecs_b)

Internal preprocessing for Mahalanobis-gap calculations.

Pads every input vector to the same length, then trims both datasets to the
largest non-zero coordinate present in either set. This ensures aligned feature
dimensions while removing irrelevant trailing zeros.

# Returns
Tuple `(A, B)` where both entries are `Vector{Vector{Float64}}`.

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.

# Throws
- `ArgumentError`: If either input collection is empty.
"""
function _prepare_vectors_for_mahalanobis(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
)
    isempty(vecs_a) && throw(ArgumentError("vecs_a must be non-empty"))
    isempty(vecs_b) && throw(ArgumentError("vecs_b must be non-empty"))
    maxlen = maximum(length.(vcat(vecs_a, vecs_b)))
    pad_to(v, n) = length(v) == n ? collect(v) : vcat(collect(v), zeros(Float64, n - length(v)))
    A = [pad_to(v, maxlen) for v in vecs_a]
    B = [pad_to(v, maxlen) for v in vecs_b]

    max_nonzero = 0
    for v in A
        idx = findlast(>(0), v)
        if idx !== nothing && idx > max_nonzero
            max_nonzero = idx
        end
    end
    for v in B
        idx = findlast(>(0), v)
        if idx !== nothing && idx > max_nonzero
            max_nonzero = idx
        end
    end
    max_nonzero == 0 && (max_nonzero = maxlen)
    A = [v[1:max_nonzero] for v in A]
    B = [v[1:max_nonzero] for v in B]

    return A, B
end

"""
    _fit_reference(B, regulator; stabilization_method=:regularization, projection_tolerance=1e-6, verbose=false, rank_tol=1e-12)

Internal helper that fits the reference Gaussian model for Mahalanobis distance.

Computes mean and covariance from reference samples `B` and returns a closure
that applies a stabilized inverse covariance operator. Two stabilization modes
are supported:
- `:regularization`: invert `Sigma + regulator*I` via Cholesky,
- `:projection`: use eigenvalue filtering and pseudoinverse projection.

# Returns
`(mu, inv_mul)` where `mu` is the reference mean vector and `inv_mul(d)` applies
the stabilized inverse covariance action to deviation vector `d`.

# Arguments
- `B`: Reference sample vectors used to fit mean and covariance.
- `regulator`: Nonnegative diagonal regularization added to covariance.

# Keyword Arguments
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff when `stabilization_method = :projection`.
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.

# Throws
- `ArgumentError`: Raised when `stabilization_method` is not one of `:regularization` or `:projection`.
- `DomainError`: Raised when covariance stabilization cannot produce a usable inverse under the chosen settings.

"""
function _fit_reference(
    B::Vector{Vector{Float64}},
    regulator::Float64;
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-6,
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
)
    n = length(B)
    d = length(B[1])
    X = Matrix{Float64}(undef, n, d)
    @inbounds for i in 1:n
        X[i, :] = B[i]
    end
    mu = vec(Statistics.mean(X; dims = 1))
    Sigma = Statistics.cov(X; dims = 1)
    if verbose && stabilization_method == :regularization
        evals = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Sigma)).values
        rank = count(>(rank_tol), evals)
        near_zero = count(x -> x <= rank_tol, evals)
        println("Covariance rank: $rank/$d (near-zero eigs: $near_zero, tol=$(rank_tol))")
    end
    if stabilization_method == :projection
        eig = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Sigma))
        evals = eig.values
        evecs = eig.vectors
        keep = evals .> projection_tolerance
        if !any(keep)
            throw(DomainError(projection_tolerance, "All eigenvalues are below projection_tolerance; cannot form pseudoinverse"))
        end
        if verbose
            println("Projection kept directions: $(count(keep))/$d (tol=$(projection_tolerance))")
        end
        V = evecs[:, keep]
        Λinv = LinearAlgebra.Diagonal(1.0 ./ evals[keep])
        Sigma_inv = V * Λinv * V'
        inv_mul = dvec -> Sigma_inv * dvec
        return mu, inv_mul
    elseif stabilization_method == :regularization
        Sigma_reg = Sigma + regulator * LinearAlgebra.I
        F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Sigma_reg); check = false)
        if !LinearAlgebra.issuccess(F)
            throw(DomainError(regulator, "Covariance matrix not invertible; increase regulator or use stabilization_method = :projection"))
        end
        inv_mul = dvec -> F \ dvec
        return mu, inv_mul
    else
        throw(ArgumentError("Unknown stabilization_method=$stabilization_method. Use :regularization or :projection"))
    end
end

"""
    _mahal_sigmas(X, mu, inv_mul)

Internal routine computing Mahalanobis sigma distances for a batch.

For each sample `x` in `X`, returns `sqrt((x-mu)' * Sigma^{-1} * (x-mu))`, with
the inverse covariance action provided by `inv_mul`.

# Arguments
- `X`: Samples for which Mahalanobis sigmas are computed.
- `mu`: Reference mean vector.
- `inv_mul`: Function applying the stabilized inverse covariance.

# Returns
- `result`: Vector of Mahalanobis sigma distances, one per input sample.

"""
function _mahal_sigmas(
    X::Vector{Vector{Float64}},
    mu::Vector{Float64},
    inv_mul::Function,
)
    s = Vector{Float64}(undef, length(X))
    @inbounds for i in eachindex(X)
        d = X[i] .- mu
        y = inv_mul(d)
        qf = LinearAlgebra.dot(d, y)
        s[i] = _sqrt_with_tolerance(qf; name = "Mahalanobis quadratic form")
    end
    return s
end

"""
    _summary_stat(sigmas, q)

Internal reducer for sigma-distance vectors.

Returns the minimum when `q == 0.0` (smallest-gap criterion), otherwise returns
the empirical `q`-quantile.

# Arguments
- `sigmas`: Sigma-distance values to summarize.
- `q`: Quantile parameter in [0, 1].

# Returns
- `result`: Scalar summary statistic (minimum for q==0, maximum for q==1, otherwise q-quantile).

"""
function _summary_stat(sigmas::Vector{Float64}, q::Float64)
    if q == 0.0
        return minimum(sigmas)
    elseif q == 1.0
        return maximum(sigmas)  
    end
    return Statistics.quantile(sigmas, q)
end

"""
    _mahal_resample_once(seed, X, regulator, q, stabilization_method, projection_tolerance, verbose, rank_tol)

Run one null-resampling draw for Mahalanobis-gap statistics.

This helper uses `seed` to deterministically split `X` into equal halves,
fits the reference model on one half, evaluates Mahalanobis sigmas on the
other half, and reduces them with `_summary_stat(..., q)`.

# Arguments
- `seed`: Seed for the resampling RNG.
- `X`: Vector-valued input data.
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `q`: Quantile parameter in [0, 1].
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff for projection stabilization.
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.

# Returns
- `result`: Single resampled summary-statistic value.
"""
function _mahal_resample_once(
    seed::UInt64,
    X::Vector{Vector{Float64}},
    regulator::Float64,
    q::Float64,
    stabilization_method::Symbol,
    projection_tolerance::Float64,
    verbose::Bool,
    rank_tol::Float64,
)
    rng_r = Random.Xoshiro(seed)
    X1, X2 = _random_split_equal(X, rng_r)
    mu2, inv2 = _fit_reference(
        X2,
        regulator;
        stabilization_method = stabilization_method,
        projection_tolerance = projection_tolerance,
        verbose = verbose,
        rank_tol = rank_tol,
    )
    sig1 = _mahal_sigmas(X1, mu2, inv2)
    return _summary_stat(sig1, q)
end

"""
    _mahal_resample_many(seeds, X, regulator, q, stabilization_method, projection_tolerance, verbose, rank_tol)

Compute Mahalanobis null-resampling statistics for multiple seeds in serial.

# Arguments
- `seeds`: Seed values for independent resampling draws.
- `X`: Vector-valued input data.
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `q`: Quantile parameter in [0, 1].
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff for projection stabilization.
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.

# Returns
- `result`: Vector of summary statistics, one value per seed.
"""
function _mahal_resample_many(
    seeds::Vector{UInt64},
    X::Vector{Vector{Float64}},
    regulator::Float64,
    q::Float64,
    stabilization_method::Symbol,
    projection_tolerance::Float64,
    verbose::Bool,
    rank_tol::Float64,
)
    out = Vector{Float64}(undef, length(seeds))
    @inbounds for r in eachindex(seeds)
        out[r] = _mahal_resample_once(
            seeds[r],
            X,
            regulator,
            q,
            stabilization_method,
            projection_tolerance,
            verbose,
            rank_tol,
        )
    end
    return out
end

"""
    _mahal_resample_many_progress(seeds, X, regulator, q, stabilization_method, projection_tolerance, verbose, rank_tol; desc="mahalanobis null", progress_step=nothing)

Compute Mahalanobis null-resampling statistics in serial with progress updates.

# Arguments
- `seeds`: Seed values for independent resampling draws.
- `X`: Vector-valued input data.
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `q`: Quantile parameter in [0, 1].
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff for projection stabilization.
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.

# Keyword Arguments
- `desc`: Progress-bar description.
- `progress_step`: Optional progress update stride.

# Returns
- `result`: Vector of summary statistics, one value per input seed.
"""
function _mahal_resample_many_progress(
    seeds::Vector{UInt64},
    X::Vector{Vector{Float64}},
    regulator::Float64,
    q::Float64,
    stabilization_method::Symbol,
    projection_tolerance::Float64,
    verbose::Bool,
    rank_tol::Float64;
    desc::AbstractString = "mahalanobis null",
    progress_step::Union{Nothing,Int} = nothing,
)
    total = length(seeds)
    out = Vector{Float64}(undef, total)
    pm = ProgressMeter.Progress(total; desc = desc)
    step = progress_step === nothing ? 1 : max(1, progress_step)
    @inbounds for r in eachindex(seeds)
        out[r] = _mahal_resample_once(
            seeds[r],
            X,
            regulator,
            q,
            stabilization_method,
            projection_tolerance,
            verbose,
            rank_tol,
        )
        if r == total || r % step == 0
            ProgressMeter.update!(pm, r)
        end
    end
    return out
end

"""
    _mahal_resample_many_threaded(seeds, X, regulator, q, stabilization_method, projection_tolerance, verbose, rank_tol)

Compute Mahalanobis null-resampling statistics in parallel across threads.

# Arguments
- `seeds`: Seed values for independent resampling draws.
- `X`: Vector-valued input data.
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `q`: Quantile parameter in [0, 1].
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff for projection stabilization.
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.

# Returns
- `result`: Vector of summary statistics, one value per input seed.
"""
function _mahal_resample_many_threaded(
    seeds::Vector{UInt64},
    X::Vector{Vector{Float64}},
    regulator::Float64,
    q::Float64,
    stabilization_method::Symbol,
    projection_tolerance::Float64,
    verbose::Bool,
    rank_tol::Float64,
    n_threads::Int,
)
    out = Vector{Float64}(undef, length(seeds))
    n_threads = min(n_threads, Threads.maxthreadid())
    Threads.@threads for t in 1:n_threads
        for r in t:n_threads:length(seeds)
            @inbounds out[r] = _mahal_resample_once(
                seeds[r],
                X,
                regulator,
                q,
                stabilization_method,
                projection_tolerance,
                verbose,
                rank_tol,
            )
        end
    end
    return out
end

"""
    _random_split_equal(B, rng)

Internal helper for null resampling that randomly partitions `B` into two equal
halves of size `length(B) ÷ 2` (dropping one element if odd).

# Arguments
- `B`: Sample set to split into two equal random halves.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Tuple of equally sized random splits (B1, B2).

# Throws
- `DomainError`: If `length(B) < 2`.
"""
function _random_split_equal(B::Vector{Vector{Float64}}, rng)
    n = length(B)
    n2 = n ÷ 2
    if !(n2 >= 1)
        throw(DomainError(n, "need at least 2 samples to split"))
    end
    perm = Random.randperm(rng, n)
    B1 = B[perm[1:n2]]
    B2 = B[perm[(n2 + 1):(2 * n2)]]
    return B1, B2
end

@inline function _safe_cov_matrix(X::AbstractMatrix{<:Real})::Matrix{Float64}
    n, d = size(X)
    if n <= 1
        return zeros(Float64, d, d)
    end
    return Matrix{Float64}(Statistics.cov(Matrix{Float64}(X); dims = 1))
end

@inline function _matrix_from_vecs(V::Vector{Vector{Float64}})::Matrix{Float64}
    n = length(V)
    d = length(V[1])
    X = Matrix{Float64}(undef, n, d)
    @inbounds for i in 1:n
        X[i, :] = V[i]
    end
    return X
end

function _pooled_project_nonzero(
    A::Matrix{Float64},
    B::Matrix{Float64},
    eps_rel::Float64,
    verbose::Bool = false,
)
    X = vcat(A, B)
    Σp = _safe_cov_matrix(X)
    eig = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Σp))
    λ = eig.values
    U = eig.vectors
    medλ = Statistics.median(λ)
    cutoff = eps_rel * medλ
    keep = findall(>(cutoff), λ)
    if isempty(keep)
        keep = [argmax(λ)]
    end
    if verbose
        _log_projection_info("mahalanobis", length(λ) - length(keep), length(λ), eps_rel)
    end
    Uk = U[:, keep]
    return A * Uk, B * Uk
end

@inline function _split_indices(n::Int, rng)::Tuple{Vector{Int},Vector{Int}}
    n2 = n ÷ 2
    perm = Random.randperm(rng, n)
    return perm[1:n2], perm[(n2 + 1):(2 * n2)]
end

function _split_floor_estimates_small_modes(
    X::Matrix{Float64},
    U_small::AbstractMatrix{<:Real},
    seeds::Vector{UInt64},
)
    m = size(U_small, 2)
    if m == 0
        return Matrix{Float64}(undef, 0, 0)
    end
    R = length(seeds)
    out = Matrix{Float64}(undef, m, 2R)
    @inbounds for r in 1:R
        rng_r = Random.Xoshiro(seeds[r])
        i1, i2 = _split_indices(size(X, 1), rng_r)
        X1 = @view X[i1, :]
        X2 = @view X[i2, :]
        μ1 = vec(Statistics.mean(X1; dims = 1))
        μ2 = vec(Statistics.mean(X2; dims = 1))
        Z1 = (X1 .- transpose(μ1)) * U_small
        Z2 = (X2 .- transpose(μ2)) * U_small
        for j in 1:m
            out[j, 2r - 1] = Statistics.var(@view Z1[:, j]; corrected = true)
            out[j, 2r] = Statistics.var(@view Z2[:, j]; corrected = true)
        end
    end
    return out
end

@inline function _mahal_sigmas_projected(
    X::AbstractMatrix{<:Real},
    mu::Vector{Float64},
    U::Matrix{Float64},
    w::Vector{Float64},
)::Vector{Float64}
    n = size(X, 1)
    d = size(X, 2)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        z = transpose(U) * (xi .- mu)
        qf = 0.0
        for j in 1:d
            qf += (z[j] * z[j]) / w[j]
        end
        out[i] = _sqrt_with_tolerance(qf; name = "Mahalanobis quadratic form")
    end
    return out
end

function _build_regularized_model_and_baseline(
    X::Matrix{Float64},
    q::Float64,
    seeds::Vector{UInt64},
    eps_rel::Float64,
    to_regularize_rel::Float64,
    ;
    progress::Bool = false,
    desc::AbstractString = "mahalanobis mc",
)
    Σ = _safe_cov_matrix(X)
    eig = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Σ))
    λ = map(x -> max(x, 0.0), eig.values)
    U = eig.vectors
    med_full = Statistics.median(λ)
    floor_abs = eps_rel * med_full

    split_med = zeros(Float64, length(λ))
    reg_cutoff = to_regularize_rel * med_full
    small_idx = findall(<(reg_cutoff), λ)
    if !isempty(small_idx)
        U_small = @view U[:, small_idx]
        split_diag_small = _split_floor_estimates_small_modes(X, U_small, seeds)
        split_med_small = vec(mapslices(Statistics.median, split_diag_small; dims = 2))
        split_med[small_idx] .= split_med_small
    end
    w = max.(λ, split_med, floor_abs)

    R = length(seeds)
    base = Vector{Float64}(undef, R)
    n = size(X, 1)
    pm = progress ? ProgressMeter.Progress(R; desc = desc) : nothing
    @inbounds for r in 1:R
        rng_r = Random.Xoshiro(seeds[r])
        i1, i2 = _split_indices(n, rng_r)
        X1 = @view X[i1, :]
        X2 = @view X[i2, :]
        mu2 = vec(Statistics.mean(X2; dims = 1))
        sig = _mahal_sigmas_projected(X1, mu2, U, w)
        base[r] = _summary_stat(sig, q)
        progress && ProgressMeter.next!(pm)
    end

    mu_full = vec(Statistics.mean(X; dims = 1))
    return (mu = mu_full, U = U, w = w, baseline = base)
end

"""
    mahalanobis_gap_distinguishability(vecs_a, vecs_b; regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, projection_tolerance=1e-6, to_regularize_rel=0.01, progress=false)

Compute Mahalanobis-gap distinguishability for two vector-valued datasets.

Pipeline:
1. Preprocess vectors to common dimensional support.
2. Fit reference model on `vecs_b`.
3. Compute summary statistic `M_obs` from Mahalanobis sigmas of `vecs_a`.
4. Build baseline null distribution by random equal splits of the reference set.
5. Compare `M_obs` against the `(1-alpha)` baseline quantile threshold.
6. Optionally repeat in reverse direction when `symmetric=true`.

# Returns
Named tuple
`(M_obs, distinguishable, threshold, z_emp, M_obs_sym, M_obs_min, threshold_sym, threshold_max)`.

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.

# Keyword Arguments
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `R`: Number of baseline resampling runs.
- `q`: Quantile parameter in [0, 1].
- `alpha`: Significance level used for thresholding (1 - alpha).
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: If true, also evaluate the reverse direction.
- `projection_tolerance`: Relative pooled/full eigenvalue cutoff for near-null projection/flooring.
- `to_regularize_rel`: Relative eigenvalue cutoff for split-based small-mode variance regularization.
- `progress`: If true, show progress for Monte Carlo baseline sampling.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function mahalanobis_gap_distinguishability(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}};
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    projection_tolerance::Float64 = 1e-6,
    to_regularize_rel::Float64 = 0.01,
    progress::Bool = false,
    verbose::Bool = false,
)
    isempty(vecs_a) && throw(ArgumentError("vecs_a must be non-empty"))
    isempty(vecs_b) && throw(ArgumentError("vecs_b must be non-empty"))
    if !(R > 0)
        throw(DomainError(R, "R must be positive"))
    end
    if !(0.0 <= q <= 1.0)
        throw(DomainError(q, "q must be in [0,1]"))
    end
    if !(0.0 <= alpha < 1.0)
        throw(DomainError(alpha, "alpha must be in [0,1)"))
    end
    if !(regulator >= 0.0)
        throw(DomainError(regulator, "regulator must be nonnegative"))
    end
    if !(to_regularize_rel > 0.0)
        throw(DomainError(to_regularize_rel, "to_regularize_rel must be positive"))
    end
    A, B = _prepare_vectors_for_mahalanobis(vecs_a, vecs_b)
    Am = _matrix_from_vecs(A)
    Bm = _matrix_from_vecs(B)
    Ap, Bp = _pooled_project_nonzero(Am, Bm, projection_tolerance, verbose)

    seeds_b = rand(rng, UInt64, R)
    modelB = _build_regularized_model_and_baseline(
        Bp,
        q,
        seeds_b,
        projection_tolerance,
        to_regularize_rel;
        progress = progress,
        desc = "mahalanobis mc (B)",
    )
    sigA = _mahal_sigmas_projected(Ap, modelB.mu, modelB.U, modelB.w)
    M_obs = _summary_stat(sigA, q)

    M_obs_sym = nothing
    M_obs_min = nothing
    threshold_sym = nothing
    threshold_max = nothing
    D_sym = nothing
    if symmetric
        seeds_a = rand(rng, UInt64, R)
        modelA = _build_regularized_model_and_baseline(
            Ap,
            q,
            seeds_a,
            projection_tolerance,
            to_regularize_rel;
            progress = progress,
            desc = "mahalanobis mc (A)",
        )
        sigB = _mahal_sigmas_projected(Bp, modelA.mu, modelA.U, modelA.w)
        M_obs_sym = _summary_stat(sigB, q)
        threshold_sym = _summary_stat(modelA.baseline, 1 - alpha)
        z_emp_sym = (M_obs_sym - Statistics.mean(modelA.baseline)) / (Statistics.std(modelA.baseline) + eps())
        D_sym = Distributions.cdf(Distributions.Normal(), z_emp_sym)
    end

    S_base = modelB.baseline
    threshold = _summary_stat(S_base, 1 - alpha)

    if symmetric
        M_obs_min = min(M_obs, M_obs_sym)
        threshold_max = max(threshold, threshold_sym)
        distinguishable = (M_obs > threshold) && (M_obs_sym > threshold_sym)
    else
        distinguishable = M_obs > threshold
    end
    z_emp = (M_obs - Statistics.mean(S_base)) / (Statistics.std(S_base) + eps())
    D = Distributions.cdf(Distributions.Normal(), z_emp)

    return (
        M_obs = M_obs,
        D = D,
        distinguishable = distinguishable,
        threshold = threshold,
        z_emp = z_emp,
        M_obs_sym = M_obs_sym,
        D_sym = D_sym,
        M_obs_min = M_obs_min,
        threshold_sym = threshold_sym,
        threshold_max = threshold_max,
    )
end

"""
    scalar_bin_mahalanobis_gap_distinguishability(data::AbstractVector{<:AbstractVector}; num_bins=nothing, regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, projection_tolerance=1e-6, to_regularize_rel=0.01, progress=false)

Bin-pair version: compare every bin to every other bin. Returns a vector of
`(s1, s2, rel_change, M_obs, distinguishable, threshold, z_emp, M_obs_sym, M_obs_min, threshold_sym, threshold_max)`.

# Arguments
- `data`: Input dataset(s) consumed by this method.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `R`: Number of baseline resampling runs.
- `q`: Quantile parameter in [0, 1].
- `alpha`: Significance level used for thresholding (1 - alpha).
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: If true, also evaluate the reverse direction.
- `projection_tolerance`: Relative pooled/full eigenvalue cutoff for near-null projection/flooring.
- `to_regularize_rel`: Relative eigenvalue cutoff for split-based small-mode variance regularization.
- `progress`: If true, display progress while processing bin tasks.

# Returns
- `result`: Vector of named tuples with Mahalanobis-gap metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_mahalanobis_gap_distinguishability(
    data::AbstractVector{<:AbstractVector};
    num_bins::Union{Nothing,Int} = nothing,
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    projection_tolerance::Float64 = 1e-6,
    to_regularize_rel::Float64 = 0.01,
    progress::Bool = false,
    verbose::Bool = false,
)
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_mahalanobis_gap_distinguishability";
        num_bins = num_bins,
    )
    bins = ctx.bins

    n_bins = length(bins)
    total = n_bins * (n_bins - 1) ÷ 2
    tasks = Vector{Tuple{Int,Int}}(undef, total)
    t = 1
    for i in 1:n_bins
        for j in (i + 1):n_bins
            tasks[t] = (i, j)
            t += 1
        end
    end
    seeds = rand(rng, UInt64, total)
    out = Vector{NamedTuple}(undef, total)
    use_pm = false
    pm = nothing
    if progress
        pm = ProgressMeter.Progress(total; desc = "mahalanobis bins")
        use_pm = true
    end
    compute_task = function (k::Int)
        i, j = tasks[k]
        s1, vals1 = bins[i]
        s2, vals2 = bins[j]
        rng_k = Random.Xoshiro(seeds[k])
        res = mahalanobis_gap_distinguishability(
            vals1,
            vals2;
            regulator = regulator,
            R = R,
            q = q,
            alpha = alpha,
            rng = rng_k,
            symmetric = symmetric,
            projection_tolerance = projection_tolerance,
            to_regularize_rel = to_regularize_rel,
            progress = progress,
            verbose = verbose,
        )
        return (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), M_obs = res.M_obs, distinguishable = res.distinguishable, threshold = res.threshold, z_emp = res.z_emp, M_obs_sym = res.M_obs_sym, M_obs_min = res.M_obs_min, threshold_sym = res.threshold_sym, threshold_max = res.threshold_max)
    end

    n_threads = Threads.maxthreadid()
    if n_threads > 1 && total > 1
        Threads.@threads for t in 1:n_threads
            for k in t:n_threads:total
                @inbounds out[k] = compute_task(k)
            end
        end
        if progress
            for _ in 1:total
                use_pm ? ProgressMeter.next!(pm) : nothing
            end
        end
    else
        done = 0
        for k in 1:total
            out[k] = compute_task(k)
            done += 1
            if progress
                if use_pm
                    ProgressMeter.next!(pm)
                elseif done % max(1, round(Int, total * 0.05)) == 0 || done == total
                    println("Progress: $done/$total")
                end
            end
        end
    end
    return out
end

"""
    scalar_bin_mahalanobis_gap_distinguishability(data::AbstractVector{<:AbstractVector}, ref::AbstractVector; num_bins=nothing, regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, projection_tolerance=1e-6, to_regularize_rel=0.01, progress=false)

See `scalar_bin_mahalanobis_gap_distinguishability(data; ...)`.

Reference variant: compares each scalar bin to fixed reference sample `ref`
instead of comparing all bin pairs.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `ref`: Reference sample used for bin-wise comparison.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `regulator`: Nonnegative diagonal regularization added to covariance.
- `R`: Number of baseline resampling runs.
- `q`: Quantile parameter in [0, 1].
- `alpha`: Significance level used for thresholding (1 - alpha).
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: If true, also evaluate the reverse direction.
- `projection_tolerance`: Relative pooled/full eigenvalue cutoff for near-null projection/flooring.
- `to_regularize_rel`: Relative eigenvalue cutoff for split-based small-mode variance regularization.
- `progress`: If true, display progress while processing bin tasks.

# Returns
- `result`: Vector of named tuples with Mahalanobis-gap metrics per bin pair or per bin vs. reference.

# Throws
- `DimensionMismatch`: Raised for incompatible input sizes or expected dataset counts.
- `TypeError`: Raised when value types do not match required contracts.
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function scalar_bin_mahalanobis_gap_distinguishability(
    data::AbstractVector{<:AbstractVector},
    ref::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    projection_tolerance::Float64 = 1e-6,
    to_regularize_rel::Float64 = 0.01,
    progress::Bool = false,
    verbose::Bool = false,
)
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_mahalanobis_gap_distinguishability";
        num_bins = num_bins,
        ref = ref,
    )
    bins = ctx.bins

    total = length(bins)
    seeds = rand(rng, UInt64, total)
    out = Vector{NamedTuple}(undef, total)
    use_pm = false
    pm = nothing
    if progress
        pm = ProgressMeter.Progress(total; desc = "mahalanobis bins")
        use_pm = true
    end
    compute_idx = function (k::Int)
        s, vals = bins[k]
        rng_k = Random.Xoshiro(seeds[k])
        res = mahalanobis_gap_distinguishability(
            vals,
            ref;
            regulator = regulator,
            R = R,
            q = q,
            alpha = alpha,
            rng = rng_k,
            symmetric = symmetric,
            projection_tolerance = projection_tolerance,
            to_regularize_rel = to_regularize_rel,
            progress = progress,
            verbose = verbose,
        )
        return (scalar = s, M_obs = res.M_obs, distinguishable = res.distinguishable, threshold = res.threshold, z_emp = res.z_emp, M_obs_sym = res.M_obs_sym, M_obs_min = res.M_obs_min, threshold_sym = res.threshold_sym, threshold_max = res.threshold_max)
    end

    n_threads = Threads.maxthreadid()
    if n_threads > 1 && total > 1
        Threads.@threads for t in 1:n_threads
            for k in t:n_threads:total
                @inbounds out[k] = compute_idx(k)
            end
        end
        if progress
            for _ in 1:total
                use_pm ? ProgressMeter.next!(pm) : nothing
            end
        end
    else
        done = 0
        for k in 1:total
            out[k] = compute_idx(k)
            done += 1
            if progress
                if use_pm
                    ProgressMeter.next!(pm)
                elseif done % max(1, round(Int, total * 0.05)) == 0 || done == total
                    println("Progress: $done/$total")
                end
            end
        end
    end
    return out
end
