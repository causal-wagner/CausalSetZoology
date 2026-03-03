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
) where {T}
    ctx = _prepare_scalar_bin_context(data, "scalar_bin_distinguishability"; num_bins = num_bins)
    return _map_scalar_bin_pairs(ctx.bins) do s1, vals1, s2, vals2
        res = histogram_distinguishability(vals1, vals2)
        (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D = res.D)
    end
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
) where {T}
    ctx = _prepare_scalar_bin_context(data, "scalar_bin_distinguishability"; num_bins = num_bins)
    return _map_scalar_bin_pairs(ctx.bins) do s1, vals1, s2, vals2
        res = histogram_distinguishability(vals1, vals2, num_draws; rng = rng)
        (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D = res.D, std = res.std)
    end
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
) where {T}
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_distinguishability";
        num_bins = num_bins,
        ref = ref,
    )
    return _map_scalar_bin_reference(ctx.bins) do s, vals
        res = histogram_distinguishability(vals, ref)
        (scalar = s, D = res.D)
    end
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
) where {T}
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_distinguishability";
        num_bins = num_bins,
        ref = ref,
    )
    return _map_scalar_bin_reference(ctx.bins) do s, vals
        res = histogram_distinguishability(vals, ref, num_draws; rng = rng)
        (scalar = s, D = res.D, std = res.std)
    end
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
) where {T}
    ctx = _prepare_scalar_bin_context(data, "scalar_bin_distinguishability_permutation"; num_bins = num_bins)
    return _map_scalar_bin_pairs(ctx.bins) do s1, vals1, s2, vals2
        res = histogram_distinguishability_permutation(vals1, vals2; n_perm = n_perm, rng = rng)
        (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
    end
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
) where {T}
    ctx = _prepare_scalar_bin_context(data, "scalar_bin_distinguishability_permutation"; num_bins = num_bins)
    return _map_scalar_bin_pairs(ctx.bins) do s1, vals1, s2, vals2
        res = histogram_distinguishability_permutation(vals1, vals2, num_draws; n_perm = n_perm, rng = rng)
        (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
    end
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
) where {T}
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_distinguishability_permutation";
        num_bins = num_bins,
        ref = ref,
    )
    return _map_scalar_bin_reference(ctx.bins) do s, vals
        res = histogram_distinguishability_permutation(vals, ref; n_perm = n_perm, rng = rng)
        (scalar = s, D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
    end
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
) where {T}
    ctx = _prepare_scalar_bin_context(
        data,
        "scalar_bin_distinguishability_permutation";
        num_bins = num_bins,
        ref = ref,
    )
    return _map_scalar_bin_reference(ctx.bins) do s, vals
        res = histogram_distinguishability_permutation(vals, ref, num_draws; n_perm = n_perm, rng = rng)
        (scalar = s, D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
    end
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
        sp[i] = sqrt(Float64(p[i]))
        sq[i] = sqrt(Float64(q[i]))
    end
    spp = LinearAlgebra.dot(sp, sp)
    sqq = LinearAlgebra.dot(sq, sq)
    spq = LinearAlgebra.dot(sp, sq)
    d2 = (spp + sqq - 2 * spq) / 2
    return sqrt(max(d2, 0.0))
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
            s[j] = sqrt(Float64(v[j]))
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

"""
    _distance_matrix_exact(vecs)

Build the exact pairwise Hellinger distance matrix for a vector dataset.

# Arguments
- `vecs`: Vector-valued input data.

# Returns
- `result`: Dense symmetric matrix `D` with `D[i, j] = hellinger_distance(vecs[i], vecs[j])`.
"""
function _distance_matrix_exact(vecs::Vector{<:AbstractVector{<:Real}})
    n = length(vecs)
    sq, norms = _sqrt_vectors_and_norms(vecs)
    D = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        D[i, i] = 0.0
    end
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        d = _hellinger_from_sqrt(sq[i], sq[j], norms[i], norms[j])
        D[i, j] = d
        D[j, i] = d
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
    histogram_distinguishability(hists_a, hists_b)

Compute the normalized energy-distance distinguishability D ∈ [0,1] between
two histogram/vector samples. Inputs can be:
- `Vector{<:AbstractDict}` histograms (will be normalized to probabilities),
- `Vector{<:AbstractVector{<:Real}}` already-normalized vectors.

Returns a named tuple `(D = value)`.
For histogram inputs, both sets are normalized with `normalize_hists(..., :probability)`
and then trimmed to the maximal nonzero bin of the union.

# Arguments
- `hists_a`: Histogram input data.
- `hists_b`: Histogram input data.

# Returns
- `result`: Named tuple containing distinguishability `D` and, for Monte Carlo overloads, `std`.

# Throws
- `DomainError`: Propagated for invalid `num_draws`.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function histogram_distinguishability(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
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

    return histogram_distinguishability(vecs_a, vecs_b)
end

"""
    histogram_distinguishability(hists_a, hists_b, num_draws; rng=...)

See `histogram_distinguishability(hists_a, hists_b)`.

This overload uses Monte Carlo estimation with `num_draws` and returns
`(D, std)` instead of only `(D,)`.

# Arguments
- `hists_a`: Histogram input data.
- `hists_b`: Histogram input data.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Named tuple containing distinguishability `D` and, for Monte Carlo overloads, `std`.

# Throws
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function histogram_distinguishability(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
    num_draws::Int;
    rng = Random.default_rng(),
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

    return histogram_distinguishability(vecs_a, vecs_b, num_draws; rng = rng)
end

"""
    histogram_distinguishability(vecs_a, vecs_b)

See `histogram_distinguishability(hists_a, hists_b)`.

This is the exact vector-input implementation (no dict normalization stage).

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.

# Returns
- `result`: Named tuple containing distinguishability `D` and, for Monte Carlo overloads, `std`.

# Throws
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function histogram_distinguishability(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    n1 = length(vecs_a)
    n2 = length(vecs_b)
    A, B = _prepare_vectors_for_distance(vecs_a, vecs_b)
    sqA, normA = _sqrt_vectors_and_norms(A)
    sqB, normB = _sqrt_vectors_and_norms(B)

    sum_xy = _pairwise_hellinger_sum_sqrt(sqA, normA, sqB, normB)
    exy = sum_xy / (n1 * n2)

    sum_xx = _pairwise_hellinger_sum_sqrt(sqA, normA, sqA, normA)
    exx = sum_xx / (n1 * n1)

    sum_yy = _pairwise_hellinger_sum_sqrt(sqB, normB, sqB, normB)
    eyy = sum_yy / (n2 * n2)

    E = 2 * exy - exx - eyy
    W = 0.5 * (exx + eyy)
    denom = E + W
    D = denom == 0 ? 0.0 : E / denom
    return (D = D,)
end

"""
    histogram_distinguishability(vecs_a, vecs_b, num_draws; rng=...)

See `histogram_distinguishability(vecs_a, vecs_b)`.

Monte Carlo variant using `num_draws`; returns `(D, std)`.

# Arguments
- `vecs_a`: Vector-valued input data.
- `vecs_b`: Vector-valued input data.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Named tuple containing distinguishability `D` and, for Monte Carlo overloads, `std`.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid/empty inputs or unsupported combinations.
"""
function histogram_distinguishability(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
    num_draws::Int;
    rng = Random.default_rng(),
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
    A, B = _prepare_vectors_for_distance(vecs_a, vecs_b)
    n1 = length(A)
    n2 = length(B)
    Dmat = _distance_matrix_exact(vcat(A, B))

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
    W = 0.5 * (exx + eyy)
    denom = E + W
    if denom == 0
        return (D = 0.0, std = 0.0)
    end

    var_E = 4 * var_exy + var_exx + var_eyy
    var_W = 0.25 * (var_exx + var_eyy)
    cov_EW = -0.5 * (var_exx + var_eyy)
    var_D = (W^2 * var_E + E^2 * var_W - 2 * E * W * cov_EW) / denom^4
    std_D = sqrt(max(var_D, 0.0))
    return (D = E / denom, std = std_D)
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

    return histogram_distinguishability_permutation(vecs_a, vecs_b; n_perm = n_perm, rng = rng)
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

    return histogram_distinguishability_permutation(vecs_a, vecs_b, num_draws; n_perm = n_perm, rng = rng)
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

    Dmat = _distance_matrix_exact(C)
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
        W = 0.5 * (m_aa + m_bb)
        denom = E + W
        return denom == 0 ? 0.0 : E / denom
    end

    D_obs = compute_D(labels_obs)

    Ds = Vector{Float64}(undef, n_perm)
    for p in 1:n_perm
        perm = Random.randperm(rng, n_total)
        labels = falses(n_total)
        @inbounds for i in 1:n_per
            labels[perm[i]] = true
        end
        Ds[p] = compute_D(labels)
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
    Dmat = _distance_matrix_exact(C)

    if !(num_draws > 0)
        throw(DomainError(num_draws, "num_draws must be positive"))
    end
    pairs_u = Vector{Int}(undef, num_draws)
    pairs_v = Vector{Int}(undef, num_draws)
    dists = Vector{Float64}(undef, num_draws)
    @inbounds for k in 1:num_draws
        i = rand(rng, 1:n_total)
        j = rand(rng, 1:(n_total - 1))
        j = j >= i ? j + 1 : j
        if i > j
            i, j = j, i
        end
        pairs_u[k] = i
        pairs_v[k] = j
        dists[k] = Dmat[i, j]
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
        W = 0.5 * (m_aa + m_bb)
        denom = E + W
        return denom == 0 ? 0.0 : E / denom
    end

    D_obs = compute_D(labels_obs)

    Ds = Vector{Float64}(undef, n_perm)
    for p in 1:n_perm
        perm = Random.randperm(rng, n_total)
        labels = falses(n_total)
        @inbounds for i in 1:n_per
            labels[perm[i]] = true
        end
        Ds[p] = compute_D(labels)
    end

    p_value = (count(>=(D_obs), Ds) + 1) / (n_perm + 1)
    std_Ts = Statistics.std(Ds)
    z_emp = (D_obs - Statistics.mean(Ds)) / (std_Ts + eps())
    p_clamped = clamp(p_value, eps(), 1 - eps())
    z_coll = Distributions.quantile(Distributions.Normal(), 1 - p_clamped)
    return (D_obs = D_obs, p_value = p_value, z_emp = z_emp, z_coll = z_coll, std_Ts = std_Ts)
end

"""
    mahalanobis_gap_distinguishability(A, B; regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, verbose=false, rank_tol=1e-12, stabilization_method=:regularization, projection_tolerance=1e-10)

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
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff when `stabilization_method = :projection`.

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
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    progress_step::Union{Nothing,Int} = nothing,
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

    return mahalanobis_gap_distinguishability(
        vecs_a,
        vecs_b;
        regulator = regulator,
        R = R,
        q = q,
        alpha = alpha,
        rng = rng,
        symmetric = symmetric,
        verbose = verbose,
        rank_tol = rank_tol,
        stabilization_method = stabilization_method,
        projection_tolerance = projection_tolerance,
        progress = progress,
        progress_step = progress_step,
    )
end

"""
    mahalanobis_gap_distinguishability(vals_a, vals_b; kwargs...)

See `mahalanobis_gap_distinguishability(vecs_a, vecs_b; ...)`.

Dispatch helper that forwards homogeneous vectors-of-dicts or
vectors-of-vectors to the corresponding concrete method.

# Arguments
- `vals_a`: First input sample collection.
- `vals_b`: Second input sample collection.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

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
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    progress_step::Union{Nothing,Int} = nothing,
)
    if !(!isempty(vals_a) && !isempty(vals_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
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
            verbose = verbose,
            rank_tol = rank_tol,
            stabilization_method = stabilization_method,
            projection_tolerance = projection_tolerance,
            progress = progress,
            progress_step = progress_step,
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
            verbose = verbose,
            rank_tol = rank_tol,
            stabilization_method = stabilization_method,
            projection_tolerance = projection_tolerance,
            progress = progress,
            progress_step = progress_step,
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


"""
function _prepare_vectors_for_mahalanobis(
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

"""
    _fit_reference(B, regulator; stabilization_method=:regularization, projection_tolerance=1e-10, verbose=false, rank_tol=1e-12)

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
    projection_tolerance::Float64 = 1e-10,
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
        s[i] = sqrt(LinearAlgebra.dot(d, y))
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
- `DomainError`: Raised for out-of-range numeric parameters.
"""
function _random_split_equal(B::Vector{Vector{Float64}}, rng)
    n = length(B)
    n2 = n ÷ 2
    if !(n2 >= 1)
        throw(DomainError(n2, "need at least 2 samples to split"))
    end
    perm = Random.randperm(rng, n)
    B1 = B[perm[1:n2]]
    B2 = B[perm[(n2 + 1):(2 * n2)]]
    return B1, B2
end

"""
    mahalanobis_gap_distinguishability(vecs_a, vecs_b; regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, verbose=false, rank_tol=1e-12, stabilization_method=:regularization, projection_tolerance=1e-10)

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
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff when `stabilization_method = :projection`.

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
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    progress_step::Union{Nothing,Int} = nothing,
)
    if !(!isempty(vecs_a) && !isempty(vecs_b))
        throw(ArgumentError("inputs must be non-empty"))
    end
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
    A, B = _prepare_vectors_for_mahalanobis(vecs_a, vecs_b)

    muB, invB = _fit_reference(
        B,
        regulator;
        stabilization_method = stabilization_method,
        projection_tolerance = projection_tolerance,
        verbose = verbose,
        rank_tol = rank_tol,
    )
    sigA = _mahal_sigmas(A, muB, invB)
    M_obs = _summary_stat(sigA, q)

    M_obs_sym = nothing
    M_obs_min = nothing
    threshold_sym = nothing
    threshold_max = nothing
    if symmetric
        muA, invA = _fit_reference(
            A,
            regulator;
            stabilization_method = stabilization_method,
            projection_tolerance = projection_tolerance,
            verbose = verbose,
            rank_tol = rank_tol,
        )
        sigB = _mahal_sigmas(B, muA, invA)
        M_obs_sym = _summary_stat(sigB, q)
    end

    S_base = Vector{Float64}(undef, R)
    seeds = rand(rng, UInt64, R)

    n_threads = Threads.maxthreadid()
    use_threaded = !progress && n_threads > 1 && R > 1

    if use_threaded
        S_base .= _mahal_resample_many_threaded(
            seeds,
            B,
            regulator,
            q,
            stabilization_method,
            projection_tolerance,
            verbose,
            rank_tol,
            n_threads,
        )
    elseif progress
        S_base .= _mahal_resample_many_progress(
            seeds,
            B,
            regulator,
            q,
            stabilization_method,
            projection_tolerance,
            verbose,
            rank_tol;
            desc = "mahalanobis null (B)",
            progress_step = progress_step,
        )
    else
        S_base .= _mahal_resample_many(
            seeds,
            B,
            regulator,
            q,
            stabilization_method,
            projection_tolerance,
            verbose,
            rank_tol,
        )
    end

    threshold = _summary_stat(S_base, 1 - alpha)
    if symmetric
        S_base_sym = Vector{Float64}(undef, R)
        seeds_sym = rand(rng, UInt64, R)
        if use_threaded
            S_base_sym .= _mahal_resample_many_threaded(
                seeds_sym,
                A,
                regulator,
                q,
                stabilization_method,
                projection_tolerance,
                verbose,
                rank_tol,
                n_threads,
            )
        elseif progress
            S_base_sym .= _mahal_resample_many_progress(
                seeds_sym,
                A,
                regulator,
                q,
                stabilization_method,
                projection_tolerance,
                verbose,
                rank_tol;
                desc = "mahalanobis null (A)",
                progress_step = progress_step,
            )
        else
            S_base_sym .= _mahal_resample_many(
                seeds_sym,
                A,
                regulator,
                q,
                stabilization_method,
                projection_tolerance,
                verbose,
                rank_tol,
            )
        end
        threshold_sym = _summary_stat(S_base_sym, 1 - alpha)
    end

    if symmetric
        M_obs_min = min(M_obs, M_obs_sym)
        threshold_max = max(threshold, threshold_sym)
        distinguishable = (M_obs > threshold) && (M_obs_sym > threshold_sym)
    else
        distinguishable = M_obs > threshold
    end
    z_emp = (M_obs - Statistics.mean(S_base)) / (Statistics.std(S_base) + eps())

    return (
        M_obs = M_obs,
        distinguishable = distinguishable,
        threshold = threshold,
        z_emp = z_emp,
        M_obs_sym = M_obs_sym,
        M_obs_min = M_obs_min,
        threshold_sym = threshold_sym,
        threshold_max = threshold_max,
    )
end

"""
    scalar_bin_mahalanobis_gap_distinguishability(data::AbstractVector{<:AbstractVector}; num_bins=nothing, regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, verbose=false, rank_tol=1e-12, stabilization_method=:regularization, projection_tolerance=1e-10, progress=false, progress_step=nothing)

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
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff when `stabilization_method = :projection`.
- `progress`: If true, display progress while processing bin tasks.
- `progress_step`: Manual progress print/update interval when progress display is active.

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
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    progress_step::Union{Nothing,Int} = nothing,
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
    step = progress_step === nothing ? max(1, round(Int, total * 0.05)) : max(1, progress_step)
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
            verbose = verbose,
            rank_tol = rank_tol,
            stabilization_method = stabilization_method,
            projection_tolerance = projection_tolerance,
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
                elseif done % step == 0 || done == total
                    println("Progress: $done/$total")
                end
            end
        end
    end
    return out
end

"""
    scalar_bin_mahalanobis_gap_distinguishability(data::AbstractVector{<:AbstractVector}, ref::AbstractVector; num_bins=nothing, regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, verbose=false, rank_tol=1e-12, stabilization_method=:regularization, projection_tolerance=1e-10, progress=false, progress_step=nothing)

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
- `verbose`: If true, print stabilization diagnostics.
- `rank_tol`: Tolerance for near-zero eigenvalue reporting.
- `stabilization_method`: Covariance inversion strategy (`:regularization` or `:projection`).
- `projection_tolerance`: Eigenvalue cutoff when `stabilization_method = :projection`.
- `progress`: If true, display progress while processing bin tasks.
- `progress_step`: Manual progress print/update interval when progress display is active.

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
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    progress_step::Union{Nothing,Int} = nothing,
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
    step = progress_step === nothing ? max(1, round(Int, total * 0.05)) : max(1, progress_step)
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
            verbose = verbose,
            rank_tol = rank_tol,
            stabilization_method = stabilization_method,
            projection_tolerance = projection_tolerance,
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
                elseif done % step == 0 || done == total
                    println("Progress: $done/$total")
                end
            end
        end
    end
    return out
end
