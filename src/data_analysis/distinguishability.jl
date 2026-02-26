"""
    densify_hists(hists::Vector{<:AbstractDict})

Convert sparse histogram dictionaries to a dense matrix with consistent binning.
Returns a matrix of size (Nsamples, nbins).

# Arguments
- `hists`: Histogram input data.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `densify_hists` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function densify_hists(hists::Vector{<:AbstractDict})
    min_k = minimum(minimum(keys(h)) for h in hists)
    max_k = maximum(maximum(keys(h)) for h in hists)
    shift = (min_k == 0)

    nbins = shift ? max_k + 1 : max_k
    dense = Matrix{Float64}(undef, length(hists), nbins)

    for (i, h) in enumerate(hists)
        fill!(view(dense, i, :), 0.0)
        for (k, v) in h
            idx = shift ? k + 1 : k
            dense[i, idx] = v
        end
    end

    return dense
end

"""
    relative_change(a::Real, b::Real)::Float64

Relative change between two positive scalars: |a-b|/(a+b).

# Arguments
- `a`: Input parameter `a` used by this method.
- `b`: Input parameter `b` used by this method.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result::Float64`: Output of `relative_change` with type annotation `Float64`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function relative_change(a::Real, b::Real)::Float64
    @assert a > 0 && b > 0 "relative_change requires positive scalars"
    return abs(a - b) / (a + b)
end

"""
    bin_scalar_pairs(pairs::Vector{<:Tuple{Any,Real}}, num_bins::Union{Nothing,Int}, bin_edges::Union{Nothing,Vector{<:Real}})

Group `(value, scalar)` pairs into bins. Returns a vector of `(bin_center, values)` pairs.
If `num_bins === nothing`, uses exact scalar values.
If `bin_edges` is provided, uses those edges; otherwise computes edges from the scalars.

# Arguments
- `pairs`: Input parameter `pairs` used by this method.
- `num_bins`: Bin selection or binning control parameter.
- `bin_edges`: Bin selection or binning control parameter.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `bin_scalar_pairs` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function bin_scalar_pairs(
    pairs::Vector{Tuple{T,Real}},
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,Vector{<:Real}} = nothing,
) where {T}
    isempty(pairs) && return Vector{Tuple{Real,Vector{T}}}()

    scalars = [s for (_, s) in pairs]
    if num_bins !== nothing
        @assert num_bins ≥ 1 "num_bins must be >= 1"
    end
    if num_bins !== nothing && bin_edges === nothing
        vmin, vmax = minimum(scalars), maximum(scalars)
        if vmin == vmax
            bin_edges = [vmin, vmax + 1e-12]
        else
            bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
        end
    end

    groups = Dict{Real,Vector{T}}()
    for (v, s) in pairs
        key = if num_bins === nothing
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
- `result`: Output of `scalar_bin_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_distinguishability(
    data::Vector{Vector{Tuple{T,Real}}};
    num_bins::Union{Nothing,Int} = nothing,
) where {T}
    @assert length(data) == 1 "scalar_bin_distinguishability expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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

    out = Vector{NamedTuple}()
    for i in 1:length(bins)
        s1, vals1 = bins[i]
        for j in (i + 1):length(bins)
            s2, vals2 = bins[j]
            res = histogram_distinguishability(vals1, vals2)
            push!(out, (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D = res.D))
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
- `num_draws`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `scalar_bin_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_distinguishability(
    data::Vector{Vector{Tuple{T,Real}}},
    num_draws::Int;
    num_bins::Union{Nothing,Int} = nothing,
    rng = Random.default_rng(),
) where {T}
    @assert length(data) == 1 "scalar_bin_distinguishability expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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

    out = Vector{NamedTuple}()
    for i in 1:length(bins)
        s1, vals1 = bins[i]
        for j in (i + 1):length(bins)
            s2, vals2 = bins[j]
            res = histogram_distinguishability(vals1, vals2, num_draws; rng = rng)
            push!(out, (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D = res.D, std = res.std))
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
- `ref`: Input parameter `ref` used by this method.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.

# Returns
- `result`: Output of `scalar_bin_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_distinguishability(
    data::Vector{Vector{Tuple{T,Real}}},
    ref::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
) where {T}
    @assert length(data) == 1 "scalar_bin_distinguishability expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"
    @assert !isempty(ref) "reference set must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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

    out = Vector{NamedTuple}()
    for (s, vals) in bins
        res = histogram_distinguishability(vals, ref)
        push!(out, (scalar = s, D = res.D))
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
- `ref`: Input parameter `ref` used by this method.
- `num_draws`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `scalar_bin_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_distinguishability(
    data::Vector{Vector{Tuple{T,Real}}},
    ref::AbstractVector,
    num_draws::Int;
    num_bins::Union{Nothing,Int} = nothing,
    rng = Random.default_rng(),
) where {T}
    @assert length(data) == 1 "scalar_bin_distinguishability expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"
    @assert !isempty(ref) "reference set must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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

    out = Vector{NamedTuple}()
    for (s, vals) in bins
        res = histogram_distinguishability(vals, ref, num_draws; rng = rng)
        push!(out, (scalar = s, D = res.D, std = res.std))
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
- `n_perm`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `scalar_bin_distinguishability_permutation` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_distinguishability_permutation(
    data::Vector{Vector{Tuple{T,Real}}};
    num_bins::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    rng = Random.default_rng(),
) where {T}
    @assert length(data) == 1 "scalar_bin_distinguishability_permutation expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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

    out = Vector{NamedTuple}()
    for i in 1:length(bins)
        s1, vals1 = bins[i]
        for j in (i + 1):length(bins)
            s2, vals2 = bins[j]
            res = histogram_distinguishability_permutation(vals1, vals2; n_perm = n_perm, rng = rng)
            push!(out, (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts))
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
- `num_draws`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `n_perm`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `scalar_bin_distinguishability_permutation` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_distinguishability_permutation(
    data::Vector{Vector{Tuple{T,Real}}},
    num_draws::Int;
    num_bins::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    rng = Random.default_rng(),
) where {T}
    @assert length(data) == 1 "scalar_bin_distinguishability_permutation expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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

    out = Vector{NamedTuple}()
    for i in 1:length(bins)
        s1, vals1 = bins[i]
        for j in (i + 1):length(bins)
            s2, vals2 = bins[j]
            res = histogram_distinguishability_permutation(vals1, vals2, num_draws; n_perm = n_perm, rng = rng)
            push!(out, (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts))
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
- `ref`: Input parameter `ref` used by this method.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `n_perm`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `scalar_bin_distinguishability_permutation` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_distinguishability_permutation(
    data::Vector{Vector{Tuple{T,Real}}},
    ref::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    rng = Random.default_rng(),
) where {T}
    @assert length(data) == 1 "scalar_bin_distinguishability_permutation expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"
    @assert !isempty(ref) "reference set must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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

    out = Vector{NamedTuple}()
    for (s, vals) in bins
        res = histogram_distinguishability_permutation(vals, ref; n_perm = n_perm, rng = rng)
        push!(out, (scalar = s, D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts))
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
- `ref`: Input parameter `ref` used by this method.
- `num_draws`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `n_perm`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `scalar_bin_distinguishability_permutation` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_distinguishability_permutation(
    data::Vector{Vector{Tuple{T,Real}}},
    ref::AbstractVector,
    num_draws::Int;
    num_bins::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    rng = Random.default_rng(),
) where {T}
    @assert length(data) == 1 "scalar_bin_distinguishability_permutation expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"
    @assert !isempty(ref) "reference set must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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

    out = Vector{NamedTuple}()
    for (s, vals) in bins
        res = histogram_distinguishability_permutation(vals, ref, num_draws; n_perm = n_perm, rng = rng)
        push!(out, (scalar = s, D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts))
    end
    return out
end

"""
    hellinger_distance(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64

Compute the Hellinger distance between two probability vectors.
Assumes `p` and `q` are nonnegative and have equal length.

# Arguments
- `p`: Input parameter `p` used by this method.
- `q`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result::Float64`: Output of `hellinger_distance` with type annotation `Float64`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function hellinger_distance(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})::Float64
    @assert length(p) == length(q) "Hellinger distance requires equal-length vectors"
    s = 0.0
    @inbounds for i in eachindex(p, q)
        s += (sqrt(p[i]) - sqrt(q[i]))^2
    end
    return sqrt(s / 2)
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

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `histogram_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function histogram_distinguishability(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
)
    @assert !isempty(hists_a) && !isempty(hists_b) "inputs must be non-empty"

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
- `num_draws`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `histogram_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function histogram_distinguishability(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
    num_draws::Int;
    rng = Random.default_rng(),
)
    @assert !isempty(hists_a) && !isempty(hists_b) "inputs must be non-empty"

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

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `histogram_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function histogram_distinguishability(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
)
    @assert !isempty(vecs_a) && !isempty(vecs_b) "inputs must be non-empty"

    n1 = length(vecs_a)
    n2 = length(vecs_b)
    maxlen = maximum(length.(vcat(vecs_a, vecs_b)))

    pad_to(v, n) = length(v) == n ? collect(v) : vcat(collect(v), zeros(Float64, n - length(v)))
    A = [pad_to(v, maxlen) for v in vecs_a]
    B = [pad_to(v, maxlen) for v in vecs_b]

    # trim to maximal nonzero bin across both sets
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

    # energy distance components (exact)
    sum_xy = 0.0
    for i in 1:n1, j in 1:n2
        sum_xy += hellinger_distance(A[i], B[j])
    end
    exy = sum_xy / (n1 * n2)

    sum_xx = 0.0
    for i in 1:n1, j in 1:n1
        sum_xx += hellinger_distance(A[i], A[j])
    end
    exx = sum_xx / (n1 * n1)

    sum_yy = 0.0
    for i in 1:n2, j in 1:n2
        sum_yy += hellinger_distance(B[i], B[j])
    end
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
- `num_draws`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `histogram_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function histogram_distinguishability(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
    num_draws::Int;
    rng = Random.default_rng(),
)
    @assert !isempty(vecs_a) && !isempty(vecs_b) "inputs must be non-empty"

    n1 = length(vecs_a)
    n2 = length(vecs_b)
    maxlen = maximum(length.(vcat(vecs_a, vecs_b)))

    pad_to(v, n) = length(v) == n ? collect(v) : vcat(collect(v), zeros(Float64, n - length(v)))
    A = [pad_to(v, maxlen) for v in vecs_a]
    B = [pad_to(v, maxlen) for v in vecs_b]

    # trim to maximal nonzero bin across both sets
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

    @assert num_draws > 0 "num_draws must be positive"
    m = num_draws

    dxy = Vector{Float64}(undef, m)
    dxx = Vector{Float64}(undef, m)
    dyy = Vector{Float64}(undef, m)

    for t in 1:m
        i = rand(rng, 1:n1); j = rand(rng, 1:n2)
        dxy[t] = hellinger_distance(A[i], B[j])
        i1 = rand(rng, 1:n1); i2 = rand(rng, 1:n1)
        dxx[t] = hellinger_distance(A[i1], A[i2])
        j1 = rand(rng, 1:n2); j2 = rand(rng, 1:n2)
        dyy[t] = hellinger_distance(B[j1], B[j2])
    end

    exy = Statistics.mean(dxy)
    exx = Statistics.mean(dxx)
    eyy = Statistics.mean(dyy)

    var_exy = Statistics.var(dxy) / m
    var_exx = Statistics.var(dxx) / m
    var_eyy = Statistics.var(dyy) / m

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
- `n_perm`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `histogram_distinguishability_permutation` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function histogram_distinguishability_permutation(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict};
    n_perm::Int = 1000,
    rng = Random.default_rng(),
)
    @assert !isempty(hists_a) && !isempty(hists_b) "inputs must be non-empty"

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
- `num_draws`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- `n_perm`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `histogram_distinguishability_permutation` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function histogram_distinguishability_permutation(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict},
    num_draws::Int;
    n_perm::Int = 1000,
    rng = Random.default_rng(),
)
    @assert !isempty(hists_a) && !isempty(hists_b) "inputs must be non-empty"

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
- `n_perm`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `histogram_distinguishability_permutation` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function histogram_distinguishability_permutation(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}};
    n_perm::Int = 1000,
    rng = Random.default_rng(),
)
    @assert !isempty(vecs_a) && !isempty(vecs_b) "inputs must be non-empty"

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

    pairs_u = Vector{Int}()
    pairs_v = Vector{Int}()
    dists = Float64[]
    for i in 1:(n_total - 1), j in (i + 1):n_total
        push!(pairs_u, i)
        push!(pairs_v, j)
        push!(dists, hellinger_distance(C[i], C[j]))
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
- `num_draws`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- `n_perm`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result`: Output of `histogram_distinguishability_permutation` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function histogram_distinguishability_permutation(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}},
    num_draws::Int;
    n_perm::Int = 1000,
    rng = Random.default_rng(),
)
    @assert !isempty(vecs_a) && !isempty(vecs_b) "inputs must be non-empty"

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

    @assert num_draws > 0 "num_draws must be positive"
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
        dists[k] = hellinger_distance(C[i], C[j])
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
    mahalanobis_gap_distinguishability(A, B; regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, num_workers=1)

See `mahalanobis_gap_distinguishability(vecs_a, vecs_b; ...)`.

This overload accepts histogram dictionaries, normalizes/densifies them, and
delegates to the vector implementation.

# Arguments
- `A`: Input parameter `A` used by this method.
- `B`: Input parameter `B` used by this method.

# Keyword Arguments
- `regulator`: Keyword option `regulator` controlling this method's behavior.
- `R`: Numeric control parameter for fitting/sampling resolution.
- `q`: Numeric control parameter for fitting/sampling resolution.
- `alpha`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: Keyword option `symmetric` controlling this method's behavior.
- `num_workers`: Keyword option `num_workers` controlling this method's behavior.

# Returns
- `result`: Output of `mahalanobis_gap_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function mahalanobis_gap_distinguishability(
    hists_a::Vector{<:AbstractDict},
    hists_b::Vector{<:AbstractDict};
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    num_workers::Int = 1,
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
)
    @assert !isempty(hists_a) && !isempty(hists_b) "inputs must be non-empty"
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
        num_workers = num_workers,
        verbose = verbose,
        rank_tol = rank_tol,
        stabilization_method = stabilization_method,
        projection_tolerance = projection_tolerance,
    )
end

"""
    mahalanobis_gap_distinguishability(vals_a, vals_b; kwargs...)

See `mahalanobis_gap_distinguishability(vecs_a, vecs_b; ...)`.

Dispatch helper that forwards homogeneous vectors-of-dicts or
vectors-of-vectors to the corresponding concrete method.

# Arguments
- `vals_a`: Input parameter `vals_a` used by this method.
- `vals_b`: Input parameter `vals_b` used by this method.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: Output of `mahalanobis_gap_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function mahalanobis_gap_distinguishability(
    vals_a::AbstractVector,
    vals_b::AbstractVector;
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    num_workers::Int = 1,
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
)
    @assert !isempty(vals_a) && !isempty(vals_b) "inputs must be non-empty"

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
            num_workers = num_workers,
            verbose = verbose,
            rank_tol = rank_tol,
            stabilization_method = stabilization_method,
            projection_tolerance = projection_tolerance,
        )
    elseif all(v -> v isa AbstractVector, vals_a) && all(v -> v isa AbstractVector, vals_b)
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
            num_workers = num_workers,
            verbose = verbose,
            rank_tol = rank_tol,
            stabilization_method = stabilization_method,
            projection_tolerance = projection_tolerance,
        )
    else
        error("mahalanobis_gap_distinguishability expects vectors of dicts or vectors of numeric vectors.")
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

# Keyword Arguments
- This method has no keyword arguments.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
- `B`: Input parameter `B` used by this method.
- `regulator`: Input parameter `regulator` used by this method.

# Keyword Arguments
- `stabilization_method`: Keyword option `stabilization_method` controlling this method's behavior.
- `projection_tolerance`: Keyword option `projection_tolerance` controlling this method's behavior.
- `verbose`: Boolean toggle controlling output or execution behavior.
- `rank_tol`: Keyword option `rank_tol` controlling this method's behavior.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
            error("All eigenvalues below projection_tolerance; cannot form pseudoinverse.")
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
            error("Covariance matrix not invertible. Increase regularization (regulator > 0) or use stabilization_method = :projection.")
        end
        inv_mul = dvec -> F \ dvec
        return mu, inv_mul
    else
        error("Unknown stabilization_method: $stabilization_method. Use :regularization or :projection.")
    end
end

"""
    _mahal_sigmas(X, mu, inv_mul)

Internal routine computing Mahalanobis sigma distances for a batch.

For each sample `x` in `X`, returns `sqrt((x-mu)' * Sigma^{-1} * (x-mu))`, with
the inverse covariance action provided by `inv_mul`.

# Arguments
- `X`: Input coordinate/data values used in the computation.
- `mu`: Input parameter `mu` used by this method.
- `inv_mul`: Input parameter `inv_mul` used by this method.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `_mahal_sigmas` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
- `sigmas`: Input parameter `sigmas` used by this method.
- `q`: Numeric control parameter for fitting/sampling resolution.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `_summary_stat` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function _summary_stat(sigmas::Vector{Float64}, q::Float64)
    if q == 0.0
        return minimum(sigmas)
    end
    return Statistics.quantile(sigmas, q)
end

"""
    _random_split_equal(B, rng)

Internal helper for null resampling that randomly partitions `B` into two equal
halves of size `length(B) ÷ 2` (dropping one element if odd).

# Arguments
- `B`: Input parameter `B` used by this method.
- `rng`: Random number generator used for stochastic steps.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `_random_split_equal` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function _random_split_equal(B::Vector{Vector{Float64}}, rng)
    n = length(B)
    n2 = n ÷ 2
    @assert n2 >= 1 "need at least 2 samples to split"
    perm = Random.randperm(rng, n)
    B1 = B[perm[1:n2]]
    B2 = B[perm[(n2 + 1):(2 * n2)]]
    return B1, B2
end

"""
    mahalanobis_gap_distinguishability(vecs_a, vecs_b; regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, num_workers=1, verbose=false, rank_tol=1e-12, stabilization_method=:regularization, projection_tolerance=1e-10)

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
- `regulator`: Keyword option `regulator` controlling this method's behavior.
- `R`: Numeric control parameter for fitting/sampling resolution.
- `q`: Numeric control parameter for fitting/sampling resolution.
- `alpha`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: Keyword option `symmetric` controlling this method's behavior.
- `num_workers`: Keyword option `num_workers` controlling this method's behavior.
- `verbose`: Boolean toggle controlling output or execution behavior.
- `rank_tol`: Keyword option `rank_tol` controlling this method's behavior.
- `stabilization_method`: Keyword option `stabilization_method` controlling this method's behavior.
- `projection_tolerance`: Keyword option `projection_tolerance` controlling this method's behavior.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function mahalanobis_gap_distinguishability(
    vecs_a::Vector{<:AbstractVector{<:Real}},
    vecs_b::Vector{<:AbstractVector{<:Real}};
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    num_workers::Int = 1,
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
)
    @assert !isempty(vecs_a) && !isempty(vecs_b) "inputs must be non-empty"
    @assert R > 0 "R must be positive"
    @assert num_workers >= 1 "num_workers must be >= 1"
    @assert 0.0 <= q <= 1.0 "q must be in [0,1]"
    @assert 0.0 <= alpha < 1.0 "alpha must be in [0,1)"
    @assert regulator >= 0.0 "regulator must be nonnegative"

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
    run_one = function (seed::UInt64, X::Vector{Vector{Float64}})
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

    workers_to_use = Int[]
    added_workers = Int[]
    use_distributed = false
    if num_workers > 1 && R > 1
        try
            n_have = Distributed.nworkers()
            if n_have < num_workers
                n_add = num_workers - n_have
                @info "Starting Distributed workers for mahalanobis_gap_distinguishability" requested = num_workers existing = n_have adding = n_add
                added_workers = Distributed.addprocs(n_add)
                @info "Distributed workers started" added = length(added_workers) total = Distributed.nworkers()
            else
                @info "Using existing Distributed workers for mahalanobis_gap_distinguishability" requested = num_workers existing = n_have
            end
            workers_to_use = Distributed.workers()[1:min(num_workers, Distributed.nworkers())]
            if !isempty(workers_to_use)
                @info "Using Distributed workers for mahalanobis resampling" workers = length(workers_to_use)
                utils_file = @__FILE__
                for w in workers_to_use
                    Distributed.remotecall_wait(w) do
                        if !isdefined(Main, :mahalanobis_gap_distinguishability)
                            include(utils_file)
                        end
                        nothing
                    end
                end
                use_distributed = true
            end
        catch err
            @info "Falling back to serial resampling (Distributed setup failed)" error = string(err)
            use_distributed = false
        end
    end

    try
        if use_distributed
            pool = Distributed.CachingPool(workers_to_use)
            S_base .= Distributed.pmap(seed -> run_one(seed, B), pool, seeds)
        else
            @inbounds for r in 1:R
                S_base[r] = run_one(seeds[r], B)
            end
        end

        threshold = alpha == 0.0 ? maximum(S_base) : Statistics.quantile(S_base, 1 - alpha)
        if symmetric
            S_base_sym = Vector{Float64}(undef, R)
            seeds_sym = rand(rng, UInt64, R)
            if use_distributed
                pool = Distributed.CachingPool(workers_to_use)
                S_base_sym .= Distributed.pmap(seed -> run_one(seed, A), pool, seeds_sym)
            else
                @inbounds for r in 1:R
                    S_base_sym[r] = run_one(seeds_sym[r], A)
                end
            end
            threshold_sym = alpha == 0.0 ? maximum(S_base_sym) : Statistics.quantile(S_base_sym, 1 - alpha)
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
    finally
        if !isempty(added_workers)
            try
                @info "Removing Distributed workers after mahalanobis resampling" removing = length(added_workers)
                Distributed.rmprocs(added_workers...)
                @info "Distributed workers removed" removed = length(added_workers) remaining = Distributed.nworkers()
            catch err
                @info "Failed to remove some Distributed workers" removing = length(added_workers) error = string(err)
            end
        end
    end
end

"""
    scalar_bin_mahalanobis_gap_distinguishability(data::AbstractVector{<:AbstractVector}; num_bins=nothing, regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, num_workers=1)

Bin-pair version: compare every bin to every other bin. Returns a vector of
`(s1, s2, rel_change, M_obs, distinguishable, threshold, z_emp, M_obs_sym, M_obs_min, threshold_sym, threshold_max)`.

# Arguments
- `data`: Input dataset(s) consumed by this method.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `regulator`: Keyword option `regulator` controlling this method's behavior.
- `R`: Numeric control parameter for fitting/sampling resolution.
- `q`: Numeric control parameter for fitting/sampling resolution.
- `alpha`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: Keyword option `symmetric` controlling this method's behavior.
- `num_workers`: Keyword option `num_workers` controlling this method's behavior.

# Returns
- `result`: Output of `scalar_bin_mahalanobis_gap_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function scalar_bin_mahalanobis_gap_distinguishability(
    data::AbstractVector{<:AbstractVector};
    num_bins::Union{Nothing,Int} = nothing,
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    num_workers::Int = 1,
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    progress_step::Union{Nothing,Int} = nothing,
)
    @assert length(data) == 1 "scalar_bin_mahalanobis_gap_distinguishability expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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
    workers_to_use = Int[]
    added_workers = Int[]
    use_distributed = false
    if num_workers > 1 && total > 1
        try
            n_have = Distributed.nworkers()
            if n_have < num_workers
                n_add = num_workers - n_have
                @info "Starting Distributed workers for scalar_bin_mahalanobis_gap_distinguishability" requested = num_workers existing = n_have adding = n_add
                added_workers = Distributed.addprocs(n_add)
                @info "Distributed workers started" added = length(added_workers) total = Distributed.nworkers()
            else
                @info "Using existing Distributed workers for scalar_bin_mahalanobis_gap_distinguishability" requested = num_workers existing = n_have
            end
            workers_to_use = Distributed.workers()[1:min(num_workers, Distributed.nworkers())]
            if !isempty(workers_to_use)
                @info "Using Distributed workers for scalar-bin pair resampling" workers = length(workers_to_use)
                utils_file = @__FILE__
                for w in workers_to_use
                    Distributed.remotecall_wait(w) do
                        if !isdefined(Main, :scalar_bin_mahalanobis_gap_distinguishability)
                            include(utils_file)
                        end
                        nothing
                    end
                end
                use_distributed = true
            end
        catch err
            @info "Falling back to serial scalar-bin pair processing (Distributed setup failed)" error = string(err)
            use_distributed = false
        end
    end

    try
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
                num_workers = 1,
                verbose = verbose,
                rank_tol = rank_tol,
                stabilization_method = stabilization_method,
                projection_tolerance = projection_tolerance,
            )
            return (s1 = s1, s2 = s2, rel_change = relative_change(s1, s2), M_obs = res.M_obs, distinguishable = res.distinguishable, threshold = res.threshold, z_emp = res.z_emp, M_obs_sym = res.M_obs_sym, M_obs_min = res.M_obs_min, threshold_sym = res.threshold_sym, threshold_max = res.threshold_max)
        end

        if use_distributed
            pool = Distributed.CachingPool(workers_to_use)
            if progress && use_pm
                out .= ProgressMeter.progress_pmap(compute_task, pool, 1:total; progress = pm)
            else
                out .= Distributed.pmap(compute_task, pool, 1:total)
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
    finally
        if !isempty(added_workers)
            try
                @info "Removing Distributed workers after scalar-bin pair resampling" removing = length(added_workers)
                Distributed.rmprocs(added_workers...)
                @info "Distributed workers removed" removed = length(added_workers) remaining = Distributed.nworkers()
            catch err
                @info "Failed to remove some Distributed workers" removing = length(added_workers) error = string(err)
            end
        end
    end
end

"""
    scalar_bin_mahalanobis_gap_distinguishability(data::AbstractVector{<:AbstractVector}, ref::AbstractVector; num_bins=nothing, regulator=0.0, R=1000, q=0.0, alpha=0.05, rng=..., symmetric=false, num_workers=1)

See `scalar_bin_mahalanobis_gap_distinguishability(data; ...)`.

Reference variant: compares each scalar bin to fixed reference sample `ref`
instead of comparing all bin pairs.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `ref`: Input parameter `ref` used by this method.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `regulator`: Keyword option `regulator` controlling this method's behavior.
- `R`: Numeric control parameter for fitting/sampling resolution.
- `q`: Numeric control parameter for fitting/sampling resolution.
- `alpha`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: Keyword option `symmetric` controlling this method's behavior.
- `num_workers`: Keyword option `num_workers` controlling this method's behavior.

# Returns
- `result`: Output of `scalar_bin_mahalanobis_gap_distinguishability` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
    num_workers::Int = 1,
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    progress_step::Union{Nothing,Int} = nothing,
)
    @assert length(data) == 1 "scalar_bin_mahalanobis_gap_distinguishability expects one dataset (one path)"
    pairs_raw = data[1]
    @assert !isempty(pairs_raw) "dataset must be non-empty"
    @assert !isempty(ref) "reference set must be non-empty"

    # normalize to (value, scalar) with scalar::Real
    p1 = pairs_raw[1]
    has_scalar_second = p1[2] isa Real
    has_scalar_first = p1[1] isa Real
    @assert has_scalar_second || has_scalar_first "could not find scalar in pair"
    if has_scalar_second
        Tval = typeof(p1[1])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (v, s)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    else
        Tval = typeof(p1[2])
        pairs = Vector{Tuple{Tval,Real}}(undef, length(pairs_raw))
        for (i, (s, v)) in enumerate(pairs_raw)
            @assert s isa Real "scalar must be Real"
            pairs[i] = (v, s)
        end
    end

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
    workers_to_use = Int[]
    added_workers = Int[]
    use_distributed = false
    if num_workers > 1 && total > 1
        try
            n_have = Distributed.nworkers()
            if n_have < num_workers
                n_add = num_workers - n_have
                @info "Starting Distributed workers for scalar_bin_mahalanobis_gap_distinguishability" requested = num_workers existing = n_have adding = n_add
                added_workers = Distributed.addprocs(n_add)
                @info "Distributed workers started" added = length(added_workers) total = Distributed.nworkers()
            else
                @info "Using existing Distributed workers for scalar_bin_mahalanobis_gap_distinguishability" requested = num_workers existing = n_have
            end
            workers_to_use = Distributed.workers()[1:min(num_workers, Distributed.nworkers())]
            if !isempty(workers_to_use)
                @info "Using Distributed workers for scalar-bin reference resampling" workers = length(workers_to_use)
                utils_file = @__FILE__
                for w in workers_to_use
                    Distributed.remotecall_wait(w) do
                        if !isdefined(Main, :scalar_bin_mahalanobis_gap_distinguishability)
                            include(utils_file)
                        end
                        nothing
                    end
                end
                use_distributed = true
            end
        catch err
            @info "Falling back to serial scalar-bin reference processing (Distributed setup failed)" error = string(err)
            use_distributed = false
        end
    end

    try
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
                num_workers = 1,
                verbose = verbose,
                rank_tol = rank_tol,
                stabilization_method = stabilization_method,
                projection_tolerance = projection_tolerance,
            )
            return (scalar = s, M_obs = res.M_obs, distinguishable = res.distinguishable, threshold = res.threshold, z_emp = res.z_emp, M_obs_sym = res.M_obs_sym, M_obs_min = res.M_obs_min, threshold_sym = res.threshold_sym, threshold_max = res.threshold_max)
        end

        if use_distributed
            pool = Distributed.CachingPool(workers_to_use)
            if progress && use_pm
                out .= ProgressMeter.progress_pmap(compute_idx, pool, 1:total; progress = pm)
            else
                out .= Distributed.pmap(compute_idx, pool, 1:total)
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
    finally
        if !isempty(added_workers)
            try
                @info "Removing Distributed workers after scalar-bin reference resampling" removing = length(added_workers)
                Distributed.rmprocs(added_workers...)
                @info "Distributed workers removed" removed = length(added_workers) remaining = Distributed.nworkers()
            catch err
                @info "Failed to remove some Distributed workers" removing = length(added_workers) error = string(err)
            end
        end
    end
end
