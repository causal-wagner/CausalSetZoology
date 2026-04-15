"""
    mc_pairwise_apply(vecs_a, vecs_b, f, num_draws; rng=...)

Apply `f(a, b)` to Monte Carlo sampled pairs `(a, b)` drawn with replacement
from `vecs_a` and `vecs_b`.

# Arguments
- `vecs_a`: First sample set.
- `vecs_b`: Second sample set.
- `f`: Function applied to each sampled pair.
- `num_draws`: Number of Monte Carlo draws.

# Keyword Arguments
- `rng`: Random number generator used for stochastic steps.

# Returns
- `values::Vector{Any}`: One sampled result per draw, in draw order.

# Throws
- `ArgumentError`: If `vecs_a` or `vecs_b` is empty.
- `DomainError`: If `num_draws <= 0`.
- Exceptions thrown while evaluating `f(a, b)` are propagated.
"""
function mc_pairwise_apply(
    vecs_a::AbstractVector,
    vecs_b::AbstractVector,
    f::Function,
    num_draws::Int;
    rng = Random.default_rng(),
)
    isempty(vecs_a) && throw(ArgumentError("vecs_a must be non-empty"))
    isempty(vecs_b) && throw(ArgumentError("vecs_b must be non-empty"))
    if !(num_draws > 0)
        throw(DomainError(num_draws, "num_draws must be positive"))
    end

    vals = Vector{Any}(undef, num_draws)
    draw_seeds = rand(rng, UInt64, num_draws)

    Threads.@threads for t in 1:num_draws
        rng_t = Random.Xoshiro(draw_seeds[t])
        a = vecs_a[rand(rng_t, 1:length(vecs_a))]
        b = vecs_b[rand(rng_t, 1:length(vecs_b))]
        vals[t] = f(a, b)
    end

    return vals
end

"""
    _sample_distinct_unordered_pair(rng, n)

Sample one unordered pair of distinct indices from `1:n`.

# Arguments
- `rng`: Random number generator used for stochastic steps.
- `n`: Number of available items.

# Returns
- `result::Tuple{Int,Int}`: Pair `(i, j)` with `1 <= i < j <= n`.

# Throws
- `DomainError`: If `n < 2`.
"""
@inline function _sample_distinct_unordered_pair(rng, n::Int)::Tuple{Int,Int}
    n >= 2 || throw(DomainError(n, "n must be >= 2"))
    i = rand(rng, 1:n)
    j = rand(rng, 1:(n - 1))
    j = j >= i ? j + 1 : j
    return i < j ? (i, j) : (j, i)
end

"""
    _sample_pooled_pair_distances!(pairs_u, pairs_v, dists, Dmat, rng)

Fill preallocated buffers with sampled pooled-pair distances from a distance matrix.

# Arguments
- `pairs_u`: Output buffer for first indices.
- `pairs_v`: Output buffer for second indices.
- `dists`: Output buffer for sampled distances.
- `Dmat`: Square pairwise-distance matrix.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `nothing`: Buffers are updated in place.

# Throws
- `DimensionMismatch`: If output buffers do not have equal length or if `Dmat`
  is not square.
- `DomainError`: If `size(Dmat, 1) < 2`.
"""
function _sample_pooled_pair_distances!(
    pairs_u::Vector{Int},
    pairs_v::Vector{Int},
    dists::Vector{Float64},
    Dmat::AbstractMatrix{<:Real},
    rng,
)
    size(Dmat, 1) == size(Dmat, 2) || throw(DimensionMismatch("Dmat must be square"))
    n_total = size(Dmat, 1)
    n_total >= 2 || throw(DomainError(n_total, "Dmat must contain at least 2 samples"))
    length(pairs_u) == length(pairs_v) == length(dists) || throw(DimensionMismatch("output buffers must have equal length"))
    @inbounds for k in eachindex(dists)
        i, j = _sample_distinct_unordered_pair(rng, n_total)
        pairs_u[k] = i
        pairs_v[k] = j
        dists[k] = Dmat[i, j]
    end
    return nothing
end

"""
    normalize_hists(
        hists::AbstractVector{<:AbstractVector{<:AbstractDict}};
        normalization::Union{Symbol,Real} = :probability,
    )::Vector{Vector{Dict{Int,Float64}}}

Normalize each histogram independently and return `Float64`-valued dictionaries.

Each histogram is scaled by a denominator chosen from `normalization`.

# Arguments
- `hists`: Nested collection of histograms (`k => count` style dictionaries).
- `normalization`: Per-histogram denominator mode:
  - `:max`: divide by the largest bin value in each histogram.
  - `:probability`: divide by the sum of values in each histogram.
  - any real number: divide all bins by this constant.

# Returns
- `Vector{Vector{Dict{Int,Float64}}}`: Same nested structure with normalized values.

# Notes
- Empty histograms or zero-valued histograms are invalid for symbolic normalization modes.

# Keyword Arguments
- `normalization`: `:max`, `:probability`, or a nonzero finite real constant.

# Throws
- `ArgumentError`: If `normalization` uses an unsupported symbol mode.
- `DomainError`: If numeric normalization is invalid or denominator evaluates to zero."""
function normalize_hists(
    hists::AbstractVector{<:AbstractVector{<:AbstractDict}};
    normalization::Union{Symbol,Real} = :probability,
)::Vector{Vector{Dict{Int,Float64}}}
    if normalization isa Symbol && normalization ∉ (:max, :probability)
        throw(ArgumentError("normalization symbol must be :max or :probability"))
    end
    if normalization isa Real
        if !isfinite(normalization)
            throw(DomainError(normalization, "normalization constant must be finite"))
        end
        if normalization == 0
            throw(DomainError(normalization, "normalization constant must be nonzero"))
        end
    end

    out = Vector{Vector{Dict{Int,Float64}}}(undef, length(hists))

    for i in eachindex(hists)
        out[i] = Vector{Dict{Int,Float64}}(undef, length(hists[i]))
        for (j, hist) in enumerate(hists[i])
            if normalization === :max
                denom = isempty(hist) ? 0.0 : maximum(Base.values(hist))
            elseif normalization === :probability
                denom = isempty(hist) ? 0.0 : sum(Base.values(hist))
            else
                denom = normalization
            end

            if denom == 0.0
                throw(DomainError(denom, "normalization denominator is zero for histogram ($i,$j)"))
            end
            out_hist = Dict{Int,Float64}()
            for (k, v) in hist
                out_hist[k] = v / denom
            end
            out[i][j] = out_hist
        end
    end

    return out
end

"""
    normalize_hists(
        hists::AbstractVector{<:AbstractVector{<:Tuple{<:AbstractDict,<:Real}}};
        normalization::Union{Symbol,Real} = :probability,
        num_bins::Union{Nothing,Int} = nothing,
    )::Vector{Vector{Tuple{Dict{Int,Float64},Real}}}

See the main method `normalize_hists(hists; normalization)` for core behavior.

# Changes in this overload
- Input elements are `(histogram, scalar)` tuples.
- Output elements are `(normalized_histogram, key)` tuples.
- `key` is either the original scalar or, when `num_bins` is set, the center of
  the assigned scalar bin.

# Arguments
- `hists`: Histogram input data.

# Keyword Arguments
- `normalization`: `:max`, `:probability`, or a nonzero finite real constant.
- `num_bins`: Optional scalar bin count (`>= 1`).

# Returns
- `Vector{Vector{Tuple{Dict{Int,Float64},Real}}}`: Normalized histogram/scalar pairs.

# Throws
- `ArgumentError`: If `normalization` uses an unsupported symbol mode.
- `DomainError`: If `num_bins`/normalization are invalid or denominator evaluates to zero."""
function normalize_hists(
    hists::AbstractVector{<:AbstractVector{<:Tuple{<:AbstractDict,<:Real}}};
    normalization::Union{Symbol,Real} = :probability,
    num_bins::Union{Nothing,Int} = nothing,
)::Vector{Vector{Tuple{Dict{Int,Float64},Real}}}
    isempty(hists) && return Vector{Vector{Tuple{Dict{Int,Float64},Real}}}()

    if num_bins !== nothing
        if !(num_bins >= 1)
            throw(DomainError(num_bins, "num_bins must be >= 1"))
        end
    end
    if normalization isa Symbol && normalization ∉ (:max, :probability)
        throw(ArgumentError("normalization symbol must be :max or :probability"))
    end
    if normalization isa Real
        if !isfinite(normalization)
            throw(DomainError(normalization, "normalization constant must be finite"))
        end
        if normalization == 0
            throw(DomainError(normalization, "normalization constant must be nonzero"))
        end
    end

    # build bin edges from all scalars if binning
    bin_edges = nothing
    if num_bins !== nothing
        scalars = [s for group in hists for (_, s) in group]
        centers = sort!(unique(Float64.(scalars)))
        is_prebinned = length(centers) < length(scalars)

        if is_prebinned
            nbins_in = length(centers)
            if num_bins == nbins_in
                # already binned at requested resolution; keep provided centers
                bin_edges = nothing
            elseif num_bins > nbins_in
                throw(DomainError(num_bins, "num_bins=$num_bins exceeds existing binned scalar count $nbins_in"))
            end
        end

        if bin_edges === nothing && (!is_prebinned || num_bins < length(centers))
            vmin, vmax = minimum(scalars), maximum(scalars)
            if vmin == vmax
                bin_edges = [vmin, vmax + 1e-12]
            else
                bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
            end
        elseif bin_edges === nothing && is_prebinned && num_bins == length(centers)
            # pass-through of existing centers
            bin_edges = nothing
        end
    end

    scalar_key(s::Real) = if num_bins === nothing
        s
    elseif bin_edges === nothing
        # pre-binned, same bin count requested
        s
    else
        idx = searchsortedlast(bin_edges, s)
        idx = clamp(idx, 1, length(bin_edges) - 1)
        (bin_edges[idx] + bin_edges[idx + 1]) / 2
    end

    out = Vector{Vector{Tuple{Dict{Int,Float64},Real}}}(undef, length(hists))
    for i in eachindex(hists)
        out[i] = Vector{Tuple{Dict{Int,Float64},Real}}(undef, length(hists[i]))
        for (j, hist_pair) in enumerate(hists[i])
            d, s = hist_pair
            key = scalar_key(s)
            if normalization === :max
                denom = isempty(d) ? 0.0 : maximum(Base.values(d))
            elseif normalization === :probability
                denom = isempty(d) ? 0.0 : sum(Base.values(d))
            else
                denom = normalization
            end
            if denom == 0.0
                throw(DomainError(denom, "normalization denominator is zero for histogram ($i,$j)"))
            end
            out_hist = Dict{Int,Float64}()
            for (k, v) in d
                out_hist[k] = v / denom
            end
            out[i][j] = (out_hist, key)
        end
    end

    return out
end

"""
    normalize_hists(
        hists::AbstractVector{<:AbstractVector};
        normalization::Union{Symbol,Real} = :probability,
        num_bins::Union{Nothing,Int} = nothing,
    )

Bridge method for nested vectors with non-concrete element types (e.g. `Vector{Vector}`).
Infers histogram payload shape at runtime and forwards to the typed implementations.

# Arguments
- `hists`: Nested histogram payloads consisting either of dictionaries or
  `(dictionary, scalar)` tuples.

# Keyword Arguments
- `normalization`: `:max`, `:probability`, or a nonzero finite real constant.
- `num_bins`: Optional scalar bin count for `(hist, scalar)` payloads.

# Returns
- `result`: Output of the matching typed `normalize_hists` method.

# Throws
- `ArgumentError`: If the runtime payload shape is unsupported or internally inconsistent.
- `DomainError`: Propagated from the typed normalization methods for invalid numeric settings.
"""
function normalize_hists(
    hists::AbstractVector{<:AbstractVector};
    normalization::Union{Symbol,Real} = :probability,
    num_bins::Union{Nothing,Int} = nothing,
)
    isempty(hists) && return Vector{Vector{Dict{Int,Float64}}}()

    first_item = nothing
    for group in hists
        isempty(group) && continue
        first_item = first(group)
        break
    end
    first_item === nothing && return Vector{Vector{Dict{Int,Float64}}}()

    if first_item isa AbstractDict
        typed = Vector{Vector{Dict{Int,Float64}}}(undef, length(hists))
        for i in eachindex(hists)
            typed[i] = Vector{Dict{Int,Float64}}(undef, length(hists[i]))
            for (j, hist) in enumerate(hists[i])
                hist isa AbstractDict ||
                    throw(ArgumentError("Expected histogram dict at ($i,$j), got $(typeof(hist))"))
                out_hist = Dict{Int,Float64}()
                for (k, v) in hist
                    out_hist[Int(k)] = Float64(v)
                end
                typed[i][j] = out_hist
            end
        end
        return normalize_hists(typed; normalization = normalization)
    end

    if first_item isa Tuple &&
       length(first_item) == 2 &&
       first_item[1] isa AbstractDict &&
       first_item[2] isa Real
        typed = Vector{Vector{Tuple{Dict{Int,Float64},Float64}}}(undef, length(hists))
        for i in eachindex(hists)
            typed[i] = Vector{Tuple{Dict{Int,Float64},Float64}}(undef, length(hists[i]))
            for (j, item) in enumerate(hists[i])
                (item isa Tuple && length(item) == 2 && item[1] isa AbstractDict && item[2] isa Real) ||
                    throw(ArgumentError("Expected (dict, scalar) tuple at ($i,$j), got $(typeof(item))"))
                d_any, s_any = item
                out_hist = Dict{Int,Float64}()
                for (k, v) in d_any
                    out_hist[Int(k)] = Float64(v)
                end
                typed[i][j] = (out_hist, Float64(s_any))
            end
        end
        return normalize_hists(typed; normalization = normalization, num_bins = num_bins)
    end

    throw(ArgumentError("Unsupported histogram payload type $(typeof(first_item)) for normalize_hists"))
end

"""
    densify_hists(hists::Vector{<:AbstractDict})

Convert sparse histogram dictionaries to a dense matrix with consistent binning.
Returns a matrix of size (Nsamples, nbins).

# Arguments
- `hists`: Histogram input data.

# Returns
- `dense::Matrix{Float64}`: Dense histogram matrix of size `(length(hists), nbins)`.

# Throws
- `ArgumentError`: If `hists` is empty or contains empty dictionaries.
"""
function densify_hists(hists::Vector{<:AbstractDict})
    isempty(hists) && throw(ArgumentError("hists must be non-empty"))
    any(isempty, hists) && throw(ArgumentError("each histogram must be non-empty"))

    min_k = minimum(minimum(keys(h)) for h in hists)
    max_k = maximum(maximum(keys(h)) for h in hists)
    shift = (min_k == 0)

    nbins = shift ? max_k + 1 : max_k
    dense = zeros(Float64, length(hists), nbins)

    for (i, h) in enumerate(hists)
        for (k, v) in h
            idx = shift ? k + 1 : k
            dense[i, idx] = v
        end
    end

    return dense
end

"""
    histogram_to_dense_pair(hist, k::Int)

Convert one observable shaped as `[class_a_samples, class_b_samples]` into two
dense matrices `(A, B)`, one row per sample.

Supported sample representations in both classes:
- histogram dictionaries (`AbstractDict`), aligned via a shared bin basis,
- vectors (`AbstractVector`), aligned by padding to a shared length.

# Arguments
- `hist`: Two-class observable data.
- `k`: Observable index used only in error messages.

# Returns
- `A::Matrix{Float64}`: Dense class-A matrix.
- `B::Matrix{Float64}`: Dense class-B matrix.

# Throws
- `ArgumentError`: If shape/types are invalid or a class is empty.
"""
function histogram_to_dense_pair(hist, k::Int)
    if !(length(hist) == 2)
        throw(ArgumentError("histogram $k must have exactly two classes [A, B]"))
    end
    a = hist[1]
    b = hist[2]
    isempty(a) && throw(ArgumentError("histogram $k class A is empty"))
    isempty(b) && throw(ArgumentError("histogram $k class B is empty"))

    if all(x -> x isa AbstractDict, a) && all(x -> x isa AbstractDict, b)
        n_a = length(a)
        n_b = length(b)
        # Materialize as a concrete Vector{AbstractDict} without attempting to
        # construct the abstract type (which would error for Dict inputs).
        dicts = AbstractDict[x for x in a]
        append!(dicts, AbstractDict[x for x in b])
        dense = densify_hists(dicts)
        return dense[1:n_a, :], dense[n_a+1:n_a+n_b, :]
    end

    if all(x -> x isa AbstractVector, a) && all(x -> x isa AbstractVector, b)
        bad_a = findfirst(v -> any(x -> !(x isa Real), v), a)
        if bad_a !== nothing
            throw(ArgumentError("histogram $k class A sample $bad_a contains non-real values"))
        end
        bad_b = findfirst(v -> any(x -> !(x isa Real), v), b)
        if bad_b !== nothing
            throw(ArgumentError("histogram $k class B sample $bad_b contains non-real values"))
        end
        n_a = length(a)
        n_b = length(b)
        maxlen = maximum(length.(vcat(a, b)))
        pad_to(v, n) = length(v) == n ? Float64.(v) : vcat(Float64.(v), zeros(Float64, n - length(v)))
        A = reduce(vcat, [reshape(pad_to(v, maxlen), 1, :) for v in a])
        B = reduce(vcat, [reshape(pad_to(v, maxlen), 1, :) for v in b])
        return A, B
    end

    throw(ArgumentError("histogram $k must contain either dictionaries or vectors in both classes"))
end

"""
    concatenate_hists(hists...)

Given multiple observables, each shaped as `[class_a_samples, class_b_samples]`,
align each observable across both classes and concatenate all observables
sample-wise.

Returns `(vecs_a, vecs_b)` suitable for `energy_based_histogram_distinguishability`.

# Arguments
- `hists`: One or more observables, each shaped as `[class_a_samples, class_b_samples]`.

# Returns
- `vecs_a::Vector{Vector{Float64}}`: Concatenated class-A samples.
- `vecs_b::Vector{Vector{Float64}}`: Concatenated class-B samples.

# Throws
- `ArgumentError`: If no observables are provided.
- `DimensionMismatch`: If sample counts differ across observables.
"""
function concatenate_hists(hists...)
    if isempty(hists)
        throw(ArgumentError("need at least one histogram"))
    end

    n_a_ref = nothing
    n_b_ref = nothing
    A_blocks = Matrix{Float64}[]
    B_blocks = Matrix{Float64}[]

    for (k, hist) in enumerate(hists)
        A, B = histogram_to_dense_pair(hist, k)
        n_a = size(A, 1)
        n_b = size(B, 1)
        if n_a_ref === nothing
            n_a_ref = n_a
            n_b_ref = n_b
        elseif n_a != n_a_ref || n_b != n_b_ref
            throw(DimensionMismatch("sample counts must match across histograms; histogram $k has (A=$n_a, B=$n_b), expected (A=$n_a_ref, B=$n_b_ref)"))
        end
        push!(A_blocks, A)
        push!(B_blocks, B)
    end

    A_concat = hcat(A_blocks...)
    B_concat = hcat(B_blocks...)
    vecs_a = [Vector{Float64}(A_concat[i, :]) for i in 1:size(A_concat, 1)]
    vecs_b = [Vector{Float64}(B_concat[i, :]) for i in 1:size(B_concat, 1)]
    return vecs_a, vecs_b
end

"""
    join_histograms(hists::Vector{Vector{Vector{Dict}}})::Vector{Vector{Dict}}

Join histograms across the first dimension by summing counts per bin.

Given a nested structure `hists[i][j][k]::Dict`, returns `out[j][k]` where
all dictionaries along `i` have been added bin-wise.

# Arguments
- `hists`: Histogram input data.

# Returns
- `result::Vector{Vector{Dict}}`: Output of `join_histograms` with type annotation `Vector{Vector{Dict}}`.

# Throws
- `DimensionMismatch`: Raised when nested histogram container dimensions are inconsistent.
"""
function join_histograms(
    hists::AbstractVector{<:AbstractVector{<:AbstractVector{<:AbstractDict}}},
)::Vector{Vector{Dict{Int,Int}}}
    isempty(hists) && return Vector{Vector{Dict{Int,Int}}}()

    n_outer = length(hists)
    n_groups = length(hists[1])
    n_hists = length(hists[1][1])

    for i in 1:n_outer
        if !(length(hists[i]) == n_groups)
            throw(DimensionMismatch("length(hists[$i])=$(length(hists[i])) must equal n_groups=$n_groups"))
        end
        for j in 1:n_groups
            if !(length(hists[i][j]) == n_hists)
                throw(DimensionMismatch("length(hists[$i][$j])=$(length(hists[i][j])) must equal n_hists=$n_hists"))
            end
        end
    end

    out = [[Dict{Int,Int}() for _ in 1:n_hists] for _ in 1:n_groups]

    for i in 1:n_outer
        for j in 1:n_groups
            for k in 1:n_hists
                d = hists[i][j][k]
                outd = out[j][k]
                for (bin, count) in d
                    outd[bin] = get(outd, bin, 0) + count
                end
            end
        end
    end

    return out
end

"""
    join_histograms(hists::Vector{Vector{Vector{Tuple{Dict,Real}}}})

See the main method `join_histograms(hists::Vector{Vector{Vector{Dict}}})` for
the aggregation logic.

# Changes in this overload
- Input histograms are `(hist, scalar)` tuples.
- Histogram counts are summed exactly as in the main method.
- Scalar labels are preserved per `(j, k)` index and required to match across
  the joined outer dimension.

# Arguments
- `hists`: Histogram input data.

# Returns
- `Vector{Vector{Tuple{Dict,Float64}}}`: Joined histogram/scalar tuples with validated scalar alignment.

# Throws
- `DimensionMismatch`: Raised when nested histogram container dimensions are inconsistent.
- `DomainError`: Raised when scalar labels for the same `(j, k)` index do not match across inputs.
"""
function join_histograms(
    hists::AbstractVector{<:AbstractVector{<:AbstractVector{<:Tuple{<:AbstractDict,S}}}},
)::Vector{Vector{Tuple{Dict{Int,Float64},Float64}}} where {S<:Real}
    isempty(hists) && return Vector{Vector{Tuple{Dict{Int,Float64},Float64}}}()

    n_outer = length(hists)
    n_mid = length(hists[1])
    n_hists = length(hists[1][1])

    for i in 1:n_outer
        if !(length(hists[i]) == n_mid)
            throw(DimensionMismatch("length(hists[$i])=$(length(hists[i])) must equal n_mid=$n_mid"))
        end
        for j in 1:n_mid
            if !(length(hists[i][j]) == n_hists)
                throw(DimensionMismatch("length(hists[$i][$j])=$(length(hists[i][j])) must equal n_hists=$n_hists"))
            end
        end
    end

    out = Vector{Vector{Tuple{Dict{Int,Float64},Float64}}}(undef, n_mid)
    for j in 1:n_mid
        out[j] = Vector{Tuple{Dict{Int,Float64},Float64}}(undef, n_hists)
        for k in 1:n_hists
            d_sum = Dict{Int,Float64}()
            s_ref = Float64(hists[1][j][k][2])
            for i in 1:n_outer
                d, s = hists[i][j][k]
                s64 = Float64(s)
                if !(s64 == s_ref)
                    throw(DomainError(s, "scalar mismatch for histogram index ($j,$k): expected $s_ref"))
                end
                for (bin, count) in d
                    d_sum[bin] = get(d_sum, bin, 0.0) + count
                end
            end
            out[j][k] = (d_sum, s_ref)
        end
    end

    return out
end

"""
    average_histogram_with_std(hists::AbstractVector{<:AbstractDict})

Compute per-bin mean and standard deviation across sparse histograms.

Input histograms are first densified with `densify_hists`, then statistics are
computed column-wise.

# Bin indexing convention
- If any histogram contains bin `k = 0`, bins are treated as 0-based and shifted.
- Otherwise, bins are treated as already 1-based and are not shifted.
- Missing bins are treated as zero.

# Returns
- `mean_vec::Vector{Float64}`: Mean value of each bin.
- `std_vec::Vector{Float64}`: Population standard deviation of each bin.

# Notes
- For empty input, returns `(Float64[], Float64[])`.

# Arguments
- `hists`: Histogram input data.

# Keyword Arguments
- None.

# Throws
- `ArgumentError`: Propagated from `densify_hists` for invalid histogram input."""
function average_histogram_with_std(
    hists::AbstractVector{<:AbstractDict},
)::Tuple{Vector{Float64},Vector{Float64}}
    isempty(hists) && return Float64[], Float64[]
    X = densify_hists(hists)
    # X: (Nsamples, nbins)
    mean_vec = vec(Statistics.mean(X; dims=1))
    # Population std (numerically safe)
    std_vec = sqrt.(max.(vec(Statistics.mean(X.^2; dims=1)) .- mean_vec.^2, 0.0))
    return mean_vec, std_vec
end

"""
    average_vectors_with_std(vs::AbstractVector)

Compute element-wise mean and standard deviation over vector samples.

Supports:
- Flat input: `vs = [v1, v2, ...]` where all vectors have equal length.
- Nested input: each sample is a vector of vectors; nested vectors are flattened
  by concatenation before computing statistics.

# Arguments
- `vs`: Collection of vector samples (flat or nested).

# Returns
- `(mean_vec, std_vec)` with one entry per index.

# Notes
- Uses population standard deviation.
- Empty input returns `(Float64[], Float64[])`.

# Keyword Arguments
- None.

# Throws
- `ArgumentError`: Raised when vector shape constraints are violated."""
function average_vectors_with_std(
    vs::AbstractVector,
)::Tuple{Vector{Float64},Vector{Float64}}
    isempty(vs) && return Float64[], Float64[]
    if !(all(v -> v isa AbstractVector, vs))
        throw(ArgumentError("all entries must be vectors"))
    end
    # Detect nested vectors: Vector{Vector{Float64}} per sample
    if vs[1] isa AbstractVector && !isempty(vs[1]) && vs[1][1] isa AbstractVector
        nested = vs
        if !(all(v -> v isa AbstractVector, nested))
            throw(ArgumentError("nested entries must be vectors"))
        end
        n = length(nested[1][1])
        for v in nested
            if !(all(w -> w isa AbstractVector, v))
                throw(ArgumentError("nested entries must be vectors"))
            end
            for w in v
                if !(length(w) == n)
                    throw(ArgumentError("all nested vectors must have the same length"))
                end
            end
        end
        # flatten nested samples by concatenation, then compute mean/std per index
        flat = Vector{Vector{Float64}}(undef, 0)
        for v in nested
            for w in v
                push!(flat, Float64.(w))
            end
        end
        vs = flat
    end

    n = length(vs[1])
    for v in vs
        if !(length(v) == n)
            throw(ArgumentError("all vectors must have the same length"))
        end
    end
    X = Matrix{Float64}(undef, length(vs), n)
    for (i, v) in enumerate(vs)
        @inbounds for j in 1:n
            X[i, j] = Float64(v[j])
        end
    end
    mean_vec = vec(Statistics.mean(X; dims=1))
    std_vec = sqrt.(max.(vec(Statistics.mean(X.^2; dims=1)) .- mean_vec.^2, 0.0))
    return mean_vec, std_vec
end

"""
    average_vectors_with_std(
        vs::AbstractVector{<:Tuple{<:AbstractVector,<:Real}};
        num_bins::Union{Nothing,Int} = nothing,
    )::Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}

See the main method `average_vectors_with_std(vs)` for the mean/std computation.

# Changes in this overload
- Input samples are `(vector, scalar)` tuples.
- Samples are grouped by scalar before aggregation.
- If `num_bins` is provided, scalars are binned and group keys become bin centers.
- Returns one `(key, mean_vec, std_vec)` per scalar group.

# Arguments
- `vs`: Vector/scalar sample tuples.

# Keyword Arguments
- `num_bins`: Optional scalar bin count (`>= 1`).

# Returns
- `Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}`: Group key, mean vector, and std vector per group.

# Throws
- `ArgumentError`: Propagated from vector-shape validation during aggregation.
- `DomainError`: If `num_bins` is invalid."""
function average_vectors_with_std(
    vs::AbstractVector{<:Tuple{<:AbstractVector,<:Real}};
    num_bins::Union{Nothing,Int} = nothing,
)::Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}
    isempty(vs) && return Tuple{Real,Vector{Float64},Vector{Float64}}[]

    if num_bins !== nothing
        if !(num_bins >= 1)
            throw(DomainError(num_bins, "num_bins must be >= 1"))
        end
    end

    bin_edges = nothing
    if num_bins !== nothing
        scalars = [s for (_, s) in vs]
        vmin, vmax = minimum(scalars), maximum(scalars)
        if vmin == vmax
            bin_edges = [vmin, vmax + 1e-12]
        else
            bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
        end
    end

    groups = Dict{Real,Vector{AbstractVector}}()
    for (v, s) in vs
        key = if bin_edges === nothing
            s
        else
            idx = searchsortedlast(bin_edges, s)
            idx = clamp(idx, 1, length(bin_edges) - 1)
            (bin_edges[idx] + bin_edges[idx + 1]) / 2
        end
        get!(groups, key, AbstractVector[])
        push!(groups[key], v)
    end

    keys_sorted = sort(collect(keys(groups)))
    out = Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}(undef, length(keys_sorted))
    for (i, k) in enumerate(keys_sorted)
        mean_vec, std_vec = average_vectors_with_std(groups[k])
        out[i] = (k, mean_vec, std_vec)
    end
    return out
end

"""
    average_histogram_with_std(
        hists::AbstractVector{<:Tuple{<:AbstractDict,<:Real}};
        num_bins::Union{Nothing,Int} = nothing,
    )::Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}

See the main method `average_histogram_with_std(hists)` for per-bin statistics.

# Changes in this overload
- Input elements are `(histogram, scalar)` tuples.
- Histograms are grouped by scalar before aggregation.
- If `num_bins` is provided, scalars are binned and group keys become bin centers.
- Returns one `(key, mean_vec, std_vec)` per scalar group.

# Arguments
- `hists`: Histogram input data.

# Keyword Arguments
- `num_bins`: Optional scalar bin count (`>= 1`).

# Returns
- `Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}`: Group key, mean vector, and std vector per group.

# Throws
- `ArgumentError`: Propagated from histogram-shape validation during aggregation.
- `DomainError`: If `num_bins` is invalid."""
function average_histogram_with_std(
    hists::AbstractVector{<:Tuple{<:AbstractDict,<:Real}};
    num_bins::Union{Nothing,Int} = nothing,
)::Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}
    isempty(hists) && return Tuple{Real,Vector{Float64},Vector{Float64}}[]

    if num_bins !== nothing
        if !(num_bins >= 1)
            throw(DomainError(num_bins, "num_bins must be >= 1"))
        end
    end

    # optional binning of scalar values
    bin_edges = nothing
    if num_bins !== nothing
        values = [v for (_, v) in hists]
        vmin, vmax = minimum(values), maximum(values)
        if vmin == vmax
            bin_edges = [vmin, vmax + 1e-12]
        else
            bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
        end
    end

    groups = Dict{Real,Vector{Dict}}()
    for (h, v) in hists
        if bin_edges !== nothing
            # assign to evenly spaced bin center
            idx = searchsortedlast(bin_edges, v)
            idx = clamp(idx, 1, length(bin_edges) - 1)
            v = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        end
        get!(groups, v, Dict[])
        push!(groups[v], h)
    end

    keys_sorted = sort(collect(keys(groups)))
    out = Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}(undef, length(keys_sorted))
    for (i, v) in enumerate(keys_sorted)
        mean_vec, std_vec = average_histogram_with_std(groups[v])
        out[i] = (v, mean_vec, std_vec)
    end
    return out
end

"""
    replace_zeros(σ::AbstractVector{<:Real}; ϵ::Real = 1e-3)

Replace zero entries with a small positive floor derived from the data.

The replacement value is `minimum(σ[σ .> 0]) * ϵ`. Non-zero entries remain
unchanged.

# Arguments
- `σ`: Input vector, typically a standard-deviation-like quantity.
- `ϵ`: Relative floor factor multiplied by the smallest positive entry.

# Returns
- A copy of `σ` where exact zeros are replaced.

# Notes
- If no positive entries exist, the original vector is returned unchanged.

# Keyword Arguments
- `ϵ`: Relative floor factor (`> 0`).

# Throws
- `DomainError`: If `ϵ` is not strictly positive."""
function replace_zeros(σ::AbstractVector{<:Real}; ϵ::Real=1e-3)
    if !(ϵ > 0)
        throw(DomainError(ϵ, "ϵ must be > 0"))
    end
    nz = σ[σ .> 0]
    isempty(nz) && return σ
    epsσ = minimum(nz) * ϵ
    σ_new = copy(σ)
    @inbounds for i in eachindex(σ)
        if σ[i] == 0
            σ_new[i] = epsσ
        end
    end
    return σ_new
end
