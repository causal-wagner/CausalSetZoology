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
    hists::Vector{Vector{Vector{Dict}}},
)::Vector{Vector{Dict}}
    isempty(hists) && return Vector{Vector{Dict}}()

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
    hists::Vector{Vector{Vector{Tuple{Dict,S}}}},
)::Vector{Vector{Tuple{Dict,Float64}}} where {S<:Real}
    isempty(hists) && return Vector{Vector{Tuple{Dict,Float64}}}()

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

    out = Vector{Vector{Tuple{Dict,Float64}}}(undef, n_mid)
    for j in 1:n_mid
        out[j] = Vector{Tuple{Dict,Float64}}(undef, n_hists)
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
