"""
    get_size(path::AbstractString)::Int

Read the configured causal set size from a dataset file.

The function opens `path` as a JLD2 file and returns `f["meta/config"]["cset_size"]`.

# Arguments
- `path`: Path to a JLD2 dataset file that stores `meta/config`.

# Returns
- `Int`: The configured causal set size.

# Keyword Arguments
- This method has no keyword arguments.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function get_size(path::AbstractString)::Int
    return JLD2.jldopen(path, "r") do f
        config = f["meta/config"]
        return config["cset_size"]
    end
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
- Empty histograms produce a zero denominator for symbolic modes and trigger the
  assertion `denom != 0.0`.

# Keyword Arguments
- `normalization`: Keyword option `normalization` controlling this method's behavior.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function normalize_hists(
    hists::AbstractVector{<:AbstractVector{<:AbstractDict}};
    normalization::Union{Symbol,Real} = :probability,
)::Vector{Vector{Dict{Int,Float64}}}
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

            @assert denom != 0.0
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
- `normalization`: Keyword option `normalization` controlling this method's behavior.
- `num_bins`: Bin selection or binning control parameter.

# Returns
- `result::Vector{Vector{Tuple{Dict{Int,Float64},Real}}}`: Output of `normalize_hists` with type annotation `Vector{Vector{Tuple{Dict{Int,Float64},Real}}}`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function normalize_hists(
    hists::AbstractVector{<:AbstractVector{<:Tuple{<:AbstractDict,<:Real}}};
    normalization::Union{Symbol,Real} = :probability,
    num_bins::Union{Nothing,Int} = nothing,
)::Vector{Vector{Tuple{Dict{Int,Float64},Real}}}
    isempty(hists) && return Vector{Vector{Tuple{Dict{Int,Float64},Real}}}()

    if num_bins !== nothing
        @assert num_bins ≥ 1 "num_bins must be >= 1"
    end

    # build bin edges from all scalars if binning
    bin_edges = nothing
    if num_bins !== nothing
        scalars = [s for group in hists for (_, s) in group]
        vmin, vmax = minimum(scalars), maximum(scalars)
        if vmin == vmax
            bin_edges = [vmin, vmax + 1e-12]
        else
            bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
        end
    end

    scalar_key(s::Real) = if bin_edges === nothing
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
            @assert denom != 0.0
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
- This method has no keyword arguments.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
- This method has no keyword arguments.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function average_vectors_with_std(
    vs::AbstractVector,
)::Tuple{Vector{Float64},Vector{Float64}}
    isempty(vs) && return Float64[], Float64[]
    @assert all(v -> v isa AbstractVector, vs) "all entries must be vectors"

    # Detect nested vectors: Vector{Vector{Float64}} per sample
    if vs[1] isa AbstractVector && !isempty(vs[1]) && vs[1][1] isa AbstractVector
        nested = vs
        @assert all(v -> v isa AbstractVector, nested) "nested entries must be vectors"
        n = length(nested[1][1])
        for v in nested
            @assert all(w -> w isa AbstractVector, v) "nested entries must be vectors"
            for w in v
                @assert length(w) == n "all nested vectors must have the same length"
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
        @assert length(v) == n "all vectors must have the same length"
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
- `vs`: Input parameter `vs` used by this method.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.

# Returns
- `result::Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}`: Output of `average_vectors_with_std` with type annotation `Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function average_vectors_with_std(
    vs::AbstractVector{<:Tuple{<:AbstractVector,<:Real}};
    num_bins::Union{Nothing,Int} = nothing,
)::Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}
    isempty(vs) && return Tuple{Real,Vector{Float64},Vector{Float64}}[]

    if num_bins !== nothing
        @assert num_bins ≥ 1 "num_bins must be >= 1"
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
- `num_bins`: Bin selection or binning control parameter.

# Returns
- `result::Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}`: Output of `average_histogram_with_std` with type annotation `Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function average_histogram_with_std(
    hists::AbstractVector{<:Tuple{<:AbstractDict,<:Real}};
    num_bins::Union{Nothing,Int} = nothing,
)::Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}
    isempty(hists) && return Tuple{Real,Vector{Float64},Vector{Float64}}[]

    if num_bins !== nothing
        @assert num_bins ≥ 1 "num_bins must be >= 1"
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
        @assert v isa Real
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
- This method has no keyword arguments.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function replace_zeros(σ::AbstractVector{<:Real}; ϵ::Real=1e-3)
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

"""
    abundance_shift(hist::Dict{Int,Int})::Dict{Int,Int}

Shift abundance histogram keys from inclusive interval indexing to shifted form.

For every key `k > 1`, the output contains `k - 2 => hist[k]`.
Keys `k <= 1` are dropped.

# Returns
- New `Dict{Int,Int}` with shifted keys.

# Arguments
- `hist`: Histogram input data.

# Keyword Arguments
- This method has no keyword arguments.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function abundance_shift(hist::Dict{Int,Int};)
    out = Dict{Int,Int}()
    for k in keys(hist)
        if k > 1
            out[k-2] = hist[k]
        end
    end
    return out
end
