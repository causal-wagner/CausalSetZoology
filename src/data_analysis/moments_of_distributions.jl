@inline function _weighted_hist_inputs(
    hist::AbstractDict,
    bins::Union{Nothing,AbstractVector{<:Real}},
)::Tuple{Vector, Vector{Float64}, Vector{Float64}}
    ks = sort!(collect(keys(hist)))
    ws = Float64[hist[k] for k in ks]

    any(!isfinite, ws) && throw(DomainError(ws, "histogram weights must be finite"))
    any(<(0.0), ws) && throw(DomainError(ws, "histogram weights must be nonnegative"))

    if bins === nothing
        xs = Float64.(ks)
    else
        length(bins) == length(ks) || throw(DimensionMismatch("bins must have the same length as the histogram support"))
        xs = Float64.(bins)
        any(!isfinite, xs) && throw(DomainError(xs, "bins must contain only finite values"))
    end

    return ks, xs, ws
end

"""
    weighted_hist_mean(hist; bins=nothing)

Compute the probability-weighted mean of a histogram.

# Arguments
- `hist`: Histogram mapping support values to nonnegative weights.

# Keyword Arguments
- `bins`: Optional numeric bin locations corresponding to the sorted histogram support.

# Returns
- `mu::Float64`: Weighted mean of the normalized histogram weights. Returns `NaN`
  when the total histogram weight is zero.

# Throws
- `DimensionMismatch`: If `bins` is provided and its length does not match the
  histogram support size.
- `DomainError`: If histogram weights are negative or non-finite, or if `bins`
  contains non-finite values.
"""
function weighted_hist_mean(
    hist::AbstractDict;
    bins::Union{Nothing,AbstractVector{<:Real}} = nothing,
)::Float64
    _, xs, ws = _weighted_hist_inputs(hist, bins)
    s = sum(ws)
    s <= 0.0 && return NaN
    p = ws ./ s
    return sum(p .* xs)
end

"""
    weighted_hist_skew(hist; bins=nothing)

Compute the probability-weighted skewness of a histogram.

# Arguments
- `hist`: Histogram mapping support values to nonnegative weights.

# Keyword Arguments
- `bins`: Optional numeric bin locations corresponding to the sorted histogram support.

# Returns
- `gamma1::Float64`: Weighted skewness of the normalized histogram weights.
  Returns `NaN` when the total histogram weight is zero or when the variance is zero.

# Throws
- `DimensionMismatch`: If `bins` is provided and its length does not match the
  histogram support size.
- `DomainError`: If histogram weights are negative or non-finite, or if `bins`
  contains non-finite values.
"""
function weighted_hist_skew(
    hist::AbstractDict;
    bins::Union{Nothing,AbstractVector{<:Real}} = nothing,
)::Float64
    _, xs, ws = _weighted_hist_inputs(hist, bins)
    s = sum(ws)
    s <= 0.0 && return NaN
    p = ws ./ s
    mu = sum(p .* xs)
    c = xs .- mu
    m2 = sum(p .* c .^ 2)
    m2 <= 0.0 && return NaN
    m3 = sum(p .* c .^ 3)
    return m3 / m2^(3 / 2)
end

"""
    weighted_hist_exkurt(hist; bins=nothing)

Compute the probability-weighted excess kurtosis of a histogram.

# Arguments
- `hist`: Histogram mapping support values to nonnegative weights.

# Keyword Arguments
- `bins`: Optional numeric bin locations corresponding to the sorted histogram support.

# Returns
- `gamma2::Float64`: Weighted excess kurtosis of the normalized histogram weights.
  Returns `NaN` when the total histogram weight is zero or when the variance is zero.

# Throws
- `DimensionMismatch`: If `bins` is provided and its length does not match the
  histogram support size.
- `DomainError`: If histogram weights are negative or non-finite, or if `bins`
  contains non-finite values.
"""
function weighted_hist_exkurt(
    hist::AbstractDict;
    bins::Union{Nothing,AbstractVector{<:Real}} = nothing,
)::Float64
    _, xs, ws = _weighted_hist_inputs(hist, bins)
    s = sum(ws)
    s <= 0.0 && return NaN
    p = ws ./ s
    mu = sum(p .* xs)
    c = xs .- mu
    m2 = sum(p .* c .^ 2)
    m2 <= 0.0 && return NaN
    m4 = sum(p .* c .^ 4)
    return m4 / m2^2 - 3.0
end

"""
    aggregate_hist_moment(hists, f)

Apply a histogram moment function to multiple histograms and aggregate the results.

# Arguments
- `hists`: Collection of histograms.
- `f`: Function mapping one histogram to one scalar moment value.

# Returns
- `result`: Named tuple `(mean, std, values)` where `values` contains the finite
  moment values returned by `f`, `mean` is their arithmetic mean, and `std` is
  their population standard deviation.

# Throws
- `ArgumentError`: If `hists` is empty or if `f` produces no finite values.
"""
function aggregate_hist_moment(hists, f)
    isempty(hists) && throw(ArgumentError("hists must be non-empty"))
    vals = [f(h) for h in hists]
    vals = filter(isfinite, vals)
    isempty(vals) && throw(ArgumentError("f(h) must yield at least one finite value"))
    return (
        mean = Statistics.mean(vals),
        std = Statistics.std(vals; corrected = false),
        values = vals,
    )
end
