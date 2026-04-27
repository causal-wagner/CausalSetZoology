"""
    _RunningMoments

Internal mutable accumulator for online mean/std computation.

# Arguments
- `n`: Number of observed samples.
- `sum`: Sum of sample values.
- `sumsq`: Sum of squared sample values.
"""
mutable struct _RunningMoments
    n::Int
    sum::Float64
    sumsq::Float64
end

_RunningMoments() = _RunningMoments(0, 0.0, 0.0)

"""
    _push_moment!(m::_RunningMoments, x::Float64)

Update a running-moments accumulator with one new observation.

# Arguments
- `m`: Mutable running-moments state.
- `x`: New scalar sample value.

# Returns
- `nothing`: The accumulator `m` is updated in place.
"""
function _push_moment!(m::_RunningMoments, x::Float64)
    m.n += 1
    m.sum += x
    m.sumsq += x * x
    return nothing
end

"""
    _mean_std(m::_RunningMoments)::Tuple{Float64,Float64}

Finalize running moments into `(mean, std)` using sample standard deviation.

# Arguments
- `m`: Running-moments state.

# Returns
- `stats::Tuple{Float64,Float64}`: `(mean, std)` for accumulated samples.
"""
function _mean_std(m::_RunningMoments)::Tuple{Float64,Float64}
    n = m.n
    n == 0 && return (NaN, NaN)
    mean_val = m.sum / n
    n == 1 && return (mean_val, 0.0)
    var = (m.sumsq - n * mean_val * mean_val) / (n - 1)
    return (mean_val, sqrt(max(var, 0.0)))
end

"""
    _ScanConfig

Internal scan configuration used by shared record iterators.

# Arguments
- `step`: Positive thinning step.
- `offset`: Phase offset for modulo-based sample selection.
"""
struct _ScanConfig
    step::Int
    offset::Int
end

"""
    _scan_config(thinning::Int)

Normalize integer thinning into an internal scan configuration.

# Arguments
- `thinning`: Keep every `thinning`-th filtered sample.

# Returns
- `cfg::_ScanConfig`: Scan configuration with integer-step semantics.

# Throws
- `DomainError`: Raised when `thinning < 1`."""
function _scan_config(thinning::Int)
    if !(thinning >= 1)
        throw(DomainError(thinning, "thinning must be >= 1"))
    end
    return _ScanConfig(thinning, 0)
end

"""
    _scan_config(thinning::Float64)

Normalize fractional thinning into an internal scan configuration.

# Arguments
- `thinning`: Fractional keep-rate in `(0, 1]`.

# Returns
- `cfg::_ScanConfig`: Scan configuration equivalent to the legacy fractional thinning behavior.

# Throws
- `DomainError`: Raised when `thinning` is outside `(0, 1]`."""
function _scan_config(thinning::Float64)
    if !(0.0 < thinning <= 1.0)
        throw(DomainError(thinning, "thinning must be in (0, 1]"))
    end
    step = max(1, round(Int, 1.0 / thinning))
    return _ScanConfig(step, 1)
end

"""
    _keep_sample(seen::Int, cfg::_ScanConfig)

Check whether the current filtered sample index should be kept.

# Arguments
- `seen`: One-based count of filtered records seen so far.
- `cfg`: Internal scan configuration.

# Returns
- `keep::Bool`: `true` when the sample is selected by the thinning rule.
"""
_keep_sample(seen::Int, cfg::_ScanConfig) = ((seen - cfg.offset) % cfg.step) == 0

"""
    _normalize_filters(paths::Vector{<:AbstractString}, filters)

Normalize optional filter inputs to a path-aligned vector.

# Arguments
- `paths`: Input data paths.
- `filters`: `nothing` or a vector of per-path filters.

# Returns
- `filters_norm`: A filter vector of length `length(paths)`.

# Throws
- `DimensionMismatch`: Raised when provided `filters` has a different length than `paths`."""
function _normalize_filters(paths::Vector{<:AbstractString}, filters)
    n = length(paths)
    if filters === nothing
        return fill(nothing, n)
    end
    if !(length(filters) == n)
        throw(DimensionMismatch("length(filters)=$(length(filters)) must equal number of paths n=$n"))
    end
    return filters
end

"""
    _scan_records(visit!::Function, path::AbstractString, filter, cfg::_ScanConfig)

Iterate records from one statistics file with shared filtering and thinning logic.

# Arguments
- `visit!`: Callback executed for each kept record.
- `path`: Input `statistics.jld2` file path.
- `filter`: Per-record predicate or `nothing`.
- `cfg`: Internal scan configuration.

# Returns
- `nothing`: Side effects are produced via `visit!`.

# Throws
- Propagates I/O and callback errors from JLD2 access, filtering, and `visit!` execution."""
function _scan_records(visit!::F, path::AbstractString, filter, cfg::_ScanConfig) where {F<:Function}
    seen = 0
    JLD2.jldopen(path, "r") do f
        nbatches = f["meta/nbatches"]
        if filter === nothing
            for b in 1:nbatches
                batch = f["batches/$b"]
                for x in batch
                    seen += 1
                    _keep_sample(seen, cfg) || continue
                    visit!(x)
                end
            end
        else
            for b in 1:nbatches
                batch = f["batches/$b"]
                for x in batch
                    !filter(x) && continue
                    seen += 1
                    _keep_sample(seen, cfg) || continue
                    visit!(x)
                end
            end
        end
    end
    return nothing
end

"""
    _extract_field_value(x, field::Symbol)

Extract a direct field value from one record.

# Arguments
- `x`: Input record.
- `field`: Symbolic field name.

# Returns
- `value`: Field value read via `getfield`.

# Throws
- Propagates errors from `getfield`."""
function _extract_field_value(x, field::Symbol)
    return getfield(x, field)
end

"""
    _extract_field_value(x, field::Tuple{Symbol,Int64})

Extract one histogram bin value from a record.

# Arguments
- `x`: Input record.
- `field`: `(histogram_symbol, bin)` specification.

# Returns
- `value`: Histogram bin value, defaulting to `0` for missing bins.

# Throws
- Propagates errors from `getfield` and histogram access."""
function _extract_field_value(x, field::Tuple{Symbol,Int64})
    sym, bin = field
    hist = getfield(x, sym)
    return get(hist, bin, 0)
end

"""
    _push_auto_typed!(vals::Vector{Any}, j::Int, value)

Append a value to column `j`, initializing its concrete element vector lazily.

# Arguments
- `vals`: Per-column storage vector.
- `j`: Target column index.
- `value`: Value to append.

# Returns
- `nothing`: `vals` is mutated in place.

# Throws
- Propagates bounds errors for invalid column indices."""
@inline function _push_auto_typed!(vals::Vector{Any}, j::Int, value)
    if vals[j] === nothing
        vals[j] = Vector{typeof(value)}()
    end
    push!(vals[j], value)
    return nothing
end

"""
    _finalize_any_columns(vals::Vector{Any}, nfields::Int)

Convert internal lazy column buffers into the public `Vector{Any}` column layout.

# Arguments
- `vals`: Internal per-column buffers, possibly containing `nothing`.
- `nfields`: Number of output columns.

# Returns
- `out::Vector{Any}`: Finalized column vectors, replacing missing buffers with `Any[]`.

# Throws
- Propagates bounds errors if `nfields` and `vals` are inconsistent."""
function _finalize_any_columns(vals::Vector{Any}, nfields::Int)
    out = Vector{Any}(undef, nfields)
    for j in 1:nfields
        out[j] = vals[j] === nothing ? Any[] : vals[j]
    end
    return out
end

"""
    load_and_average_std_scalar(
        data_paths::Vector{String},
        fields::Vector{Symbol},
        scalar::Symbol;
        num_bins::Union{Nothing,Int} = nothing,
        verbose::Bool = false,
    )

Load fields and an ordering scalar, then group by scalar (optionally binned)
before computing mean/std per field.

Returns, for each dataset, a vector of `(scalar_value, stats)` pairs where
`stats` is a vector of `(mean, std)` for each field in `fields`, ordered by
`scalar_value`.

# Arguments
- `data_paths`: Path or collection of paths used for loading/saving data.
- `fields`: Observable/field names to extract, plot, or process.
- `scalar`: Scalar value(s) or scalar field identifier.

# Keyword Arguments
- `num_bins`: Bin selection or binning control parameter.
- `verbose`: Boolean toggle controlling output or execution behavior.

# Returns
- `result`: Output of `load_and_average_std_scalar` as described in the summary above.

# Throws
- `ArgumentError`: Raised when no field data could be loaded.
- `DimensionMismatch`: Propagated when delegated loading receives mismatched `paths`/`filters` lengths.
- `DomainError`: Raised for invalid `num_bins`, or propagated for invalid delegated thinning settings.
- `TypeError`: Propagated when delegated scalar fields are not `Real`."""
function load_and_average_std_scalar(
    data_paths::Vector{String},
    fields::Vector{Symbol},
    scalar::Symbol;
    num_bins::Union{Nothing,Int} = nothing,
    verbose::Bool = false,
)
    loaded = load_fields_from_paths(data_paths, fields, scalar; verbose = verbose)

    out = Vector{Vector{Tuple{Float64,Vector{Tuple{Float64,Float64}}}}}(undef, length(loaded))

    for (i, dataset) in enumerate(loaded)
        # dataset[j] is Vector{Tuple{value, scalar}}
        if !(!isempty(dataset))
            throw(ArgumentError("no fields loaded"))
        end
        scalars = Float64[s for (_, s) in dataset[1]]

        bin_edges::Union{Nothing,Vector{Float64}} = nothing
        if num_bins !== nothing
            if !(num_bins ≥ 1)
                throw(DomainError(num_bins, "num_bins must be >= 1"))
            end
            vmin, vmax = minimum(scalars), maximum(scalars)
            if vmin == vmax
                bin_edges = [vmin, vmax + 1e-12]
            else
                bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
            end
        end

        nfields = length(dataset)
        groups = Dict{Float64,Vector{_RunningMoments}}() # scalar => per-field running moments
        for j in eachindex(dataset)
            for (v_raw, s_raw) in dataset[j]
                s = Float64(s_raw)
                if bin_edges !== nothing
                    idx = searchsortedlast(bin_edges, s)
                    idx = clamp(idx, 1, length(bin_edges) - 1)
                    s = (bin_edges[idx] + bin_edges[idx + 1]) / 2
                end
                per_field = get!(groups, s) do
                    [_RunningMoments() for _ in 1:nfields]
                end
                _push_moment!(per_field[j], Float64(v_raw))
            end
        end

        scalars_sorted = sort!(collect(keys(groups)))
        out[i] = Vector{Tuple{Float64,Vector{Tuple{Float64,Float64}}}}(undef, length(scalars_sorted))
        for (k, s) in enumerate(scalars_sorted)
            stats = Vector{Tuple{Float64,Float64}}(undef, nfields)
            for j in 1:nfields
                stats[j] = _mean_std(groups[s][j])
            end
            out[i][k] = (s, stats)
        end
    end

    return out
end

"""
    load_histograms_from_paths(
        paths::Vector{<:AbstractString},
        histname::Symbol;
        filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
        thinning::Int = 1,
    )::Vector{Vector{Dict}}

Load a single histogram field `histname` from multiple `statistics.jld2` files.

Returns a vector `out` such that:
- `out[i]` is a `Vector{Dict}` containing the histograms from `paths[i]`
- only the requested histogram field is loaded (RAM-safe)
- optional per-file filters can be applied

`filters[i] === nothing` means no filtering for that file.

`thinning` keeps every N-th histogram after filtering (N >= 1).

# Arguments
- `paths`: Path or collection of paths used for loading/saving data.
- `histname`: Histogram input data.

# Keyword Arguments
- `filters`: Keyword option `filters` controlling this method's behavior.
- `thinning`: Numeric control parameter for fitting/sampling resolution.

# Returns
- `result::Vector{Vector{Dict}}`: Output of `load_histograms_from_paths` with type annotation `Vector{Vector{Dict}}`.

# Throws
- `DimensionMismatch`: Raised when `length(filters) != length(paths)`.
- `DomainError`: Raised when `thinning < 1`."""
function load_histograms_from_paths(
    paths::Vector{<:AbstractString},
    histname::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Int = 1,
    size::Bool = false,
    verbose::Bool = false,
)::Vector
    cfg = _scan_config(thinning)
    filters_norm = _normalize_filters(paths, filters)
    extractor = _HistogramExtractor(histname, nothing, size)

    n = length(paths)
    out = Vector{Any}(undef, n)
    Threads.@threads for i in 1:n
        vals = _load_histograms_one(paths[i], filters_norm[i], cfg, extractor)
        verbose && println("  Loaded $(length(vals)) histograms from $(paths[i])")
        out[i] = vals
    end
    return out
end

"""
    load_histograms_from_paths(
        paths::Vector{<:AbstractString},
        histname::Symbol,
        scalar::Symbol;
        filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
        thinning::Int = 1,
    )::Vector{Vector{Tuple{Dict,Float64}}}

See the main method `load_histograms_from_paths(paths, histname; ...)` for core
loading, filtering, and thinning behavior.

# Changes in this overload
- Also reads scalar field `scalar` for every kept sample.
- Returns `(histogram, scalar)` pairs instead of bare histograms.
- Asserts that scalar values are `Real`.

# Arguments
- `paths`: Path or collection of paths used for loading/saving data.
- `histname`: Histogram input data.
- `scalar`: Scalar value(s) or scalar field identifier.

# Keyword Arguments
- `filters`: Keyword option `filters` controlling this method's behavior.
- `thinning`: Numeric control parameter for fitting/sampling resolution.

# Returns
- `result::Vector{Vector{Tuple{Dict,Real}}}`: Output of `load_histograms_from_paths` with type annotation `Vector{Vector{Tuple{Dict,Real}}}`.

# Throws
- `DimensionMismatch`: Raised when `length(filters) != length(paths)`.
- `DomainError`: Raised when `thinning < 1`.
- `TypeError`: Raised when the loaded scalar field is not `Real`."""
function load_histograms_from_paths(
    paths::Vector{<:AbstractString},
    histname::Symbol,
    scalar::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Int = 1,
    size::Bool = false,
    verbose::Bool = false,
)::Vector
    cfg = _scan_config(thinning)
    filters_norm = _normalize_filters(paths, filters)
    extractor = _HistogramExtractor(histname, scalar, size)

    n = length(paths)
    out = Vector{Any}(undef, n)
    Threads.@threads for i in 1:n
        vals = _load_histograms_one(paths[i], filters_norm[i], cfg, extractor)
        verbose && println("  Loaded $(length(vals)) histograms from $(paths[i]) with scalar $(scalar)")
        out[i] = vals
    end
    return out
end

"""
    _FieldExtractor{S}

Internal extractor plan for loading multiple fields (optionally with scalar pairing).

# Arguments
- `nfields`: Number of requested fields.
- `symbol_specs`: Indexed direct-field extraction specs.
- `hist_specs`: Indexed histogram-bin extraction specs.
- `scalar`: Scalar field symbol or `nothing`.

# Returns
- `extractor::_FieldExtractor{S}`: Field extraction plan.
"""
struct _FieldExtractor{S}
    nfields::Int
    symbol_specs::Vector{Tuple{Int,Symbol}}
    hist_specs::Vector{Tuple{Int,Symbol,Int64}}
    scalar::S
    size::Bool
end

"""
    _HistogramExtractor{S}

Internal extractor plan for loading histograms (optionally with scalar pairing).

# Arguments
- `histname`: Histogram field name.
- `scalar`: Scalar field symbol or `nothing`.

# Returns
- `extractor::_HistogramExtractor{S}`: Histogram extraction plan.
"""
struct _HistogramExtractor{S}
    histname::Symbol
    scalar::S
    size::Bool
end

"""
    _split_field_specs(fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}})

Split mixed field specs into direct-field and histogram-bin extraction lists.

# Arguments
- `fields`: Requested field specifications.

# Returns
- `symbol_specs`: Indexed `Symbol` extraction specs.
- `hist_specs`: Indexed `(Symbol, Int64)` extraction specs.
"""
function _split_field_specs(fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}})
    symbol_specs = Tuple{Int,Symbol}[]
    hist_specs = Tuple{Int,Symbol,Int64}[]
    for (j, field) in enumerate(fields)
        if field isa Symbol
            push!(symbol_specs, (j, field))
        else
            sym, bin = field
            push!(hist_specs, (j, sym, bin))
        end
    end
    return symbol_specs, hist_specs
end

"""
    _field_extractor(fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}})

Build a field extractor for plain field loading (no scalar pairing).

# Arguments
- `fields`: Requested field specifications.

# Returns
- `extractor::_FieldExtractor{Nothing}`: Internal extractor plan.
"""
function _field_extractor(
    fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}},
    size::Bool = false,
)
    symbol_specs, hist_specs = _split_field_specs(fields)
    return _FieldExtractor(length(fields), symbol_specs, hist_specs, nothing, size)
end

"""
    _field_extractor(fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}}, scalar::Symbol)

Build a field extractor for field loading with scalar pairing.

# Arguments
- `fields`: Requested field specifications.
- `scalar`: Scalar field name to pair with each extracted value.

# Returns
- `extractor::_FieldExtractor{Symbol}`: Internal extractor plan.
"""
function _field_extractor(
    fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}},
    scalar::Symbol,
    size::Bool = false,
)
    symbol_specs, hist_specs = _split_field_specs(fields)
    return _FieldExtractor(length(fields), symbol_specs, hist_specs, scalar, size)
end

"""
    _load_fields_one(path, filter, cfg, extractor::_FieldExtractor{Nothing})

Load one file's requested fields without scalar pairing.

# Arguments
- `path`: Input file path.
- `filter`: Per-record predicate or `nothing`.
- `cfg`: Internal scan configuration.
- `extractor`: Plain field extractor plan.

# Returns
- `vals::Vector{Any}`: Per-field vectors for one file.

# Throws
- Propagates data access and callback errors from scanning and extraction."""
function _load_fields_one(
    path::AbstractString,
    filter,
    cfg::_ScanConfig,
    extractor::_FieldExtractor{Nothing},
)
    vals = Vector{Any}(undef, extractor.nfields)
    for j in 1:extractor.nfields
        vals[j] = nothing
    end
    _scan_records(path, filter, cfg) do x
        n = extractor.size ? getfield(x, :n) : nothing
        if extractor.size && !(n isa Real)
            throw(TypeError(:load_fields_from_paths, "size field n", Real, n))
        end
        n64 = extractor.size ? Float64(n) : nothing
        for (j, sym) in extractor.symbol_specs
            value = getfield(x, sym)
            _push_auto_typed!(vals, j, extractor.size ? (value, n64) : value)
        end
        for (j, sym, bin) in extractor.hist_specs
            hist = getfield(x, sym)
            value = get(hist, bin, 0)
            _push_auto_typed!(vals, j, extractor.size ? (value, n64) : value)
        end
    end
    return _finalize_any_columns(vals, extractor.nfields)
end

"""
    _load_fields_one(path, filter, cfg, extractor::_FieldExtractor{Symbol})

Load one file's requested fields paired with a scalar value.

# Arguments
- `path`: Input file path.
- `filter`: Per-record predicate or `nothing`.
- `cfg`: Internal scan configuration.
- `extractor`: Scalar-paired field extractor plan.

# Returns
- `vals::Vector{Any}`: Per-field vectors of `(value, scalar)` pairs for one file.

# Throws
- `TypeError`: Raised when the loaded scalar field is not `Real`.
- Propagates data access errors from scanning and extraction."""
function _load_fields_one(
    path::AbstractString,
    filter,
    cfg::_ScanConfig,
    extractor::_FieldExtractor{Symbol},
)
    vals = Vector{Any}(undef, extractor.nfields)
    for j in 1:extractor.nfields
        vals[j] = nothing
    end
    scalar = extractor.scalar
    _scan_records(path, filter, cfg) do x
        s = getfield(x, scalar)
        if !(s isa Real)
            throw(TypeError(:load_fields_from_paths, "scalar field", Real, s))
        end
        s64 = Float64(s)
        n = extractor.size ? getfield(x, :n) : nothing
        if extractor.size && !(n isa Real)
            throw(TypeError(:load_fields_from_paths, "size field n", Real, n))
        end
        n64 = extractor.size ? Float64(n) : nothing
        for (j, sym) in extractor.symbol_specs
            value = getfield(x, sym)
            _push_auto_typed!(vals, j, extractor.size ? (value, s64, n64) : (value, s64))
        end
        for (j, sym, bin) in extractor.hist_specs
            hist = getfield(x, sym)
            value = get(hist, bin, 0)
            _push_auto_typed!(vals, j, extractor.size ? (value, s64, n64) : (value, s64))
        end
    end
    return _finalize_any_columns(vals, extractor.nfields)
end

"""
    _load_histograms_one(path, filter, cfg, extractor::_HistogramExtractor{Nothing})

Load one file's histogram values without scalar pairing.

# Arguments
- `path`: Input file path.
- `filter`: Per-record predicate or `nothing`.
- `cfg`: Internal scan configuration.
- `extractor`: Histogram extractor plan.

# Returns
- `hists::Vector{Dict}`: Loaded histograms for one file.

# Throws
- Propagates data access errors from scanning and extraction."""
function _load_histograms_one(
    path::AbstractString,
    filter,
    cfg::_ScanConfig,
    extractor::_HistogramExtractor{Nothing},
)
    hists = nothing
    _scan_records(path, filter, cfg) do x
        h = getfield(x, extractor.histname)
        n = extractor.size ? getfield(x, :n) : nothing
        if extractor.size && !(n isa Real)
            throw(TypeError(:load_histograms_from_paths, "size field n", Real, n))
        end
        value = extractor.size ? (h, Float64(n)) : h
        if hists === nothing
            hists = Vector{typeof(value)}()
        end
        push!(hists, value)
    end
    return hists === nothing ? (extractor.size ? Tuple{Dict,Float64}[] : Dict[]) : hists
end

"""
    _load_histograms_one(path, filter, cfg, extractor::_HistogramExtractor{Symbol})

Load one file's histogram values paired with scalar labels.

# Arguments
- `path`: Input file path.
- `filter`: Per-record predicate or `nothing`.
- `cfg`: Internal scan configuration.
- `extractor`: Scalar-paired histogram extractor plan.

# Returns
- `pairs::Vector{Tuple{Dict,Float64}}`: Loaded histogram/scalar pairs for one file.

# Throws
- `TypeError`: Raised when the loaded scalar field is not `Real`.
- Propagates data access errors from scanning and extraction."""
function _load_histograms_one(
    path::AbstractString,
    filter,
    cfg::_ScanConfig,
    extractor::_HistogramExtractor{Symbol},
)
    pairs = nothing
    scalar = extractor.scalar
    _scan_records(path, filter, cfg) do x
        h = getfield(x, extractor.histname)
        v = getfield(x, scalar)
        if !(v isa Real)
            throw(TypeError(:load_histograms_from_paths, "scalar field", Real, v))
        end
        if extractor.size
            n = getfield(x, :n)
            if !(n isa Real)
                throw(TypeError(:load_histograms_from_paths, "size field n", Real, n))
            end
            pair = (h, Float64(v), Float64(n))
        else
            pair = (h, Float64(v))
        end
        if pairs === nothing
            pairs = Vector{typeof(pair)}()
        end
        push!(pairs, pair)
    end
    if pairs === nothing
        return extractor.size ? Tuple{Dict,Float64,Float64}[] : Tuple{Dict,Float64}[]
    end
    return pairs
end

"""
    _load_field_with_scalar_one(path, filter, cfg, field, scalar)

Load one file's single field paired with scalar labels.

# Arguments
- `path`: Input file path.
- `filter`: Per-record predicate or `nothing`.
- `cfg`: Internal scan configuration.
- `field`: Requested field specification.
- `scalar`: Scalar field name.

# Returns
- `vals`: Concrete vector of `(value, Float64)` pairs for one file.

# Throws
- `TypeError`: Raised when the loaded scalar field is not `Real`.
- Propagates data access errors from scanning and extraction."""
function _load_field_with_scalar_one(
    path::AbstractString,
    filter,
    cfg::_ScanConfig,
    field::Union{Symbol,Tuple{Symbol,Int64}},
    scalar::Symbol,
    size::Bool = false,
)
    vals = nothing
    _scan_records(path, filter, cfg) do x
        s = getfield(x, scalar)
        if !(s isa Real)
            throw(TypeError(:load_field_with_scalar, "scalar field", Real, s))
        end
        if size
            n = getfield(x, :n)
            if !(n isa Real)
                throw(TypeError(:load_field_with_scalar, "size field n", Real, n))
            end
            pair = (_extract_field_value(x, field), Float64(s), Float64(n))
        else
            pair = (_extract_field_value(x, field), Float64(s))
        end
        if vals === nothing
            vals = Vector{typeof(pair)}()
        end
        push!(vals, pair)
    end
    if vals === nothing
        return size ? Vector{Tuple{Any,Float64,Float64}}() : Vector{Tuple{Any,Float64}}()
    end
    return vals
end

"""
    load_fields_from_paths(
        paths::Vector{<:AbstractString},
        fields::AbstractVector{<:Union{Symbol,Tuple{Symbol,Int64}}};
        filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
        thinning::Float64 = 1.0,
    )

Load multiple scalar fields from `statistics.jld2` files.

Returns `out[i][j]` as a `Vector` for `fields[j]` from `paths[i]`. Values are
returned in the same type as saved. If `fields[j]` is `(sym, bin)`, the value
is taken from histogram `sym` at `bin`.

Only the requested fields are loaded (RAM-safe). Optional per-file filters can be applied.
`thinning` keeps roughly a fraction of entries (e.g. 0.1 keeps ~every 10th sample).
`filters[i] === nothing` means no filtering for that file.

# Arguments
- `paths`: Path or collection of paths used for loading/saving data.
- `fields`: Observable/field names to extract, plot, or process.

# Keyword Arguments
- `filters`: Keyword option `filters` controlling this method's behavior.
- `thinning`: Numeric control parameter for fitting/sampling resolution.

# Returns
- `result`: Output of `load_fields_from_paths` as described in the summary above.

# Throws
- `DimensionMismatch`: Raised when `length(filters) != length(paths)`.
- `DomainError`: Raised when `thinning` is outside `(0, 1]`."""
function load_fields_from_paths(
    paths::Vector{<:AbstractString},
    fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}};
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Float64 = 1.0,
    size::Bool = false,
    verbose::Bool = false,
)
    cfg = _scan_config(thinning)
    filters_norm = _normalize_filters(paths, filters)
    extractor = _field_extractor(fields, size)

    n = length(paths)
    out = Vector{Vector{Any}}(undef, n)
    Threads.@threads for i in 1:n
        vals = _load_fields_one(paths[i], filters_norm[i], cfg, extractor)
        verbose && println("  Loaded $(length(vals[1])) values per field from $(paths[i])")
        out[i] = vals
    end
    return out
end

"""
    load_fields_from_paths(
        paths::Vector{<:AbstractString},
        fields::AbstractVector{<:Union{Symbol,Tuple{Symbol,Int64}}},
        scalar::Symbol;
        filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
        thinning::Float64 = 1.0,
        verbose::Bool = false,
    )

See the main method `load_fields_from_paths(paths, fields; ...)` for field
selection, filtering, and thinning behavior.

# Changes in this overload
- Also reads scalar field `scalar` for each kept sample.
- Returns `(value, scalar)` tuples per requested field.
- Asserts that scalar values are `Real`.

# Arguments
- `paths`: Path or collection of paths used for loading/saving data.
- `fields`: Observable/field names to extract, plot, or process.
- `scalar`: Scalar value(s) or scalar field identifier.

# Keyword Arguments
- `filters`: Keyword option `filters` controlling this method's behavior.
- `thinning`: Numeric control parameter for fitting/sampling resolution.
- `verbose`: Boolean toggle controlling output or execution behavior.

# Returns
- `result`: Output of `load_fields_from_paths` as described in the summary above.

# Throws
- `DimensionMismatch`: Raised when `length(filters) != length(paths)`.
- `DomainError`: Raised when `thinning` is outside `(0, 1]`.
- `TypeError`: Raised when the loaded scalar field is not `Real`."""
function load_fields_from_paths(
    paths::Vector{<:AbstractString},
    fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}},
    scalar::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Float64 = 1.0,
    size::Bool = false,
    verbose::Bool = false,
)
    cfg = _scan_config(thinning)
    filters_norm = _normalize_filters(paths, filters)
    extractor = _field_extractor(fields, scalar, size)

    n = length(paths)
    out = Vector{Vector{Any}}(undef, n)
    Threads.@threads for i in 1:n
        vals = _load_fields_one(paths[i], filters_norm[i], cfg, extractor)
        verbose && println("  Loaded $(length(vals[1])) values per field from $(paths[i]) with scalar $(scalar)")
        out[i] = vals
    end
    return out
end

"""
    load_field_with_scalar(
        paths::Vector{<:AbstractString},
        field::Union{Symbol,Tuple{Symbol,Int64}},
        scalar::Symbol;
        filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
        thinning::Float64 = 1.0,
        verbose::Bool = false,
    )

Load a single field together with a scalar, returning a concrete-typed
`Vector{Vector{Tuple{T,Float64}}}` where `T` matches the stored field type.

# Arguments
- `paths`: Path or collection of paths used for loading/saving data.
- `field`: Input parameter `field` used by this method.
- `scalar`: Scalar value(s) or scalar field identifier.

# Keyword Arguments
- `filters`: Keyword option `filters` controlling this method's behavior.
- `thinning`: Numeric control parameter for fitting/sampling resolution.
- `verbose`: Boolean toggle controlling output or execution behavior.

# Returns
- `result`: Output of `load_field_with_scalar` as described in the summary above.

# Throws
- `DimensionMismatch`: Raised when `length(filters) != length(paths)`.
- `DomainError`: Raised when `thinning` is outside `(0, 1]`.
- `TypeError`: Raised when the loaded scalar field is not `Real`."""
function load_field_with_scalar(
    paths::Vector{<:AbstractString},
    field::Union{Symbol,Tuple{Symbol,Int64}},
    scalar::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Float64 = 1.0,
    size::Bool = false,
    verbose::Bool = false,
)
    cfg = _scan_config(thinning)
    filters_norm = _normalize_filters(paths, filters)
    n = length(paths)

    # infer element type from first path to keep a concrete output type
    first_vals = _load_field_with_scalar_one(paths[1], filters_norm[1], cfg, field, scalar, size)
    out = Vector{typeof(first_vals)}(undef, n)
    out[1] = first_vals
    verbose && println("  Loaded $(length(out[1])) values from $(paths[1]) with scalar $(scalar)")

    Threads.@threads for i in 2:n
        vals = _load_field_with_scalar_one(paths[i], filters_norm[i], cfg, field, scalar, size)
        out[i] = vals
        verbose && println("  Loaded $(length(out[i])) values from $(paths[i]) with scalar $(scalar)")
    end

    return out
end

"""
    load_and_average_std_scalar(
        data_paths::Vector{String},
        fields::Vector{Symbol};
        verbose::Bool = false,
    )

See the main method
`load_and_average_std_scalar(data_paths, fields, scalar; ...)` for the grouped
scalar workflow.

# Changes in this overload
- No scalar grouping is performed.
- Loads plain field vectors via `load_fields_from_paths(paths, fields; ...)`.
- Returns per-dataset vectors of `(mean, std)` per requested field.

# Arguments
- `data_paths`: Path or collection of paths used for loading/saving data.
- `fields`: Observable/field names to extract, plot, or process.

# Keyword Arguments
- `verbose`: Boolean toggle controlling output or execution behavior.

# Returns
- `result`: Output of `load_and_average_std_scalar` as described in the summary above.

# Throws
- `DimensionMismatch`: Propagated when delegated loading receives mismatched `paths`/`filters` lengths.
- `DomainError`: Propagated for invalid delegated thinning settings."""
function load_and_average_std_scalar(
    data_paths::Vector{String},
    fields::Vector{Symbol};
    verbose::Bool = false,
)
    loaded = load_fields_from_paths(data_paths, fields; verbose = verbose)
    return [[(Statistics.mean(field), Statistics.std(field)) for field in dataset] for dataset in loaded]
end
