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
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function load_and_average_std_scalar(
    data_paths::Vector{String},
    fields::Vector{Symbol},
    scalar::Symbol;
    num_bins::Union{Nothing,Int} = nothing,
    verbose::Bool = false,
)
    loaded = load_fields_from_paths(data_paths, fields, scalar; verbose = verbose)

    out = Vector{Vector{Tuple{Real,Vector{Tuple{Float64,Float64}}}}}(undef, length(loaded))

    for (i, dataset) in enumerate(loaded)
        # dataset[j] is Vector{Tuple{value, scalar}}
        @assert !isempty(dataset) "no fields loaded"
        scalars = [s for (_, s) in dataset[1]]

        bin_edges = nothing
        if num_bins !== nothing
            @assert num_bins ≥ 1 "num_bins must be >= 1"
            vmin, vmax = minimum(scalars), maximum(scalars)
            if vmin == vmax
                bin_edges = [vmin, vmax + 1e-12]
            else
                bin_edges = collect(range(vmin, vmax; length = num_bins + 1))
            end
        end

        groups = Dict{Real,Vector{Vector{Any}}}() # scalar => list of per-field value vectors
        for j in 1:length(dataset)
            for (v, s) in dataset[j]
                if bin_edges !== nothing
                    idx = searchsortedlast(bin_edges, s)
                    idx = clamp(idx, 1, length(bin_edges) - 1)
                    s = (bin_edges[idx] + bin_edges[idx + 1]) / 2
                end
                if !haskey(groups, s)
                    groups[s] = [Any[] for _ in 1:length(dataset)]
                end
                push!(groups[s][j], v)
            end
        end

        scalars_sorted = sort(collect(keys(groups)))
        out[i] = Vector{Tuple{Real,Vector{Tuple{Float64,Float64}}}}(undef, length(scalars_sorted))
        for (k, s) in enumerate(scalars_sorted)
            stats = Vector{Tuple{Float64,Float64}}(undef, length(dataset))
            for j in 1:length(dataset)
                vals = groups[s][j]
                stats[j] = (Statistics.mean(vals), Statistics.std(vals))
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
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function load_histograms_from_paths(
    paths::Vector{<:AbstractString},
    histname::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Int = 1,
    verbose::Bool = false,
)::Vector{Vector{Dict}}
    n = length(paths)
    filters === nothing && (filters = fill(nothing, n))
    @assert length(filters) == n
    @assert thinning >= 1 "thinning must be >= 1"

    out = Vector{Vector{Dict}}(undef, n)

    Threads.@threads for i in 1:n
        path   = paths[i]
        filter = filters[i]

        hists = Dict[]
        seen = 0

        JLD2.jldopen(path, "r") do f
            nbatches = f["meta/nbatches"]

            for b in 1:nbatches
                batch = f["batches/$b"]

                for x in batch
                    filter !== nothing && !filter(x) && continue
                    seen += 1
                    if seen % thinning != 0
                        continue
                    end
                    push!(hists, getfield(x, histname))
                end
            end
        end

        verbose && println("  Loaded $(length(hists)) histograms from $path")
        out[i] = hists
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
    )::Vector{Vector{Tuple{Dict,Real}}}

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
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function load_histograms_from_paths(
    paths::Vector{<:AbstractString},
    histname::Symbol,
    scalar::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Int = 1,
    verbose::Bool = false,
)::Vector{Vector{Tuple{Dict,Real}}}
    n = length(paths)
    filters === nothing && (filters = fill(nothing, n))
    @assert length(filters) == n
    @assert thinning >= 1 "thinning must be >= 1"

    out = Vector{Vector{Tuple{Dict,Real}}}(undef, n)

    Threads.@threads for i in 1:n
        path   = paths[i]
        filter = filters[i]

        pairs = Tuple{Dict,Real}[]
        seen = 0

        JLD2.jldopen(path, "r") do f
            nbatches = f["meta/nbatches"]

            for b in 1:nbatches
                batch = f["batches/$b"]

                for x in batch
                    filter !== nothing && !filter(x) && continue
                    seen += 1
                    if seen % thinning != 0
                        continue
                    end
                    v = getfield(x, scalar)
                    @assert v isa Real "scalar field must be Real"
                    push!(pairs, (getfield(x, histname), v))
                end
            end
        end

        verbose && println("  Loaded $(length(pairs)) histograms from $path with scalar $(scalar)")
        out[i] = pairs
    end

    return out
end

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
    join_histograms(hists::Vector{Vector{Vector{Dict}}})::Vector{Vector{Dict}}

Join histograms across the first dimension by summing counts per bin.

Given a nested structure `hists[i][j][k]::Dict`, returns `out[j][k]` where
all dictionaries along `i` have been added bin-wise.

# Arguments
- `hists`: Histogram input data.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result::Vector{Vector{Dict}}`: Output of `join_histograms` with type annotation `Vector{Vector{Dict}}`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function join_histograms(
    hists::Vector{Vector{Vector{Dict}}},
)::Vector{Vector{Dict}}
    isempty(hists) && return Vector{Vector{Dict}}()

    n_outer = length(hists)
    n_groups = length(hists[1])
    n_hists = length(hists[1][1])

    for i in 1:n_outer
        @assert length(hists[i]) == n_groups
        for j in 1:n_groups
            @assert length(hists[i][j]) == n_hists
        end
    end

    out = [ [ Dict{Int,Int}() for _ in 1:n_hists ] for _ in 1:n_groups ]

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

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `join_histograms` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function join_histograms(
    hists::Vector{Vector{Vector{Tuple{Dict,Real}}}},
)::Vector{Vector{Tuple{Dict,Real}}}
    isempty(hists) && return Vector{Vector{Tuple{Dict,Real}}}()

    n_outer = length(hists)
    n_mid = length(hists[1])
    n_hists = length(hists[1][1])

    for i in 1:n_outer
        @assert length(hists[i]) == n_mid
        for j in 1:n_mid
            @assert length(hists[i][j]) == n_hists
        end
    end

    out = Vector{Vector{Tuple{Dict,Real}}}(undef, n_mid)
    for j in 1:n_mid
        out[j] = Vector{Tuple{Dict,Real}}(undef, n_hists)
        for k in 1:n_hists
            d_sum = Dict{Int,Float64}()
            s_ref = hists[1][j][k][2]
            for i in 1:n_outer
                d, s = hists[i][j][k]
                @assert s == s_ref "scalar mismatch for histogram index ($j,$k)"
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
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function load_fields_from_paths(
    paths::Vector{<:AbstractString},
    fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}};
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Float64 = 1.0,
    verbose::Bool = false,
)
    n = length(paths)
    filters === nothing && (filters = fill(nothing, n))
    @assert length(filters) == n
    @assert 0.0 < thinning <= 1.0
    step = max(1, round(Int, 1.0 / thinning))

    nfields = length(fields)
    out = [Vector{Vector}(undef, nfields) for _ in 1:n]

    Threads.@threads for i in 1:n
        path   = paths[i]
        filter = filters[i]
        vals = Vector{Any}(undef, nfields)
        for j in 1:nfields
            vals[j] = nothing
        end
        seen = 0

        JLD2.jldopen(path, "r") do f
            nbatches = f["meta/nbatches"]
            for b in 1:nbatches
                batch = f["batches/$b"]
                for x in batch
                    filter !== nothing && !filter(x) && continue
                    seen += 1
                    (seen - 1) % step != 0 && continue
                    for (j, field) in enumerate(fields)
                        local v
                        if field isa Symbol
                            v = getfield(x, field)
                        else
                            sym, bin = field
                            hist = getfield(x, sym)
                            v = get(hist, bin, 0)
                        end
                        if vals[j] === nothing
                            vals[j] = Vector{typeof(v)}()
                        end
                        push!(vals[j], v)
                    end
                end
            end
        end

        for j in 1:nfields
            out[i][j] = vals[j] === nothing ? Any[] : vals[j]
        end
        verbose && println("  Loaded $(length(vals[1])) values per field from $path")
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
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function load_fields_from_paths(
    paths::Vector{<:AbstractString},
    fields::Vector{<:Union{Symbol,Tuple{Symbol,Int64}}},
    scalar::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Float64 = 1.0,
    verbose::Bool = false,
)
    n = length(paths)
    filters === nothing && (filters = fill(nothing, n))
    @assert length(filters) == n
    @assert 0.0 < thinning <= 1.0
    step = max(1, round(Int, 1.0 / thinning))

    nfields = length(fields)
    out = [Vector{Vector}(undef, nfields) for _ in 1:n]

    Threads.@threads for i in 1:n
        path   = paths[i]
        filter = filters[i]
        vals = Vector{Any}(undef, nfields)
        for j in 1:nfields
            vals[j] = nothing
        end
        seen = 0

        JLD2.jldopen(path, "r") do f
            nbatches = f["meta/nbatches"]
            for b in 1:nbatches
                batch = f["batches/$b"]
                for x in batch
                    filter !== nothing && !filter(x) && continue
                    seen += 1
                    (seen - 1) % step != 0 && continue
                    s = getfield(x, scalar)
                    @assert s isa Real "scalar field must be Real"
                    for (j, field) in enumerate(fields)
                        local v
                        if field isa Symbol
                            v = getfield(x, field)
                        else
                            sym, bin = field
                            hist = getfield(x, sym)
                            v = get(hist, bin, 0)
                        end
                        pair = (v, s)
                        if vals[j] === nothing
                            vals[j] = Vector{typeof(pair)}()
                        end
                        push!(vals[j], pair)
                    end
                end
            end
        end

        for j in 1:nfields
            out[i][j] = vals[j] === nothing ? Any[] : vals[j]
        end
        verbose && println("  Loaded $(length(vals[1])) values per field from $path with scalar $(scalar)")
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
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function load_field_with_scalar(
    paths::Vector{<:AbstractString},
    field::Union{Symbol,Tuple{Symbol,Int64}},
    scalar::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    thinning::Float64 = 1.0,
    verbose::Bool = false,
)
    n = length(paths)
    filters === nothing && (filters = fill(nothing, n))
    @assert length(filters) == n
    @assert 0.0 < thinning <= 1.0
    step = max(1, round(Int, 1.0 / thinning))

    # infer element type from first path to keep a concrete output type
    function load_one(path, filter)
        vals = nothing
        seen = 0
        JLD2.jldopen(path, "r") do f
            nbatches = f["meta/nbatches"]
            for b in 1:nbatches
                batch = f["batches/$b"]
                for x in batch
                    filter !== nothing && !filter(x) && continue
                    seen += 1
                    (seen - 1) % step != 0 && continue
                    s = getfield(x, scalar)
                    @assert s isa Real "scalar field must be Real"
                    local v
                    if field isa Symbol
                        v = getfield(x, field)
                    else
                        sym, bin = field
                        hist = getfield(x, sym)
                        v = get(hist, bin, 0)
                    end
                    pair = (v, Float64(s))
                    if vals === nothing
                        vals = Vector{typeof(pair)}()
                    end
                    push!(vals, pair)
                end
            end
        end
        return vals === nothing ? Vector{Tuple{Any,Float64}}() : vals
    end

    first_vals = load_one(paths[1], filters[1])
    out = Vector{typeof(first_vals)}(undef, n)
    out[1] = first_vals
    verbose && println("  Loaded $(length(out[1])) values from $(paths[1]) with scalar $(scalar)")

    Threads.@threads for i in 2:n
        path   = paths[i]
        filter = filters[i]
        vals = load_one(path, filter)
        out[i] = vals
        verbose && println("  Loaded $(length(out[i])) values from $path with scalar $(scalar)")
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
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function load_and_average_std_scalar(
    data_paths::Vector{String},
    fields::Vector{Symbol};
    verbose::Bool = false,
)
    loaded = load_fields_from_paths(data_paths, fields; verbose = verbose)
    return [[(Statistics.mean(field), Statistics.std(field)) for field in dataset] for dataset in loaded]
end
