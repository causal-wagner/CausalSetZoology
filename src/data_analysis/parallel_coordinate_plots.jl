"""
    transform_to_scale!(col)

Normalize a numeric vector to `[0, 1]` for parallel-coordinate plotting.

If all values are (approximately) equal and nonzero, values are scaled by the
common value. If all values are approximately zero, input is returned unchanged.

# Arguments
- `col`: Input parameter `col` used by this method.

# Keyword Arguments
- This method has no keyword arguments.

# Returns
- `result`: Output of `transform_to_scale!` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function transform_to_scale!(col)
    min_val = minimum(col)
    max_val = maximum(col)
    if abs(max_val - min_val) < 1e-3
        if abs(max_val) < 1e-3
            return col
        end
        return col ./ max_val
    else
        return (col .- min_val) ./ (max_val - min_val)
    end
end

"""
    parallel_plot_df(data, observables; kinds=nothing, thinning=1.0)

Build a long-format `DataFrame` for parallel-coordinate plots.

`data[i]` must contain one vector per observable for dataset `i`. All vectors in
one dataset must have equal sample count. Rows are optionally thinned via
`thinning` and annotated with `kind` and `id` columns.

# Arguments
- `data`: Input dataset(s) consumed by this method.
- `observables`: Observable/field names to extract, plot, or process.

# Keyword Arguments
- `kinds`: Category labels used for grouping or legend entries.
- `thinning`: Numeric control parameter for fitting/sampling resolution.

# Returns
- `result`: Output of `parallel_plot_df` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function parallel_plot_df(
    data,
    observables::AbstractVector{<:Union{Symbol,AbstractString}};
    kinds = nothing,
    thinning::Float64 = 1.0,
)
    npaths = length(data)
    nfields = length(observables)
    kinds === nothing && (kinds = ["set$(i)" for i in 1:npaths])
    dfs = Vector{DataFrames.DataFrame}(undef, npaths)
    @assert 0.0 < thinning <= 1.0
    for i in 1:npaths
        vals = data[i]
        @assert length(vals) == nfields
        nsamples = length(vals[1])
        for j in 2:nfields
            @assert length(vals[j]) == nsamples
        end
        step = max(1, round(Int, 1.0 / thinning))
        idxs = 1:step:nsamples
        df = DataFrames.DataFrame()
        for (field, v) in zip(observables, vals)
            df[!, String(field)] = Float64.(v[idxs])
        end
        df[!, "kind"] = fill(kinds[i], length(idxs))
        df[!, "id"] = 1:length(idxs)
        dfs[i] = df
    end
    return vcat(dfs...)
end

"""
    create_parallel_plot(plot_data, observables, kinds; kwargs...)::AlgebraOfGraphics.FigureGrid

Create a parallel-coordinate plot using AlgebraOfGraphics/CairoMakie.

Pipeline:
1. Build dataframe via `parallel_plot_df`.
2. Normalize numeric observable columns to `[0, 1]`.
3. Optionally reorder/filter categories (`order_vec`, `choose_kinds`).
4. Optionally subsample rows (`sample_n`).
5. Draw lines + points with categorical color mapping and optional legend.

If `fig_path` is provided, the figure is saved.

# Arguments
- `plot_data`: Input dataset(s) consumed by this method.
- `observables`: Observable/field names to extract, plot, or process.
- `kinds`: Category labels used for grouping or legend entries.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result::AlgebraOfGraphics.FigureGrid`: Output of `create_parallel_plot` with type annotation `AlgebraOfGraphics.FigureGrid`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function create_parallel_plot(
    plot_data::Vector{Vector{Vector{Float64}}},
    observables::AbstractVector{<:Union{Symbol,AbstractString}},
    kinds::Vector{String};
    thinning::Float64 = 1.0,
    color_transparency::Float64 = 1.0,
    legend::Bool = true,
    legendpos::Symbol = :tl,
    legend_offset::Tuple{<:Real,<:Real} = (0, 0),
    fig_path::Union{Nothing,String}=nothing,
    sample_n::Union{Nothing,Int} = nothing,
    color_vec::Union{Nothing,Vector{Int64}} = nothing,
    order_vec::Union{Nothing,Vector{Int64}} = nothing,
    choose_kinds::Union{Nothing,Vector{Int64}} = nothing,
)::AlgebraOfGraphics.FigureGrid

    parallel_df = parallel_plot_df(plot_data, observables; kinds=kinds, thinning = thinning)

    normalized_parallel_df = deepcopy(parallel_df)
    for col in names(normalized_parallel_df)
        if col != "kind" && col != "id" && eltype(normalized_parallel_df[!, col]) <: Number
            normalized_parallel_df[!, col] = transform_to_scale!(normalized_parallel_df[!, col])
        end
    end
    base_indices = collect(1:length(kinds))
    ordered_indices = order_vec === nothing ? base_indices : order_vec
    @assert all(in.(ordered_indices, Ref(base_indices))) "order_vec must be valid indices of kinds"
    selected_indices = choose_kinds === nothing ? ordered_indices : [i for i in ordered_indices if i in choose_kinds]
    selected_kinds = kinds[selected_indices]
    if choose_kinds !== nothing
        normalized_parallel_df = filter(row -> row.kind in selected_kinds, normalized_parallel_df)
    end
    if "kind" in names(normalized_parallel_df)
        normalized_parallel_df[!, "kind"] = CategoricalArrays.categorical(
            normalized_parallel_df[!, "kind"];
            levels = selected_kinds,
            ordered = true,
        )
        sort!(normalized_parallel_df, [:kind, :id])
    end

    nrows = Base.size(normalized_parallel_df, 1)
    if sample_n === nothing
        idxs = 1:nrows
    else
        n_sample = min(sample_n, nrows)
        idxs = StatsBase.sample(1:nrows, n_sample; replace=false)
    end

    normalized_parallel_df_long = DataFrames.stack(
        normalized_parallel_df[idxs, :],
        DataFrames.Not(:id, :kind),
        variable_name = :variable,
        value_name = :value,
    )

    variables = String.(observables)
    colors_obs = CairoMakie.theme(:palette).color
    colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
    palette_full = if color_vec === nothing
        [colors[mod1(i, length(colors))] for i in base_indices]
    else
        if all(x -> x isa Integer, color_vec)
            [colors[mod1(i, length(colors))] for i in color_vec]
        else
            color_vec
        end
    end
    palette = [palette_full[i] for i in selected_indices]

    pp_specs(df) = AlgebraOfGraphics.data(df) * AlgebraOfGraphics.mapping(
        :variable => AlgebraOfGraphics.sorter(variables),
        :value,
        color = :kind,
        group = :id,
    ) * (AlgebraOfGraphics.visual(CairoMakie.Lines, linewidth=0.5) + AlgebraOfGraphics.visual(CairoMakie.Scatter, alpha=0.3, markersize=7))

    parallel_plot = pp_specs(normalized_parallel_df_long)

    figsize = apply_paper_theme!(double_column = true, magnification = 1., color_transparency = color_transparency)

    fig = AlgebraOfGraphics.draw(
        parallel_plot,
        AlgebraOfGraphics.scales(Color = (; palette = palette)),
        figure = (; size = figsize),
        axis = (xticklabelrotation = pi / 4, limits = (nothing, (0.0, 1.0)), xlabel = "", ylabel = ""),
        legend = (show = false,),
    )
    ax = fig.figure.current_axis[]
    if ax !== nothing
        ax.xticks = (1:length(variables), observables)
        n = length(variables)
        CairoMakie.xlims!(ax, 0.8, n + 0.2)
    end

    if legend
        # Custom opaque legend
        colors_obs = CairoMakie.theme(:palette).color
        colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
        opaque(c) = begin
            col = CairoMakie.to_color(c)
            CairoMakie.RGBAf(col.r, col.g, col.b, 1f0)
        end
        legend_colors = map(opaque, palette)
        elements = [CairoMakie.LineElement(color = c, linewidth = 2.0) for c in legend_colors]
        pos = legendpos === :lt ? CairoMakie.TopLeft() :
            legendpos === :rt ? CairoMakie.TopRight() :
            legendpos === :lb ? CairoMakie.BottomLeft() :
            legendpos === :rb ? CairoMakie.BottomRight() :
            CairoMakie.TopRight()
        leg = CairoMakie.Legend(
            fig.figure[1, 1, pos],
            elements,
            selected_kinds;
            tellheight = false,
            tellwidth = false,
            margin = (10, 10, 10, 10),
        )
        CairoMakie.translate!(leg.blockscene, legend_offset[1], legend_offset[2], 0)
    end
    if !isnothing(fig_path)
        CairoMakie.save(fig_path, fig)
    end
    return fig
end
