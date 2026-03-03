function validate_series_meta_lengths(n::Int, hist_labels, plot_types)
    if hist_labels !== nothing && length(hist_labels) != n
        throw(ArgumentError("hist_labels and data must have same length"))
    end
    if plot_types !== nothing && length(plot_types) != n
        throw(ArgumentError("plot_types and data must have same length"))
    end
    return nothing
end

function create_hist_axis(;
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = false,
    logscale_y::Bool = false,
    plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
)
    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
    )
    fig = CairoMakie.Figure(size = figsize)
    ax = CairoMakie.Axis(
        fig[1, 1];
        xscale = logscale_x ? log10 : identity,
        yscale = logscale_y ? log10 : identity,
    )
    ax.ylabel = ylabel === nothing ? "count" : ylabel
    ax.xlabel = xlabel === nothing ? LaTeXStrings.L"n" : xlabel
    plotlabel !== nothing && (ax.title = plotlabel)
    xlim !== nothing && CairoMakie.xlims!(ax, xlim...)
    ylim !== nothing && CairoMakie.ylims!(ax, ylim...)
    return fig, ax
end

function compute_log_eps_from_means(all_means, ylim, logscale_y::Bool)
    if !logscale_y
        return -Inf
    end
    if ylim !== nothing
        return ylim[1] * 1e-3
    end
    positives = [v for mean in all_means for v in mean if v > 0]
    if isempty(positives)
        throw(DomainError(logscale_y, "logscale_y=true requires at least one positive mean value"))
    end
    return minimum(positives) * 1e-3
end

function prepare_plot_series(mean, std, logscale_y::Bool, eps)
    if length(mean) != length(std)
        throw(ArgumentError("mean and std vectors must have equal length"))
    end
    x = collect(1:length(mean))
    ylo = mean .- std
    yhi = mean .+ std
    if logscale_y
        mask = mean .> 0
        x = x[mask]
        mean = mean[mask]
        ylo = max.(ylo[mask], eps)
        yhi = max.(yhi[mask], eps)
    end
    return x, mean, ylo, yhi
end

function draw_series!(
    ax,
    x,
    mean,
    ylo,
    yhi,
    color,
    plot_type::Symbol;
    plot_std::Bool = true,
    label = nothing,
    linewidth::Union{Nothing,Real} = nothing,
    markersize::Union{Nothing,Real} = nothing,
)
    if plot_type == :line
        plot_std && CairoMakie.band!(ax, x, ylo, yhi; color = (color, 0.2))
        if label === nothing
            isnothing(linewidth) ? CairoMakie.lines!(ax, x, mean; color = color) :
                CairoMakie.lines!(ax, x, mean; color = color, linewidth = linewidth)
        else
            isnothing(linewidth) ? CairoMakie.lines!(ax, x, mean; color = color, label = label) :
                CairoMakie.lines!(ax, x, mean; color = color, linewidth = linewidth, label = label)
        end
    elseif plot_type == :scatter
        if label === nothing
            isnothing(markersize) ? CairoMakie.scatter!(ax, x, mean; color = color) :
                CairoMakie.scatter!(ax, x, mean; color = color, markersize = markersize)
        else
            isnothing(markersize) ? CairoMakie.scatter!(ax, x, mean; color = color, label = label) :
                CairoMakie.scatter!(ax, x, mean; color = color, label = label, markersize = markersize)
        end
        if plot_std
            err = mean .- ylo
            CairoMakie.errorbars!(ax, x, mean, err, err; color = color)
        end
    else
        throw(ArgumentError("plot_types entries must be :line or :scatter"))
    end
end

function maybe_add_legend!(ax, hist_labels, legendpos, legendpadding, legendmargin, n_Legend_columns::Int)
    if hist_labels === nothing
        return nothing
    end
    legend_kwargs = (position = legendpos,)
    legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
    legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
    n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
    CairoMakie.axislegend(ax; legend_kwargs...)
    return nothing
end

function save_plot_result(plot, fig_path::String, return_axis::Bool)
    if return_axis
        fig, _ = plot
        CairoMakie.save(fig_path, fig)
        return plot
    end
    CairoMakie.save(fig_path, plot)
    return plot
end

function coerce_plain_vector_groups(vectors::AbstractVector)::Union{Nothing,Vector{Vector{AbstractVector}}}
    isempty(vectors) && return Vector{Vector{AbstractVector}}()
    groups = Vector{Vector{AbstractVector}}()
    sizehint!(groups, length(vectors))
    for group in vectors
        group isa AbstractVector || return nothing
        converted = Vector{AbstractVector}()
        sizehint!(converted, length(group))
        for sample in group
            sample isa AbstractVector || return nothing
            push!(converted, sample)
        end
        push!(groups, converted)
    end
    return groups
end

function coerce_scalar_vector_groups(vectors::AbstractVector)::Union{Nothing,Vector{Vector{Tuple{AbstractVector,Real}}}}
    isempty(vectors) && return Vector{Vector{Tuple{AbstractVector,Real}}}()
    groups = Vector{Vector{Tuple{AbstractVector,Real}}}()
    sizehint!(groups, length(vectors))
    for group in vectors
        group isa AbstractVector || return nothing
        converted = Vector{Tuple{AbstractVector,Real}}()
        sizehint!(converted, length(group))
        for sample in group
            sample isa Tuple || return nothing
            length(sample) == 2 || return nothing
            sample[1] isa AbstractVector || return nothing
            sample[2] isa Real || return nothing
            push!(converted, (sample[1], sample[2]))
        end
        push!(groups, converted)
    end
    return groups
end

"""
    plot_mean_histograms_with_std(
        data::Vector{Tuple{Vector{Float64},Vector{Float64}}};
        xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
        ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
        logscale_x::Bool = false,
        logscale_y::Bool = false,
        plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
        xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
        ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
        hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
        double_column::Bool = false,
        magnification::Real = 1.0,
        legendpos = :rt,
        legendpadding = nothing,
        legendmargin = nothing,
        n_Legend_columns::Int = 1,
        linewidth::Union{Nothing,Real} = nothing,
        plot_types::Union{Nothing,Vector{Symbol}} = nothing,
        markersize::Union{Nothing,Real} = nothing,
        return_axis::Bool = false,
    )::CairoMakie.Figure

Plot mean histograms with ±1σ bands.

Each element of `data` must be `(mean, std)`, where both are vectors
defined on the same binning.

# Arguments
- `data`: Collection of `(mean, std)` vectors, one per series.

# Keyword Arguments
- `xlim`: Axis limits for plotting.
- `ylim`: Axis limits for plotting.
- `logscale_x`: Toggle for logarithmic axis scaling.
- `logscale_y`: Toggle for logarithmic axis scaling.
- `plotlabel`: Text label shown in the plot output.
- `xlabel`: Text label shown in the plot output.
- `ylabel`: Text label shown in the plot output.
- `hist_labels`: Histogram input data.
- `double_column`: Boolean toggle controlling output or execution behavior.
- `magnification`: Keyword option `magnification` controlling this method's behavior.
- `legendpos`: Keyword option `legendpos` controlling this method's behavior.
- `legendpadding`: Keyword option `legendpadding` controlling this method's behavior.
- `legendmargin`: Keyword option `legendmargin` controlling this method's behavior.
- `n_Legend_columns`: Keyword option `n_Legend_columns` controlling this method's behavior.
- `linewidth`: Keyword option `linewidth` controlling this method's behavior.
- `plot_types`: Keyword option `plot_types` controlling this method's behavior.
- `markersize`: Keyword option `markersize` controlling this method's behavior.
- `return_axis`: Boolean toggle controlling output or execution behavior.

# Returns
- `result`: `Figure` or `(Figure, Axis)` when `return_axis=true`.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `DomainError`: Raised when log-scale plotting is requested without positive values."""
function plot_mean_histograms_with_std(
    data::Vector{Tuple{Vector{Float64},Vector{Float64}}};
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = false,
    logscale_y::Bool = false,
    plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
    linewidth::Union{Nothing,Real} = nothing,
    markersize::Union{Nothing,Real} = nothing,
    plot_types::Union{Nothing,Vector{Symbol}} = nothing,
    plot_std::Bool = true,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, CairoMakie.Axis}}

    validate_series_meta_lengths(length(data), hist_labels, plot_types)
    fig, ax = create_hist_axis(
        xlim = xlim,
        ylim = ylim,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        plotlabel = plotlabel,
        xlabel = xlabel,
        ylabel = ylabel,
        double_column = double_column,
        magnification = magnification,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
    )
    eps = compute_log_eps_from_means((m for (m, _) in data), ylim, logscale_y)

    for (i, (mean, std)) in enumerate(data)
        colors_obs = CairoMakie.theme(:palette).color
        colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
        color = colors[mod1(i, length(colors))]
        x, mean_plot, ylo, yhi = prepare_plot_series(mean, std, logscale_y, eps)
        plot_type = plot_types === nothing ? :line : plot_types[i]
        label = hist_labels === nothing ? nothing : hist_labels[i]
        draw_series!(
            ax, x, mean_plot, ylo, yhi, color, plot_type;
            plot_std = plot_std,
            label = label,
            linewidth = linewidth,
            markersize = markersize,
        )
    end

    maybe_add_legend!(ax, hist_labels, legendpos, legendpadding, legendmargin, n_Legend_columns)

    return return_axis ? (fig, ax) : fig
end

"""
    plot_mean_histograms_with_std(data::AbstractVector{<:Tuple}; kwargs...)

See `plot_mean_histograms_with_std(data::Vector{Tuple{Vector{Float64},Vector{Float64}}}; ...)`.

This overload expects `(value, mean, std)` tuples and uses `value` for colormap
encoding plus colorbar rendering.

# Arguments
- `data`: Collection of `(scalar, mean, std)` tuples.

# Keyword Arguments
- Same as the plain overload plus colorbar/comparison options (`colormap`, `colorbar_*`, `comp`, ...).

# Returns
- `result`: `Figure` or `(Figure, Axis)` when `return_axis=true`.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `DomainError`: Raised when log-scale plotting is requested without positive values."""
function plot_mean_histograms_with_std(
    data::AbstractVector{<:Tuple};
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = false,
    logscale_y::Bool = false,
    plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
    linewidth::Union{Nothing,Real} = nothing,
    plot_types::Union{Nothing,Vector{Symbol}} = nothing,
    markersize::Union{Nothing,Real} = nothing,
    colormap = :viridis,
    colorbar_label::Union{Nothing,AbstractString,LaTeXStrings.LaTeXString} = nothing,
    colorbar_ticks::Union{Nothing,Vector{<:Tuple{<:Real,Any}}} = nothing,
    plot_std::Bool = true,
    invert_color_scaling::Bool = false,
    colorbar_pos::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    colorbar_size::Union{Nothing,Tuple{Real,Real}} = nothing,
    comp::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}} = nothing,
    comp_color = :black,
    comp_linewidth::Union{Nothing,Real} = 2,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, CairoMakie.Axis}}

    # coerce to concrete (Real, Vector{Float64}, Vector{Float64}) tuples
    data = try
        [(Float64(d[1]), d[2], d[3]) for d in data]
    catch
        throw(ArgumentError("data entries must be tuples of the form (scalar, mean, std)"))
    end

    validate_series_meta_lengths(length(data), hist_labels, plot_types)
    fig, ax = create_hist_axis(
        xlim = xlim,
        ylim = ylim,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        plotlabel = plotlabel,
        xlabel = xlabel,
        ylabel = ylabel,
        double_column = double_column,
        magnification = magnification,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
    )
    eps = compute_log_eps_from_means((m for (_, m, _) in data), ylim, logscale_y)

    values = [v for (v, _, _) in data]
    vmin, vmax = minimum(values), maximum(values)
    denom = vmax == vmin ? 1.0 : (vmax - vmin)

    comp_x = nothing
    comp_mean = nothing
    # optional comparison band, plotted first
    if comp !== nothing
        mean_comp, std_comp = comp
        if length(mean_comp) != length(std_comp)
            throw(ArgumentError("comp mean and std must have equal length"))
        end
        x, mean_comp, ylo, yhi = prepare_plot_series(mean_comp, std_comp, logscale_y, eps)
        CairoMakie.band!(ax, x, ylo, yhi; color = (comp_color, 0.2))
        comp_x = x
        comp_mean = mean_comp
    end

    iter = invert_color_scaling ? reverse(data) : data
    for (i, (val, mean, std)) in enumerate(iter)
        t = (val - vmin) / denom
        t = invert_color_scaling ? (1 - t) : t
        color = PlotUtils.get(CairoMakie.cgrad(colormap), t)
        x, mean_plot, ylo, yhi = prepare_plot_series(mean, std, logscale_y, eps)
        plot_type = plot_types === nothing ? :line : plot_types[i]
        label = hist_labels === nothing ? nothing : hist_labels[i]
        draw_series!(
            ax, x, mean_plot, ylo, yhi, color, plot_type;
            plot_std = plot_std,
            label = label,
            linewidth = linewidth,
            markersize = markersize,
        )
    end

    if comp_x !== nothing
        isnothing(comp_linewidth) ? CairoMakie.lines!(ax, comp_x, comp_mean; color = comp_color) :
            CairoMakie.lines!(ax, comp_x, comp_mean; color = comp_color, linewidth = comp_linewidth)
    end

    cb_cmap = invert_color_scaling ? CairoMakie.Reverse(colormap) : colormap
    cb = if colorbar_pos === nothing
        CairoMakie.Colorbar(fig[1, 2], limits = (vmin, vmax), colormap = cb_cmap)
    else
        cb = CairoMakie.Colorbar(fig[1, 1], limits = (vmin, vmax), colormap = cb_cmap)
        cb.halign = colorbar_pos[1]
        cb.valign = colorbar_pos[2]
        cb.tellwidth = false
        cb.tellheight = false
        cb
    end
    if colorbar_size !== nothing
        cb.width = colorbar_size[1]
        cb.height = colorbar_size[2]
    end
    if colorbar_label !== nothing
        cb.label = colorbar_label
    end
    if colorbar_ticks !== nothing
        cb.ticks = ([t[1] for t in colorbar_ticks], [t[2] for t in colorbar_ticks])
    end

    return return_axis ? (fig, ax) : fig
end

"""
    plot_and_save_hists(hists::Vector{Vector{Dict{Int,Float64}}}, fig_path; kwargs...)

Compute `(mean, std)` per histogram group via `average_histogram_with_std`,
plot with `plot_mean_histograms_with_std`, save to `fig_path`, and
return the produced figure (or `(fig, ax)` when `return_axis=true`).

# Arguments
- `hists`: Histogram input data.
- `fig_path`: Output figure path used when saving plots.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: `Figure` or `(Figure, Axis)` when `return_axis=true`.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `DomainError`: Propagated from inner plotting when log-scale constraints are violated."""
function plot_and_save_hists(
    hists::Vector{Vector{Dict{Int,Float64}}},
    fig_path::String;
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = (10, 8, 8, 8),
    legendmargin = (5,5,5,5),
    n_Legend_columns::Int = 1,
    linewidth::Union{Nothing,Real} = nothing,
    markersize::Union{Nothing,Real} = nothing,
    plot_types::Union{Nothing,Vector{Symbol}} = nothing,
    return_axis::Bool = false,
    )::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, CairoMakie.Axis}}
    average_std = [average_histogram_with_std(hists[i]) for i in 1:length(hists)]
    plot = plot_mean_histograms_with_std(
        average_std; 
        xlim = xlim,
        ylim = ylim,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        plotlabel = plotlabel,
        xlabel = xlabel,
        ylabel = ylabel,
        hist_labels = hist_labels,
        double_column = double_column,
        magnification = magnification,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
        linewidth = linewidth,
        markersize = markersize,
        plot_types = plot_types,
        return_axis = return_axis,
        )

    return save_plot_result(plot, fig_path, return_axis)
end

"""
    plot_and_save_hists(hists::Vector{Vector{Tuple{D,Real}}}, fig_path; kwargs...) where {D<:AbstractDict}

See `plot_and_save_hists(hists::Vector{Vector{Dict{Int,Float64}}}, fig_path; ...)`.

This overload expects scalar-tagged histogram samples `(hist, scalar)`, computes
scalar-aware averages, and forwards colorbar-related keyword options to
`plot_mean_histograms_with_std`.

# Arguments
- `hists`: Histogram input data.
- `fig_path`: Output figure path used when saving plots.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: `Figure` or `(Figure, Axis)` when `return_axis=true`.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `DomainError`: Propagated from inner plotting when log-scale constraints are violated."""
function plot_and_save_hists(
    hists::Vector{Vector{Tuple{D,Real}}},
    fig_path::String;
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
    linewidth::Union{Nothing,Real} = nothing,
    plot_types::Union{Nothing,Vector{Symbol}} = nothing,
    markersize::Union{Nothing,Real} = nothing,
    colormap = :viridis,
    colorbar_label::Union{Nothing,AbstractString,LaTeXStrings.LaTeXString} = nothing,
    colorbar_ticks::Union{Nothing,Vector{<:Tuple{<:Real,Any}}} = nothing,
    plot_std::Bool = true,
    invert_color_scaling::Bool = false,
    colorbar_pos::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    colorbar_size::Union{Nothing,Tuple{Real,Real}} = nothing,
    comp::Union{Nothing,AbstractVector} = nothing,
    comp_color = :black,
    comp_linewidth::Union{Nothing,Real} = 2,
    return_axis::Bool = false,
    num_bins::Union{Nothing,Int} = nothing,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, CairoMakie.Axis}} where {D<:AbstractDict}
    average_std = [average_histogram_with_std(hists[i]; num_bins = num_bins) for i in 1:length(hists)]
    data = length(average_std) == 1 ? average_std[1] : vcat(average_std...)
    comp_mean_std = nothing
    if comp !== nothing
        comp_hists = comp isa AbstractVector && !isempty(comp) && comp[1] isa AbstractVector ? vcat(comp...) : comp
        comp_mean_std = average_histogram_with_std(comp_hists)
    end
    plot = plot_mean_histograms_with_std(
        data;
        xlim = xlim,
        ylim = ylim,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        plotlabel = plotlabel,
        xlabel = xlabel,
        ylabel = ylabel,
        hist_labels = hist_labels,
        double_column = double_column,
        magnification = magnification,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
        linewidth = linewidth,
        plot_types = plot_types,
        markersize = markersize,
        colormap = colormap,
        colorbar_label = colorbar_label,
        colorbar_ticks = colorbar_ticks,
        plot_std = plot_std,
        invert_color_scaling = invert_color_scaling,
        colorbar_pos = colorbar_pos,
        colorbar_size = colorbar_size,
        comp = comp_mean_std,
        comp_color = comp_color,
        comp_linewidth = comp_linewidth,
        return_axis = return_axis,
    )

    return save_plot_result(plot, fig_path, return_axis)
end

"""
    plot_and_save_vectors(vectors, fig_path; kwargs...)

Vector analogue of `plot_and_save_hists`: compute per-group `(mean, std)` using
`average_vectors_with_std`, plot with `plot_mean_histograms_with_std`, save to
`fig_path`, and return the plot object.

# Arguments
- `vectors`: Vector-valued input data.
- `fig_path`: Output figure path used when saving plots.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: `Figure` or `(Figure, Axis)` when `return_axis=true`.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `DomainError`: Propagated from inner plotting when log-scale constraints are violated."""
function plot_and_save_vectors(
    vectors::AbstractVector,
    fig_path::String;
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
    linewidth::Union{Nothing,Real} = nothing,
    plot_types::Union{Nothing,Vector{Symbol}} = nothing,
    markersize::Union{Nothing,Real} = nothing,
    colormap = :viridis,
    colorbar_label::Union{Nothing,AbstractString,LaTeXStrings.LaTeXString} = nothing,
    colorbar_ticks::Union{Nothing,Vector{<:Tuple{<:Real,Any}}} = nothing,
    plot_std::Bool = true,
    invert_color_scaling::Bool = false,
    colorbar_pos::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    colorbar_size::Union{Nothing,Tuple{Real,Real}} = nothing,
    comp::Union{Nothing,AbstractVector} = nothing,
    comp_color = :black,
    comp_linewidth::Union{Nothing,Real} = 2,
    return_axis::Bool = false,
    num_bins::Union{Nothing,Int} = nothing,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, CairoMakie.Axis}}
    scalar_groups = coerce_scalar_vector_groups(vectors)
    if scalar_groups !== nothing
        return plot_and_save_vectors_scalar(
            scalar_groups,
            fig_path;
            xlim = xlim,
            ylim = ylim,
            logscale_x = logscale_x,
            logscale_y = logscale_y,
            plotlabel = plotlabel,
            xlabel = xlabel,
            ylabel = ylabel,
            hist_labels = hist_labels,
            double_column = double_column,
            magnification = magnification,
            legendpos = legendpos,
            legendpadding = legendpadding,
            legendmargin = legendmargin,
            n_Legend_columns = n_Legend_columns,
            linewidth = linewidth,
            plot_types = plot_types,
            markersize = markersize,
            colormap = colormap,
            colorbar_label = colorbar_label,
            colorbar_ticks = colorbar_ticks,
            plot_std = plot_std,
            invert_color_scaling = invert_color_scaling,
            colorbar_pos = colorbar_pos,
            colorbar_size = colorbar_size,
            comp = comp,
            comp_color = comp_color,
            comp_linewidth = comp_linewidth,
            return_axis = return_axis,
            num_bins = num_bins,
        )
    end

    plain_groups = coerce_plain_vector_groups(vectors)
    if plain_groups !== nothing
        plain_legendpadding = isnothing(legendpadding) ? (10, 8, 8, 8) : legendpadding
        plain_legendmargin = isnothing(legendmargin) ? (5, 5, 5, 5) : legendmargin
        return plot_and_save_vectors_plain(
            plain_groups,
            fig_path;
            xlim = xlim,
            ylim = ylim,
            logscale_x = logscale_x,
            logscale_y = logscale_y,
            plotlabel = plotlabel,
            xlabel = xlabel,
            ylabel = ylabel,
            hist_labels = hist_labels,
            double_column = double_column,
            magnification = magnification,
            legendpos = legendpos,
            legendpadding = plain_legendpadding,
            legendmargin = plain_legendmargin,
            n_Legend_columns = n_Legend_columns,
            linewidth = linewidth,
            markersize = markersize,
            plot_types = plot_types,
            return_axis = return_axis,
        )
    end

    throw(ArgumentError("plot_and_save_vectors: unsupported input format"))
end

function plot_and_save_vectors_plain(
    vectors::Vector{Vector{AbstractVector}},
    fig_path::String;
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = (10, 8, 8, 8),
    legendmargin = (5,5,5,5),
    n_Legend_columns::Int = 1,
    linewidth::Union{Nothing,Real} = nothing,
    markersize::Union{Nothing,Real} = nothing,
    plot_types::Union{Nothing,Vector{Symbol}} = nothing,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, CairoMakie.Axis}}
    average_std = [average_vectors_with_std(vectors[i]) for i in 1:length(vectors)]
    plot = plot_mean_histograms_with_std(
        average_std;
        xlim = xlim,
        ylim = ylim,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        plotlabel = plotlabel,
        xlabel = xlabel,
        ylabel = ylabel,
        hist_labels = hist_labels,
        double_column = double_column,
        magnification = magnification,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
        linewidth = linewidth,
        markersize = markersize,
        plot_types = plot_types,
        return_axis = return_axis,
    )

    return save_plot_result(plot, fig_path, return_axis)
end

function plot_and_save_vectors_scalar(
    vectors::Vector{Vector{Tuple{AbstractVector,Real}}},
    fig_path::String;
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    plotlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    xlabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    ylabel::Union{AbstractString,LaTeXStrings.LaTeXString,Nothing} = nothing,
    hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
    linewidth::Union{Nothing,Real} = nothing,
    plot_types::Union{Nothing,Vector{Symbol}} = nothing,
    markersize::Union{Nothing,Real} = nothing,
    colormap = :viridis,
    colorbar_label::Union{Nothing,AbstractString,LaTeXStrings.LaTeXString} = nothing,
    colorbar_ticks::Union{Nothing,Vector{<:Tuple{<:Real,Any}}} = nothing,
    plot_std::Bool = true,
    invert_color_scaling::Bool = false,
    colorbar_pos::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    colorbar_size::Union{Nothing,Tuple{Real,Real}} = nothing,
    comp::Union{Nothing,AbstractVector} = nothing,
    comp_color = :black,
    comp_linewidth::Union{Nothing,Real} = 2,
    return_axis::Bool = false,
    num_bins::Union{Nothing,Int} = nothing,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, CairoMakie.Axis}}
    average_std = [average_vectors_with_std(vectors[i]; num_bins = num_bins) for i in 1:length(vectors)]
    data = length(average_std) == 1 ? average_std[1] : vcat(average_std...)
    comp_mean_std = nothing
    if comp !== nothing
        comp_vecs = comp isa AbstractVector && !isempty(comp) && comp[1] isa AbstractVector ? vcat(comp...) : comp
        comp_mean_std = average_vectors_with_std(comp_vecs)
    end
    plot = plot_mean_histograms_with_std(
        data;
        xlim = xlim,
        ylim = ylim,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        plotlabel = plotlabel,
        xlabel = xlabel,
        ylabel = ylabel,
        hist_labels = hist_labels,
        double_column = double_column,
        magnification = magnification,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
        linewidth = linewidth,
        plot_types = plot_types,
        markersize = markersize,
        colormap = colormap,
        colorbar_label = colorbar_label,
        colorbar_ticks = colorbar_ticks,
        plot_std = plot_std,
        invert_color_scaling = invert_color_scaling,
        colorbar_pos = colorbar_pos,
        colorbar_size = colorbar_size,
        comp = comp_mean_std,
        comp_color = comp_color,
        comp_linewidth = comp_linewidth,
        return_axis = return_axis,
    )

    return save_plot_result(plot, fig_path, return_axis)
end
