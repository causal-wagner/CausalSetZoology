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
    )::Figure

Plot mean histograms with ±1σ bands.

Each element of `data` must be `(mean, std)`, where both are vectors
defined on the same binning.

# Arguments
- `data`: Input dataset(s) consumed by this method.

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
- `result::Figure`: Output of `plot_mean_histograms_with_std` with type annotation `Figure`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
)::Union{Figure, Tuple{Figure, Axis}}

    if hist_labels !== nothing
        @assert length(hist_labels) == length(data) "hist_labels and data must have same length"
    end
    if plot_types !== nothing
        @assert length(plot_types) == length(data) "plot_types and data must have same length"
    end

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

    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1];
        xscale = logscale_x ? log10 : identity,
        yscale = logscale_y ? log10 : identity,
    )

    ax.ylabel = ylabel === nothing ? "count" : ylabel
    ax.xlabel = xlabel === nothing ? L"n" : xlabel
    plotlabel !== nothing && (ax.title  = plotlabel)

    xlim !== nothing && xlims!(ax, xlim...)
    ylim !== nothing && ylims!(ax, ylim...)

    eps = if logscale_y
        if ylim !== nothing
            ylim[1] * 1e-3
        else
            minpos = minimum(v for (m, _) in data for v in m if v > 0)
            minpos * 1e-3
        end
    else
        -Inf
    end

    for (i, (mean, std)) in enumerate(data)
        @assert length(mean) == length(std)

        colors_obs = Makie.theme(:palette).color
        colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
        color = colors[mod1(i, length(colors))]

        x   = collect(1:length(mean))
        ylo = mean .- std
        yhi = mean .+ std

        if logscale_y
            mask = mean .> 0
            x = x[mask]
            mean = mean[mask]
            ylo = ylo[mask]
            yhi = yhi[mask]

            ylo = max.(ylo, eps)
            yhi = max.(yhi, eps)
        end

        plot_type = plot_types === nothing ? :line : plot_types[i]
        if plot_type == :line
            if plot_std
                band!(ax, x, ylo, yhi; color = (color, 0.2))
            end
            if hist_labels === nothing
                isnothing(linewidth) ? lines!(ax, x, mean; color = color) : lines!(ax, x, mean; color = color, linewidth = linewidth)
            else
                isnothing(linewidth) ? lines!(ax, x, mean; color = color, label = hist_labels[i]) : lines!(ax, x, mean; color = color, linewidth = linewidth, label = hist_labels[i])
            end
        elseif plot_type == :scatter
            if hist_labels === nothing
                isnothing(markersize) ? scatter!(ax, x, mean; color = color) : scatter!(ax, x, mean; color = color, markersize = markersize)
            else
                isnothing(markersize) ? scatter!(ax, x, mean; color = color, label = hist_labels[i]) : scatter!(ax, x, mean; color = color, label = hist_labels[i], markersize = markersize)
            end
            if plot_std
                err = mean .- ylo
                Makie.errorbars!(ax, x, mean, err, err; color = color)
            end
        else
            error("plot_types entries must be :line or :scatter")
        end
    end

    if hist_labels !== nothing
        legend_kwargs = (position = legendpos,)
        legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
        legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
        n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
        axislegend(ax; legend_kwargs...)
    end

    return return_axis ? (fig, ax) : fig
end

"""
    plot_mean_histograms_with_std(data::AbstractVector{<:Tuple}; kwargs...)

See `plot_mean_histograms_with_std(data::Vector{Tuple{Vector{Float64},Vector{Float64}}}; ...)`.

This overload expects `(value, mean, std)` tuples and uses `value` for colormap
encoding plus colorbar rendering.

# Arguments
- `data`: Input dataset(s) consumed by this method.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: Output of `plot_mean_histograms_with_std` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
)::Union{Figure, Tuple{Figure, Axis}}

    # coerce to concrete (Real, Vector{Float64}, Vector{Float64}) tuples
    data = [(Float64(d[1]), d[2], d[3]) for d in data]

    if hist_labels !== nothing
        @assert length(hist_labels) == length(data) "hist_labels and data must have same length"
    end
    if plot_types !== nothing
        @assert length(plot_types) == length(data) "plot_types and data must have same length"
    end

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

    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1];
        xscale = logscale_x ? log10 : identity,
        yscale = logscale_y ? log10 : identity,
    )

    ax.ylabel = ylabel === nothing ? "count" : ylabel
    ax.xlabel = xlabel === nothing ? L"n" : xlabel
    plotlabel !== nothing && (ax.title  = plotlabel)

    xlim !== nothing && xlims!(ax, xlim...)
    ylim !== nothing && ylims!(ax, ylim...)

    eps = if logscale_y
        if ylim !== nothing
            ylim[1] * 1e-3
        else
            minpos = minimum(v for (_, m, _) in data for v in m if v > 0)
            minpos * 1e-3
        end
    else
        -Inf
    end

    values = [v for (v, _, _) in data]
    vmin, vmax = minimum(values), maximum(values)
    denom = vmax == vmin ? 1.0 : (vmax - vmin)

    comp_x = nothing
    comp_mean = nothing
    # optional comparison band, plotted first
    if comp !== nothing
        mean_comp, std_comp = comp
        x = collect(1:length(mean_comp))
        ylo = mean_comp .- std_comp
        yhi = mean_comp .+ std_comp
        if logscale_y
            mask = mean_comp .> 0
            x = x[mask]
            mean_comp = mean_comp[mask]
            ylo = ylo[mask]
            yhi = yhi[mask]
            ylo = max.(ylo, eps)
            yhi = max.(yhi, eps)
        end
        band!(ax, x, ylo, yhi; color = (comp_color, 0.2))
        comp_x = x
        comp_mean = mean_comp
    end

    iter = invert_color_scaling ? reverse(data) : data
    for (i, (val, mean, std)) in enumerate(iter)
        @assert length(mean) == length(std)

        t = (val - vmin) / denom
        t = invert_color_scaling ? (1 - t) : t
        color = PlotUtils.get(Makie.cgrad(colormap), t)

        x   = collect(1:length(mean))
        ylo = mean .- std
        yhi = mean .+ std

        if logscale_y
            mask = mean .> 0
            x = x[mask]
            mean = mean[mask]
            ylo = ylo[mask]
            yhi = yhi[mask]

            ylo = max.(ylo, eps)
            yhi = max.(yhi, eps)
        end

        plot_type = plot_types === nothing ? :line : plot_types[i]
        if plot_type == :line
            if plot_std
                band!(ax, x, ylo, yhi; color = (color, 0.2))
            end
            if hist_labels === nothing
                isnothing(linewidth) ? lines!(ax, x, mean; color = color) : lines!(ax, x, mean; color = color, linewidth = linewidth)
            else
                isnothing(linewidth) ? lines!(ax, x, mean; color = color, label = hist_labels[i]) : lines!(ax, x, mean; color = color, linewidth = linewidth, label = hist_labels[i])
            end
        elseif plot_type == :scatter
            if hist_labels === nothing
                isnothing(markersize) ? scatter!(ax, x, mean; color = color) : scatter!(ax, x, mean; color = color, markersize = markersize)
            else
                isnothing(markersize) ? scatter!(ax, x, mean; color = color, label = hist_labels[i]) : scatter!(ax, x, mean; color = color, label = hist_labels[i], markersize = markersize)
            end
            if plot_std
                err = mean .- ylo
                Makie.errorbars!(ax, x, mean, err, err; color = color)
            end
        else
            error("plot_types entries must be :line or :scatter")
        end
    end

    if comp_x !== nothing
        isnothing(comp_linewidth) ? lines!(ax, comp_x, comp_mean; color = comp_color) :
            lines!(ax, comp_x, comp_mean; color = comp_color, linewidth = comp_linewidth)
    end

    cb_cmap = invert_color_scaling ? Makie.Reverse(colormap) : colormap
    cb = if colorbar_pos === nothing
        Colorbar(fig[1, 2], limits = (vmin, vmax), colormap = cb_cmap)
    else
        cb = Colorbar(fig[1, 1], limits = (vmin, vmax), colormap = cb_cmap)
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
    plot_and_save_hists(hists::Vector{Vector{Dict{Int,Float64}}}, fig_name; kwargs...)

Compute `(mean, std)` per histogram group via `average_histogram_with_std`,
plot with `plot_mean_histograms_with_std`, save to `fig_path(fig_name)`, and
return the produced figure (or `(fig, ax)` when `return_axis=true`).

# Arguments
- `hists`: Histogram input data.
- `fig_name`: Output figure name/path used when saving plots.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: Output of `plot_and_save_hists` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function plot_and_save_hists(
    hists::Vector{Vector{Dict{Int,Float64}}},
    fig_name::String;
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
    )::Union{Figure, Tuple{Figure, Axis}}
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

    save(fig_path(fig_name), plot)

    return plot
end

"""
    plot_and_save_hists(hists::Vector{Vector{Tuple{D,Real}}}, fig_name; kwargs...) where {D<:AbstractDict}

See `plot_and_save_hists(hists::Vector{Vector{Dict{Int,Float64}}}, fig_name; ...)`.

This overload expects scalar-tagged histogram samples `(hist, scalar)`, computes
scalar-aware averages, and forwards colorbar-related keyword options to
`plot_mean_histograms_with_std`.

# Arguments
- `hists`: Histogram input data.
- `fig_name`: Output figure name/path used when saving plots.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: Output of `plot_and_save_hists` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function plot_and_save_hists(
    hists::Vector{Vector{Tuple{D,Real}}},
    fig_name::String;
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
)::Union{Figure, Tuple{Figure, Axis}} where {D<:AbstractDict}
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

    if return_axis
        fig, ax = plot
        save(fig_path(fig_name), fig)
        return plot
    else
        save(fig_path(fig_name), plot)
        return plot
    end
end

"""
    plot_and_save_vectors(vectors, fig_name; kwargs...)

Vector analogue of `plot_and_save_hists`: compute per-group `(mean, std)` using
`average_vectors_with_std`, plot with `plot_mean_histograms_with_std`, save to
`fig_path(fig_name)`, and return the plot object.

# Arguments
- `vectors`: Vector-valued input data.
- `fig_name`: Output figure name/path used when saving plots.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: Output of `plot_and_save_vectors` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function plot_and_save_vectors(
    vectors::AbstractVector,
    fig_name::String;
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
)::Union{Figure, Tuple{Figure, Axis}}
    @assert all(v -> v isa AbstractVector, vectors) "vectors must be a collection of vector samples"
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

    save(fig_path(fig_name), plot)
    return plot
end

"""
    plot_and_save_vectors(vectors::AbstractVector, fig_name; kwargs...)

See `plot_and_save_vectors(vectors, fig_name; ...)`.

Dispatch helper: if `vectors` looks like grouped `(vector, scalar)` tuples, it
forwards to the scalar-aware overload; otherwise throws an informative error.

# Arguments
- `vectors`: Vector-valued input data.
- `fig_name`: Output figure name/path used when saving plots.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: Output of `plot_and_save_vectors` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function plot_and_save_vectors(
    vectors::AbstractVector,
    fig_name::String;
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
    return_axis::Bool = false,
    num_bins::Union{Nothing,Int} = nothing,
)::Union{Figure, Tuple{Figure, Axis}}
    if !isempty(vectors) && vectors[1] isa AbstractVector &&
       !isempty(vectors[1]) && vectors[1][1] isa Tuple &&
       length(vectors[1][1]) == 2
        return plot_and_save_vectors(
            Vector{Vector{Tuple{AbstractVector,Real}}}(vectors),
            fig_name;
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
            return_axis = return_axis,
            num_bins = num_bins,
        )
    end

    error("plot_and_save_vectors: input does not look like a vector+scalar tuple dataset; use the non-colorbar overload.")
end

"""
    plot_and_save_vectors(vectors::AbstractVector{<:AbstractVector{<:Tuple{<:AbstractVector,<:Real}}}, fig_name; kwargs...)

See `plot_and_save_vectors(vectors, fig_name; ...)`.

Scalar-aware overload for grouped `(vector, scalar)` data with colormap/colorbar
support and optional comparison band.

# Arguments
- `vectors`: Vector-valued input data.
- `fig_name`: Output figure name/path used when saving plots.

# Keyword Arguments
- `kwargs`: Additional keyword arguments forwarded to inner methods.

# Returns
- `result`: Output of `plot_and_save_vectors` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function plot_and_save_vectors(
    vectors::AbstractVector{<:AbstractVector{<:Tuple{<:AbstractVector,<:Real}}},
    fig_name::String;
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
)::Union{Figure, Tuple{Figure, Axis}}
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

    if return_axis
        fig, ax = plot
        save(fig_path(fig_name), fig)
        return plot
    else
        save(fig_path(fig_name), plot)
        return plot
    end
end
