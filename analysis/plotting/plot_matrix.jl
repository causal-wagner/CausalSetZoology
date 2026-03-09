using CairoMakie
using LaTeXStrings
using PlotUtils
using Random

"""
    plot_hist_or_vec_panel!(
        ax,
        data,
        comp_mean_std,
        xlim_i,
        ylim_i,
        xlabel_i,
        ylabel_i,
        xticks_i,
        yticks_i;
        logscale_y,
        invert_color_scaling,
        plot_std,
        vmin,
        denom,
        colormap,
        comp_color,
        comp_linewidth,
    )

Render one histogram/vector panel with optional comparison band and line.

# Arguments
- `ax`: Target `CairoMakie.Axis`.
- `data`: Vector of `(scalar, mean, std)` tuples.
- `comp_mean_std`: Optional `(mean, std)` reference tuple.
- `xlim_i`: Optional x-axis limits.
- `ylim_i`: Optional y-axis limits.
- `xlabel_i`: Optional x-axis label.
- `ylabel_i`: Optional y-axis label.
- `xticks_i`: Optional x-tick specification.
- `yticks_i`: Optional y-tick specification.

# Keyword Arguments
- `logscale_y`: Use log-safe clipping for y values.
- `invert_color_scaling`: Reverse color traversal over scalar values.
- `plot_std`: Draw mean±std bands.
- `vmin`: Minimum scalar for color normalization.
- `denom`: Color normalization denominator.
- `colormap`: Colormap used for line/band colors.
- `comp_color`: Color used for comparison curves.
- `comp_linewidth`: Optional linewidth for comparison line.

# Returns
- `nothing`

# Throws
- `ArgumentError`: Raised when per-curve mean/std lengths mismatch.
"""
function plot_hist_or_vec_panel!(
    ax,
    data::AbstractVector{<:Tuple},
    comp_mean_std,
    xlim_i,
    ylim_i,
    xlabel_i,
    ylabel_i,
    xticks_i,
    yticks_i;
    logscale_y::Bool,
    invert_color_scaling::Bool,
    plot_std::Bool,
    vmin::Real,
    denom::Real,
    colormap,
    comp_color,
    comp_linewidth::Union{Nothing,Real},
)
    xlabel_i !== nothing && (ax.xlabel = xlabel_i)
    ylabel_i !== nothing && (ax.ylabel = ylabel_i)
    xlim_i !== nothing && CairoMakie.xlims!(ax, xlim_i...)
    ylim_i !== nothing && CairoMakie.ylims!(ax, ylim_i...)
    if xticks_i !== nothing
        ax.xticks = ([t[1] for t in xticks_i], [t[2] for t in xticks_i])
    end
    if yticks_i !== nothing
        ax.yticks = ([t[1] for t in yticks_i], [t[2] for t in yticks_i])
    end

    eps = if logscale_y
        if ylim_i !== nothing
            ylim_i[1] * 1e-3
        else
            minpos = minimum(v for (_, m, _) in data for v in m if v > 0)
            minpos * 1e-3
        end
    else
        -Inf
    end

    comp_x = nothing
    comp_mean = nothing
    if comp_mean_std !== nothing
        mean_comp, std_comp = comp_mean_std
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
        CairoMakie.band!(ax, x, ylo, yhi; color = (comp_color, 0.2))
        comp_x = x
        comp_mean = mean_comp
    end

    iter = invert_color_scaling ? reverse(data) : data
    for (val, mean, std) in iter
        if !(length(mean) == length(std))
            throw(ArgumentError("each curve requires mean and std vectors of equal length"))
        end
        t = (val - vmin) / denom
        t = invert_color_scaling ? (1 - t) : t
        color = PlotUtils.get(CairoMakie.cgrad(colormap), t)

        x = collect(1:length(mean))
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

        if plot_std
            CairoMakie.band!(ax, x, ylo, yhi; color = (color, 0.2))
        end
        CairoMakie.lines!(ax, x, mean; color = color)
    end

    if comp_x !== nothing
        isnothing(comp_linewidth) ? CairoMakie.lines!(ax, comp_x, comp_mean; color = comp_color) :
            CairoMakie.lines!(ax, comp_x, comp_mean; color = comp_color, linewidth = comp_linewidth)
    end
    return nothing
end

"""
    avg_hist_or_vec(data_group; num_bins = nothing)

Average one histogram/scalar or vector/scalar data group to `(scalar, mean, std)` tuples.

# Arguments
- `data_group`: A group of either histogram/scalar tuples or vector/scalar tuples.

# Keyword Arguments
- `num_bins`: Optional bin count used for vector averaging.

# Returns
- `Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}`: Averaged curves with uncertainty.
"""
function avg_hist_or_vec(data_group; num_bins::Union{Nothing,Int} = nothing)
    if isempty(data_group)
        return Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}()
    end
    if data_group[1][1] isa AbstractDict
        return average_histogram_with_std(data_group)
    end
    return average_vectors_with_std(data_group; num_bins = num_bins)
end

"""
    comp_avg_hist_or_vec(data_group; num_bins = nothing)

Average a comparison group to `(mean, std)` or return `nothing` for empty input.

# Arguments
- `data_group`: Comparison group of histogram/scalar or vector/scalar tuples.

# Keyword Arguments
- `num_bins`: Optional bin count used for vector averaging.

# Returns
- `Tuple{Vector{Float64},Vector{Float64}}` or `nothing`: Averaged comparison curve.
"""
function comp_avg_hist_or_vec(data_group; num_bins::Union{Nothing,Int} = nothing)
    if isempty(data_group)
        return nothing
    end
    first_entry = data_group[1]
    if first_entry isa Tuple
        vals = getindex.(data_group, 1)
        if vals[1] isa AbstractDict
            return average_histogram_with_std(vals)
        end
        if vals[1] isa AbstractVector
            return average_vectors_with_std(vals)
        end
        return nothing
    end
    if first_entry isa AbstractDict
        return average_histogram_with_std(data_group)
    end
    if first_entry isa AbstractVector
        return average_vectors_with_std(data_group)
    end
    return nothing
end

"""
    hist_hist_vec_hist_plot_matrix(
        data::Tuple{
            Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
            Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
            Vector{Vector{Tuple{Vector{Float64}, Float64}}},
            Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}}
        },
        fig_path::String;
        xlim, ylim, xlabel, ylabel,
        num_bins = nothing,
        colormap = :viridis,
        invert_color_scaling = false,
        colorbar_label = nothing,
        colorbar_ticks = nothing,
        colorbar_pos = nothing,
        colorbar_size = nothing,
        logscale_x = true,
        logscale_y = true,
        double_column = false,
        magnification = 1.0,
        plot_std = true,
        return_axis = false,
    )

Create a 2×2 matrix plot. The 1st, 2nd, and 4th entries are histogram+scalar
datasets; the 3rd is a vector+scalar dataset. A single colorbar is placed to
the right, spanning both rows. The color scaling is shared across all panels.

Returns `fig` or `(fig, axs)` when `return_axis=true`.

# Arguments
- `data`: Tuple `(h1, h2, v3, h4)` with three histogram+scalar groups and one vector+scalar group.
- `fig_path`: Output path passed to `CairoMakie.save`.

# Keyword Arguments
- `xlim`, `ylim`, `xlabel`, `ylabel`: Per-panel vectors of length 4.
- `num_bins`: Optional common bin count used when averaging vector/histogram data.
- `colormap`: Colormap used for scalar-conditioned curve coloring.
- `invert_color_scaling`: Reverse scalar-to-color mapping.
- `colorbar_label`, `colorbar_ticks`: Optional colorbar label and tick mapping.
- `colorbar_pos`, `colorbar_size`: Optional manual colorbar placement/alignment sizing.
- `colorbar_side`: Side for automatic colorbar placement (`:left`, `:right`, `:top`, `:bottom`).
- `colorbar_label_pos`: Colorbar label placement (`:side` or `:top`).
- `comp`: Optional tuple of comparison datasets for panel overlays.
- `comp_color`, `comp_linewidth`: Styling for comparison overlays.
- `xticks`, `yticks`: Optional per-panel tick specifications (length 4 when provided).
- `logscale_x`, `logscale_y`: Axis scaling toggles.
- `double_column`, `magnification`: Theme/layout controls.
- `plot_std`: Draw mean ± std bands.
- `right_yaxis`, `top_xaxis`: Move right-column y-axes / top-row x-axes.
- `rowgap`, `colgap`: Optional layout gap overrides.
- `return_axis`: Return `(fig, axs)` instead of only `fig`.

# Returns
- `fig`: Rendered figure, or `(fig, axs)` when `return_axis=true`.

# Throws
- `ArgumentError`: Raised when structural inputs are inconsistent.
- `DomainError`: Raised when numeric parameters violate domain constraints."""
function hist_hist_vec_hist_plot_matrix(
    data::Tuple{
        Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
        Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
        Vector{Vector{Tuple{Vector{Float64}, Float64}}},
        Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
    },
    fig_path::String;
    xlim::AbstractVector{<:Union{Tuple{Float64,Float64},Nothing}},
    ylim::AbstractVector{<:Union{Tuple{Float64,Float64},Nothing}},
    xlabel::AbstractVector{<:Union{AbstractString,LaTeXStrings.LaTeXString,Nothing}},
    ylabel::AbstractVector{<:Union{AbstractString,LaTeXStrings.LaTeXString,Nothing}},
    num_bins::Union{Nothing,Int} = nothing,
    colormap = :viridis,
    invert_color_scaling::Bool = false,
    colorbar_label::Union{Nothing,AbstractString,LaTeXStrings.LaTeXString} = nothing,
    colorbar_ticks::Union{Nothing,Vector{<:Tuple{<:Real,Any}}} = nothing,
    colorbar_pos::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    colorbar_size::Union{Nothing,Tuple{Real,Real}} = nothing,
    colorbar_side::Symbol = :right,
    colorbar_label_pos::Symbol = :side,
    comp::Union{Nothing,Tuple{AbstractVector,AbstractVector,AbstractVector,AbstractVector}} = nothing,
    comp_color = :black,
    comp_linewidth::Union{Nothing,Real} = 2,
    xticks::Union{Nothing,AbstractVector} = nothing,
    yticks::Union{Nothing,AbstractVector} = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    double_column::Bool = false,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 0.0,
    colgap::Union{Nothing,Real} = 0.0,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, Vector{CairoMakie.Axis}}}
    if !(length(xlim) == 4)
        throw(ArgumentError("xlim must have length 4"))
    end
    if !(length(ylim) == 4)
        throw(ArgumentError("ylim must have length 4"))
    end
    if !(length(xlabel) == 4)
        throw(ArgumentError("xlabel must have length 4"))
    end
    if !(length(ylabel) == 4)
        throw(ArgumentError("ylabel must have length 4"))
    end
    if !(magnification > 0)
        throw(DomainError(magnification, "magnification must be > 0"))
    end
    if rowgap !== nothing && !(rowgap >= 0)
        throw(DomainError(rowgap, "rowgap must be >= 0 when provided"))
    end
    if colgap !== nothing && !(colgap >= 0)
        throw(DomainError(colgap, "colgap must be >= 0 when provided"))
    end
    if colorbar_size !== nothing && !(colorbar_size[1] > 0 && colorbar_size[2] > 0)
        throw(DomainError(colorbar_size, "colorbar_size entries must be > 0"))
    end
    h1, h2, v3, h4 = data

    avg1 = [avg_hist_or_vec(h1[i]; num_bins = num_bins) for i in 1:length(h1)]
    avg2 = [avg_hist_or_vec(h2[i]; num_bins = num_bins) for i in 1:length(h2)]
    avg3 = [avg_hist_or_vec(v3[i]; num_bins = num_bins) for i in 1:length(v3)]
    avg4 = [avg_hist_or_vec(h4[i]; num_bins = num_bins) for i in 1:length(h4)]

    d1 = length(avg1) == 1 ? avg1[1] : vcat(avg1...)
    d2 = length(avg2) == 1 ? avg2[1] : vcat(avg2...)
    d3 = length(avg3) == 1 ? avg3[1] : vcat(avg3...)
    d4 = length(avg4) == 1 ? avg4[1] : vcat(avg4...)

    all_vals = [v for (v, _, _) in d1]
    append!(all_vals, (v for (v, _, _) in d2))
    append!(all_vals, (v for (v, _, _) in d3))
    append!(all_vals, (v for (v, _, _) in d4))
    vmin = isempty(all_vals) ? 0.0 : minimum(all_vals)
    vmax = isempty(all_vals) ? 1.0 : maximum(all_vals)
    denom = vmax == vmin ? 1.0 : (vmax - vmin)

    figsize = 2 .* apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
    )

    fig = CairoMakie.Figure(size = figsize)
    if rowgap !== nothing
        fig.layout.default_rowgap = CairoMakie.Fixed(rowgap)
    end
    if colgap !== nothing
        fig.layout.default_colgap = CairoMakie.Fixed(colgap)
    end
    axs = [
        CairoMakie.Axis(fig[1,1]; xscale = logscale_x ? log10 : identity, yscale = logscale_y ? log10 : identity),
        CairoMakie.Axis(fig[1,2]; xscale = logscale_x ? log10 : identity, yscale = logscale_y ? log10 : identity),
        CairoMakie.Axis(fig[2,1]; xscale = logscale_x ? log10 : identity, yscale = logscale_y ? log10 : identity),
        CairoMakie.Axis(fig[2,2]; xscale = logscale_x ? log10 : identity, yscale = logscale_y ? log10 : identity),
    ]
    if top_xaxis
        for ax in (axs[1], axs[2])
            ax.xaxisposition = :top
            ax.xticklabelalign = (:center, :bottom)
        end
    end
    if right_yaxis
        for ax in (axs[2], axs[4])
            ax.yaxisposition = :right
            ax.yticklabelalign = (:left, :center)
            ax.flip_ylabel = true
        end
        # add dummy left ticks on right-hand plots
        for (ax, cell) in zip((axs[2], axs[4]), (fig[1,2], fig[2,2]))
            ax_left = CairoMakie.Axis(cell;
                xscale = logscale_x ? log10 : identity,
                yscale = logscale_y ? log10 : identity,
                xlabelvisible = false,
                xticksvisible = false,
                xticklabelsvisible = false,
                xgridvisible = false,
                xminorgridvisible = false,
                yticklabelsvisible = false,
                ygridvisible = false,
                yminorgridvisible = false,
                backgroundcolor = :transparent,
            )
            ax_left.yaxisposition = :left
            ax_left.rightspinevisible = false
            CairoMakie.linkxaxes!(ax, ax_left)
            CairoMakie.linkyaxes!(ax, ax_left)
        end
    end

    comp1 = nothing
    comp2 = nothing
    comp3 = nothing
    comp4 = nothing
    if comp !== nothing
        comp1_flat = vcat(comp[1]...)
        comp2_flat = vcat(comp[2]...)
        comp3_flat = vcat(comp[3]...)
        comp4_flat = vcat(comp[4]...)
        comp1 = comp_avg_hist_or_vec(comp1_flat; num_bins = num_bins)
        comp2 = comp_avg_hist_or_vec(comp2_flat; num_bins = num_bins)
        comp3 = comp_avg_hist_or_vec(comp3_flat; num_bins = num_bins)
        comp4 = comp_avg_hist_or_vec(comp4_flat; num_bins = num_bins)
    end

    xt = xticks === nothing ? fill(nothing, 4) : xticks
    yt = yticks === nothing ? fill(nothing, 4) : yticks
    if !(length(xt) == 4)
        throw(ArgumentError("xticks must have length 4"))
    end
    if !(length(yt) == 4)
        throw(ArgumentError("yticks must have length 4"))
    end
    plot_hist_or_vec_panel!(axs[1], d1, comp1, xlim[1], ylim[1], xlabel[1], ylabel[1], xt[1], yt[1];
        logscale_y = logscale_y, invert_color_scaling = invert_color_scaling, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[2], d2, comp2, xlim[2], ylim[2], xlabel[2], ylabel[2], xt[2], yt[2];
        logscale_y = logscale_y, invert_color_scaling = invert_color_scaling, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[3], d3, comp3, xlim[3], ylim[3], xlabel[3], ylabel[3], xt[3], yt[3];
        logscale_y = logscale_y, invert_color_scaling = invert_color_scaling, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[4], d4, comp4, xlim[4], ylim[4], xlabel[4], ylabel[4], xt[4], yt[4];
        logscale_y = logscale_y, invert_color_scaling = invert_color_scaling, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)

    cb_cmap = invert_color_scaling ? CairoMakie.Reverse(colormap) : colormap
    cb = if colorbar_pos === nothing
        if colorbar_side == :right
            CairoMakie.Colorbar(fig[1:2, 3], limits = (vmin, vmax), colormap = cb_cmap)
        elseif colorbar_side == :left
            CairoMakie.Colorbar(fig[1:2, 0], limits = (vmin, vmax), colormap = cb_cmap)
        elseif colorbar_side == :top
            CairoMakie.Colorbar(fig[0, 1:2], limits = (vmin, vmax), colormap = cb_cmap, vertical = false)
        elseif colorbar_side == :bottom
            CairoMakie.Colorbar(fig[3, 1:2], limits = (vmin, vmax), colormap = cb_cmap, vertical = false)
        else
            throw(ArgumentError("colorbar_side must be one of :left, :right, :top, :bottom"))
        end
    else
        cb = CairoMakie.Colorbar(fig[1:2, 3], limits = (vmin, vmax), colormap = cb_cmap)
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
        if !(colorbar_label_pos in (:side, :top))
            throw(ArgumentError("colorbar_label_pos must be :side or :top"))
        end
        if colorbar_label_pos == :top
            cb.label = ""
            if colorbar_side == :left
                lbl = CairoMakie.Label(fig[1:2, 0], colorbar_label; tellwidth = false, tellheight = false)
                lbl.halign = 0.5
                lbl.valign = 1.055
            elseif colorbar_side == :right
                lbl = CairoMakie.Label(fig[1:2, 3], colorbar_label; tellwidth = false, tellheight = false)
                lbl.halign = 0.5
                lbl.valign = 1.055
            else
                cb.label = colorbar_label
            end
        else
            if colorbar_side == :left
                cb.label = ""
                # put label in its own column to the left of the colorbar
                lbl = CairoMakie.Label(
                    fig[1:2, -1],
                    colorbar_label;
                    tellwidth = true,
                    tellheight = false,
                )
                lbl.halign = 0.5
                lbl.valign = 0.5
            else
                cb.label = colorbar_label
            end
        end
    end
    if colorbar_ticks !== nothing
        cb.ticks = ([t[1] for t in colorbar_ticks], [t[2] for t in colorbar_ticks])
    end

    CairoMakie.save(fig_path, fig)
    return return_axis ? (fig, axs) : fig
end

"""
    hist_hist_vec_distinguishability_plot_matrix(
        data_paths,
        comp_paths,
        scalar::Symbol,
        fig_path::String;
        xlim, ylim, xlabel, ylabel,
        sqrt_scalars::Bool = false,
        # scalar_bin_mahalanobis_gap_distinguishability kwargs
        num_bins = nothing,
        regulator = 0.0,
        R = 1000,
        q = 0.0,
        alpha = 0.05,
        rng = Random.default_rng(),
        symmetric = false,
        projection_tolerance = 1e-10,
        progress = false,
        # plot kwargs (aligned with hist_hist_vec_hist_plot_matrix)
        colormap = :viridis,
        invert_color_scaling = false,
        colorbar_label = nothing,
        colorbar_ticks = nothing,
        colorbar_pos = nothing,
        colorbar_size = nothing,
        colorbar_side = :right,
        colorbar_label_pos = :side,
        comp_color = :black,
        comp_linewidth = 2,
        xticks = nothing,
        yticks = nothing,
        logscale_x = true,
        logscale_y = true,
        double_column = false,
        magnification = 1.0,
        plot_std = true,
        right_yaxis = true,
        top_xaxis = true,
        rowgap = 0.0,
        colgap = 0.0,
    )::CairoMakie.Figure

See `hist_hist_vec_hist_plot_matrix(...)`.

This variant replaces the fourth panel with a distinguishability summary.

Panels (in order):
1. connectivity link histograms
2. max path length histograms
3. ev_sym_link vectors
4. `M_obs` vs scalar from `scalar_bin_mahalanobis_gap_distinguishability` for the
   three observables above (computed with `symmetric = false`).

Returns the saved `CairoMakie.Figure`.

# Arguments
- `data_paths`: Path or collection of paths used for loading/saving data.
- `comp_paths`: Path or collection of paths used for loading/saving data.
- `scalar`: Scalar field used for x/color conditioning.
- `fig_path`: Output path passed to `CairoMakie.save`.

# Keyword Arguments
- `xlim`, `ylim`, `xlabel`, `ylabel`: Per-panel vectors of length 4.
- `sqrt_scalars`: Apply `sqrt` to dataset scalar values before plotting/distinguishability.
- `num_bins`, `regulator`, `R`, `q`, `alpha`: Distinguishability estimator controls.
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: Must remain `false` for this function.
    - `projection_tolerance`, `progress`: Distinguishability runtime controls.
- `invert_color_scaling`: Reverse scalar-to-color mapping.
- `colorbar_label`, `colorbar_ticks`: Optional colorbar label and tick mapping.
- `colorbar_pos`, `colorbar_size`: Optional manual colorbar placement/alignment sizing.
- `colorbar_side`: Side for automatic colorbar placement (`:left`, `:right`, `:top`, `:bottom`).
- `colorbar_label_pos`: Colorbar label placement (`:side` or `:top`).
- `comp_color`, `comp_linewidth`: Styling for comparison overlays in panels 1-3.
- `xticks`, `yticks`: Optional per-panel tick specifications (length 4 when provided).
- `logscale_x`: Toggle for logarithmic axis scaling.
- `logscale_y`: Toggle for logarithmic axis scaling.
- `double_column`, `magnification`: Theme/layout controls.
- `plot_std`: Draw mean ± std bands in panels 1-3.
- `right_yaxis`, `top_xaxis`: Move right-column y-axes / top-row x-axes.
- `rowgap`, `colgap`: Optional layout gap overrides.

# Returns
- `fig::CairoMakie.Figure`: Saved and returned plot figure.

# Throws
- `ArgumentError`: Raised when structural inputs are inconsistent.
- `DomainError`: Raised when numeric parameters violate domain constraints."""
function hist_hist_vec_distinguishability_plot_matrix(
    data_paths_in,
    comp_paths_in,
    scalar::Symbol,
    fig_path::String;
    xlim::AbstractVector{<:Union{Tuple{Float64,Float64},Nothing}},
    ylim::AbstractVector{<:Union{Tuple{Float64,Float64},Nothing}},
    xlabel::AbstractVector{<:Union{AbstractString,LaTeXStrings.LaTeXString,Nothing}},
    ylabel::AbstractVector{<:Union{AbstractString,LaTeXStrings.LaTeXString,Nothing}},
    sqrt_scalars::Bool = false,
    # scalar_bin_mahalanobis_gap_distinguishability kwargs
    num_bins::Union{Nothing,Int} = nothing,
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    symmetric::Bool = false,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    # plot kwargs
    colormap = :viridis,
    invert_color_scaling::Bool = false,
    colorbar_label::Union{Nothing,AbstractString,LaTeXStrings.LaTeXString} = nothing,
    colorbar_ticks::Union{Nothing,Vector{<:Tuple{<:Real,Any}}} = nothing,
    colorbar_pos::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    colorbar_size::Union{Nothing,Tuple{Real,Real}} = nothing,
    colorbar_side::Symbol = :right,
    colorbar_label_pos::Symbol = :side,
    comp_color = :black,
    comp_linewidth::Union{Nothing,Real} = 2,
    xticks::Union{Nothing,AbstractVector} = nothing,
    yticks::Union{Nothing,AbstractVector} = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    double_column::Bool = false,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 0.0,
    colgap::Union{Nothing,Real} = 0.0,
)::CairoMakie.Figure
    if !(length(xlim) == 4)
        throw(ArgumentError("xlim must have length 4"))
    end
    if !(length(ylim) == 4)
        throw(ArgumentError("ylim must have length 4"))
    end
    if !(length(xlabel) == 4)
        throw(ArgumentError("xlabel must have length 4"))
    end
    if !(length(ylabel) == 4)
        throw(ArgumentError("ylabel must have length 4"))
    end
    if symmetric
        throw(ArgumentError("symmetric must be false for hist_hist_vec_distinguishability_plot_matrix"))
    end
    if !(magnification > 0)
        throw(DomainError(magnification, "magnification must be > 0"))
    end
    if rowgap !== nothing && !(rowgap >= 0)
        throw(DomainError(rowgap, "rowgap must be >= 0 when provided"))
    end
    if colgap !== nothing && !(colgap >= 0)
        throw(DomainError(colgap, "colgap must be >= 0 when provided"))
    end
    if colorbar_size !== nothing && !(colorbar_size[1] > 0 && colorbar_size[2] > 0)
        throw(DomainError(colorbar_size, "colorbar_size entries must be > 0"))
    end
    to_paths(x) = x isa AbstractString ? [String(x)] : String.(collect(x))
    data_paths = to_paths(data_paths_in)
    comp_paths = to_paths(comp_paths_in)

    # dataset
    in_degree_link_hists_data = load_histograms_from_paths(data_paths, :in_degree_hist_link, scalar)
    out_degree_link_hists_data = load_histograms_from_paths(data_paths, :out_degree_hist_link, scalar)
    connectivity_link_hists_data = join_histograms([in_degree_link_hists_data, out_degree_link_hists_data])
    ev_sym_link_data = load_field_with_scalar(data_paths, :ev_sym_link, scalar)
    max_pathlen_hists_data = load_histograms_from_paths(data_paths, :max_pathlen_hist, scalar)

    # comparison dataset
    in_degree_link_hists_comp = load_histograms_from_paths(comp_paths, :in_degree_hist_link, scalar)
    out_degree_link_hists_comp = load_histograms_from_paths(comp_paths, :out_degree_hist_link, scalar)
    connectivity_link_hists_comp = join_histograms([in_degree_link_hists_comp, out_degree_link_hists_comp])
    ev_sym_link_comp = load_field_with_scalar(comp_paths, :ev_sym_link, scalar)
    max_pathlen_hists_comp = load_histograms_from_paths(comp_paths, :max_pathlen_hist, scalar)

    # normalization as requested
    normalized_connectivity_data = normalize_hists(connectivity_link_hists_data; num_bins = num_bins)
    normalized_connectivity_comp = normalize_hists(connectivity_link_hists_comp; num_bins = num_bins)
    normalized_max_pathlen_data = normalize_hists(max_pathlen_hists_data; normalization = 1, num_bins = num_bins)
    normalized_max_pathlen_comp = normalize_hists(max_pathlen_hists_comp; normalization = 1, num_bins = num_bins)

    # optionally transform only dataset scalars
    maybe_sqrt_pairs(pairs) = sqrt_scalars ? [(v, sqrt(Float64(s))) for (v, s) in pairs] : pairs
    normalized_connectivity_data = [maybe_sqrt_pairs(g) for g in normalized_connectivity_data]
    normalized_max_pathlen_data = [maybe_sqrt_pairs(g) for g in normalized_max_pathlen_data]
    ev_sym_link_data = [maybe_sqrt_pairs(g) for g in ev_sym_link_data]

    # scalar_bin_mahalanobis expects one dataset (one path)
    to_single_dataset(groups) = [length(groups) == 1 ? groups[1] : vcat(groups...)]
    to_reference(groups) = length(groups) == 1 ? groups[1] : vcat(groups...)

    connectivity_data_single = to_single_dataset(normalized_connectivity_data)
    max_pathlen_data_single = to_single_dataset(normalized_max_pathlen_data)
    ev_sym_link_data_single = to_single_dataset(ev_sym_link_data)

    connectivity_ref = to_reference(normalized_connectivity_comp)
    max_pathlen_ref = to_reference(normalized_max_pathlen_comp)
    ev_sym_link_ref = to_reference(ev_sym_link_comp)

    # distinguishability (symmetric = false by design)
    conn_dist = scalar_bin_mahalanobis_gap_distinguishability(
        connectivity_data_single,
        connectivity_ref;
        num_bins = num_bins,
        regulator = regulator,
        R = R,
        q = q,
        alpha = alpha,
        rng = rng,
        symmetric = false,
        projection_tolerance = projection_tolerance,
        progress = progress,
    )
    max_path_dist = scalar_bin_mahalanobis_gap_distinguishability(
        max_pathlen_data_single,
        max_pathlen_ref;
        num_bins = num_bins,
        regulator = regulator,
        R = R,
        q = q,
        alpha = alpha,
        rng = rng,
        symmetric = false,
        projection_tolerance = projection_tolerance,
        progress = progress,
    )
    ev_sym_dist = scalar_bin_mahalanobis_gap_distinguishability(
        ev_sym_link_data_single,
        ev_sym_link_ref;
        num_bins = num_bins,
        regulator = regulator,
        R = R,
        q = q,
        alpha = alpha,
        rng = rng,
        symmetric = false,
        projection_tolerance = projection_tolerance,
        progress = progress,
    )

    # prepare panel data (first 3 panels like hist_hist_vec_hist_plot_matrix)
    d1 = avg_hist_or_vec(connectivity_data_single[1]; num_bins = num_bins)
    d2 = avg_hist_or_vec(max_pathlen_data_single[1]; num_bins = num_bins)
    d3 = avg_hist_or_vec(ev_sym_link_data_single[1]; num_bins = num_bins)
    comp1 = comp_avg_hist_or_vec(connectivity_ref; num_bins = num_bins)
    comp2 = comp_avg_hist_or_vec(max_pathlen_ref; num_bins = num_bins)
    comp3 = comp_avg_hist_or_vec(ev_sym_link_ref; num_bins = num_bins)

    all_vals = [v for (v, _, _) in d1]
    append!(all_vals, (v for (v, _, _) in d2))
    append!(all_vals, (v for (v, _, _) in d3))
    vmin = isempty(all_vals) ? 0.0 : minimum(all_vals)
    vmax = isempty(all_vals) ? 1.0 : maximum(all_vals)
    denom = vmax == vmin ? 1.0 : (vmax - vmin)

    figsize = 2 .* apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
    )

    fig = CairoMakie.Figure(size = figsize)
    if rowgap !== nothing
        fig.layout.default_rowgap = CairoMakie.Fixed(rowgap)
    end
    if colgap !== nothing
        fig.layout.default_colgap = CairoMakie.Fixed(colgap)
    end
    axs = [
        CairoMakie.Axis(fig[1,1]; xscale = logscale_x ? log10 : identity, yscale = logscale_y ? log10 : identity),
        CairoMakie.Axis(fig[1,2]; xscale = logscale_x ? log10 : identity, yscale = logscale_y ? log10 : identity),
        CairoMakie.Axis(fig[2,1]; xscale = logscale_x ? log10 : identity, yscale = logscale_y ? log10 : identity),
        CairoMakie.Axis(fig[2,2]; xscale = logscale_x ? log10 : identity, yscale = logscale_y ? log10 : identity),
    ]

    if top_xaxis
        for ax in (axs[1], axs[2])
            ax.xaxisposition = :top
            ax.xticklabelalign = (:center, :bottom)
        end
    end
    if right_yaxis
        for ax in (axs[2], axs[4])
            ax.yaxisposition = :right
            ax.yticklabelalign = (:left, :center)
            ax.flip_ylabel = true
        end
        for (ax, cell) in zip((axs[2], axs[4]), (fig[1,2], fig[2,2]))
            ax_left = CairoMakie.Axis(cell;
                xscale = logscale_x ? log10 : identity,
                yscale = logscale_y ? log10 : identity,
                xlabelvisible = false,
                xticksvisible = false,
                xticklabelsvisible = false,
                xgridvisible = false,
                xminorgridvisible = false,
                yticklabelsvisible = false,
                ygridvisible = false,
                yminorgridvisible = false,
                backgroundcolor = :transparent,
            )
            ax_left.yaxisposition = :left
            ax_left.rightspinevisible = false
            CairoMakie.linkxaxes!(ax, ax_left)
            CairoMakie.linkyaxes!(ax, ax_left)
        end
    end

    xt = xticks === nothing ? fill(nothing, 4) : xticks
    yt = yticks === nothing ? fill(nothing, 4) : yticks
    if !(length(xt) == 4)
        throw(ArgumentError("xticks must have length 4"))
    end
    if !(length(yt) == 4)
        throw(ArgumentError("yticks must have length 4"))
    end
    plot_hist_or_vec_panel!(axs[1], d1, comp1, xlim[1], ylim[1], xlabel[1], ylabel[1], xt[1], yt[1];
        logscale_y = logscale_y, invert_color_scaling = invert_color_scaling, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[2], d2, comp2, xlim[2], ylim[2], xlabel[2], ylabel[2], xt[2], yt[2];
        logscale_y = logscale_y, invert_color_scaling = invert_color_scaling, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[3], d3, comp3, xlim[3], ylim[3], xlabel[3], ylabel[3], xt[3], yt[3];
        logscale_y = logscale_y, invert_color_scaling = invert_color_scaling, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)

    # panel 4: M_obs vs scalar for the three observables (default cycle colors)
    ax4 = axs[4]
    xlabel[4] !== nothing && (ax4.xlabel = xlabel[4])
    ylabel[4] !== nothing && (ax4.ylabel = ylabel[4])
    xlim[4] !== nothing && CairoMakie.xlims!(ax4, xlim[4]...)
    ylim[4] !== nothing && CairoMakie.ylims!(ax4, ylim[4]...)
    if xt[4] !== nothing
        ax4.xticks = ([t[1] for t in xt[4]], [t[2] for t in xt[4]])
    end
    if yt[4] !== nothing
        ax4.yticks = ([t[1] for t in yt[4]], [t[2] for t in yt[4]])
    end

    conn_x = [r.scalar for r in conn_dist]
    conn_y = [r.M_obs for r in conn_dist]
    max_x = [r.scalar for r in max_path_dist]
    max_y = [r.M_obs for r in max_path_dist]
    ev_x = [r.scalar for r in ev_sym_dist]
    ev_y = [r.M_obs for r in ev_sym_dist]

    CairoMakie.lines!(ax4, conn_x, conn_y, label = "connectivity_link")
    CairoMakie.lines!(ax4, max_x, max_y, label = "max_pathlen")
    CairoMakie.lines!(ax4, ev_x, ev_y, label = "ev_sym_link")
    CairoMakie.axislegend(ax4)

    cb_cmap = invert_color_scaling ? CairoMakie.Reverse(colormap) : colormap
    cb = if colorbar_pos === nothing
        if colorbar_side == :right
            CairoMakie.Colorbar(fig[1:2, 3], limits = (vmin, vmax), colormap = cb_cmap)
        elseif colorbar_side == :left
            CairoMakie.Colorbar(fig[1:2, 0], limits = (vmin, vmax), colormap = cb_cmap)
        elseif colorbar_side == :top
            CairoMakie.Colorbar(fig[0, 1:2], limits = (vmin, vmax), colormap = cb_cmap, vertical = false)
        elseif colorbar_side == :bottom
            CairoMakie.Colorbar(fig[3, 1:2], limits = (vmin, vmax), colormap = cb_cmap, vertical = false)
        else
            throw(ArgumentError("colorbar_side must be one of :left, :right, :top, :bottom"))
        end
    else
        cb = CairoMakie.Colorbar(fig[1:2, 3], limits = (vmin, vmax), colormap = cb_cmap)
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
        if !(colorbar_label_pos in (:side, :top))
            throw(ArgumentError("colorbar_label_pos must be :side or :top"))
        end
        if colorbar_label_pos == :top
            cb.label = ""
            if colorbar_side == :left
                lbl = CairoMakie.Label(fig[1:2, 0], colorbar_label; tellwidth = false, tellheight = false)
                lbl.halign = 0.5
                lbl.valign = 1.055
            elseif colorbar_side == :right
                lbl = CairoMakie.Label(fig[1:2, 3], colorbar_label; tellwidth = false, tellheight = false)
                lbl.halign = 0.5
                lbl.valign = 1.055
            else
                cb.label = colorbar_label
            end
        else
            if colorbar_side == :left
                cb.label = ""
                lbl = CairoMakie.Label(
                    fig[1:2, -1],
                    colorbar_label;
                    tellwidth = true,
                    tellheight = false,
                )
                lbl.halign = 0.5
                lbl.valign = 0.5
            else
                cb.label = colorbar_label
            end
        end
    end
    if colorbar_ticks !== nothing
        cb.ticks = ([t[1] for t in colorbar_ticks], [t[2] for t in colorbar_ticks])
    end

    CairoMakie.save(fig_path, fig)
    return fig
end
