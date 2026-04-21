using CairoMakie
using LaTeXStrings
using PlotUtils
using ProgressMeter
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
        invert_plot_order,
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
- `invert_plot_order`: Reverse the draw order of scalar-conditioned curves.
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
    invert_plot_order::Bool,
    log_color_scaling::Bool,
    plot_std::Bool,
    vmin::Real,
    denom::Real,
    colormap,
    comp_color,
    comp_linewidth::Union{Nothing,Real},
    xscale::Real = 1.0,
)
    apply_axis_metadata!(ax, xlim_i, ylim_i, xlabel_i, ylabel_i, xticks_i, yticks_i)

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
        x = xscale .* collect(1:length(mean_comp))
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

    iter = invert_plot_order ? reverse(data) : data
    for (val, mean, std) in iter
        if !(length(mean) == length(std))
            throw(ArgumentError("each curve requires mean and std vectors of equal length"))
        end
        color_val = log_color_scaling ? log10(Float64(val)) : Float64(val)
        t = (color_val - vmin) / denom
        t = invert_color_scaling ? (1 - t) : t
        color = PlotUtils.get(CairoMakie.cgrad(colormap), t)

        x = xscale .* collect(1:length(mean))
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

function scale_mean_std(mean_std, yscale::Real)
    yscale == 1.0 && return mean_std
    mean, std = mean_std
    return (yscale .* mean, yscale .* std)
end

function scale_panel_series(data::AbstractVector{<:Tuple}, yscale::Real)
    yscale == 1.0 && return data
    return [(val, yscale .* mean, yscale .* std) for (val, mean, std) in data]
end

function group_scalar_pairs_for_plot(
    pairs::AbstractVector{<:Tuple{T,S}};
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    log_binning::Union{Nothing,Bool} = nothing,
) where {T,S<:Real}
    isempty(pairs) && return Vector{Tuple{Real,Vector{T}}}()
    if num_bins !== nothing && !(num_bins >= 1)
        throw(DomainError(num_bins, "num_bins must be >= 1"))
    end

    if num_bins === nothing && bin_edges === nothing
        groups = Dict{Real,Vector{T}}()
        for (v, s) in pairs
            get!(groups, s, T[])
            push!(groups[s], v)
        end
        keys_sorted = sort(collect(keys(groups)))
        return [(k, groups[k]) for k in keys_sorted]
    end

    edges = bin_edges
    if edges === nothing
        scalars = [Float64(s) for (_, s) in pairs]
        vmin, vmax = minimum(scalars), maximum(scalars)
        if log_binning
            if !(vmin > 0)
                throw(DomainError(vmin, "log_binning requires positive scalar values"))
            end
            edges = vmin == vmax ?
                [vmin, vmin * (1 + 1e-12)] :
                collect(10 .^ range(log10(vmin), log10(vmax); length = num_bins + 1))
        else
            edges = vmin == vmax ?
                [vmin, vmax + 1e-12] :
                collect(range(vmin, vmax; length = num_bins + 1))
        end
    end

    groups = Dict{Real,Vector{T}}()
    for (v, s_raw) in pairs
        s = Float64(s_raw)
        idx = searchsortedlast(edges, s)
        idx = clamp(idx, 1, length(edges) - 1)
        lo = Float64(edges[idx])
        hi = Float64(edges[idx + 1])
        key = log_binning ? sqrt(lo * hi) : (lo + hi) / 2
        get!(groups, key, T[])
        push!(groups[key], v)
    end

    keys_sorted = sort(collect(keys(groups)))
    return [(k, groups[k]) for k in keys_sorted]
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
function avg_hist_or_vec(
    data_group;
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    log_binning::Bool = false,
)
    if isempty(data_group)
        return Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}()
    end
    if data_group[1][1] isa AbstractDict
        if num_bins !== nothing || bin_edges !== nothing
            bins = group_scalar_pairs_for_plot(
                data_group;
                num_bins = num_bins,
                bin_edges = bin_edges,
                log_binning = log_binning,
            )
            return [(s, average_histogram_with_std(vals)...) for (s, vals) in bins]
        end
        return average_histogram_with_std(data_group)
    end
    if num_bins !== nothing || bin_edges !== nothing
        bins = group_scalar_pairs_for_plot(
            data_group;
            num_bins = num_bins,
            bin_edges = bin_edges,
            log_binning = log_binning,
        )
        return [(s, average_vectors_with_std(vals)...) for (s, vals) in bins]
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

function compute_plot_matrix_scalar_bin_edges(
    data::Tuple;
    num_bins::Union{Nothing,Int} = nothing,
    log_binning::Bool = false,
)
    if num_bins === nothing || !log_binning
        return nothing
    end
    scalars = Float64[]
    for panel_groups in data
        for group in panel_groups
            append!(scalars, (Float64(s) for (_, s) in group))
        end
    end
    isempty(scalars) && return nothing
    vmin, vmax = minimum(scalars), maximum(scalars)
    if !(vmin > 0)
        throw(DomainError(vmin, "log_binning requires positive scalar values"))
    end
    return vmin == vmax ?
        [vmin, vmin * (1 + 1e-12)] :
        collect(10 .^ range(log10(vmin), log10(vmax); length = num_bins + 1))
end

function validate_plot_matrix_inputs(
    xlim,
    ylim,
    xlabel,
    ylabel;
    magnification::Real,
    rowgap,
    colgap,
    colorbar_size,
)
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
    return nothing
end

function compute_color_scale(
    panel_data::AbstractVector{<:AbstractVector{<:Tuple}};
    log_color_scaling::Bool = false,
)
    all_vals = Float64[]
    for data in panel_data
        append!(all_vals, (Float64(v) for (v, _, _) in data))
    end
    if log_color_scaling && !isempty(all_vals) && !(minimum(all_vals) > 0)
        throw(DomainError(minimum(all_vals), "log color scaling requires positive scalar values"))
    end
    raw_vmin = isempty(all_vals) ? 0.0 : minimum(all_vals)
    raw_vmax = isempty(all_vals) ? 1.0 : maximum(all_vals)
    color_vals = log_color_scaling ? log10.(all_vals) : all_vals
    vmin = isempty(color_vals) ? 0.0 : minimum(color_vals)
    vmax = isempty(color_vals) ? 1.0 : maximum(color_vals)
    denom = vmax == vmin ? 1.0 : (vmax - vmin)
    return (; raw_vmin, raw_vmax, vmin, vmax, denom)
end

function normalize_panel_ticks(ticks, name::AbstractString)
    normalized = ticks === nothing ? fill(nothing, 4) : ticks
    if !(length(normalized) == 4)
        throw(ArgumentError("$name must have length 4"))
    end
    return normalized
end

function normalize_panel_bools(values::AbstractVector{Bool}, name::AbstractString)
    normalized = collect(values)
    if !(length(normalized) == 4)
        throw(ArgumentError("$name must have length 4"))
    end
    return normalized
end

function apply_axis_scale_theme!(ax; logscale_x::Bool, logscale_y::Bool)
    if logscale_x
        ax.xticks = logticks
        ax.xminorticks = logminorticks
    end
    if logscale_y
        ax.yticks = logticks
        ax.yminorticks = logminorticks
    end
    return nothing
end

function apply_axis_metadata!(ax, xlim_i, ylim_i, xlabel_i, ylabel_i, xticks_i, yticks_i)
    xlabel_i !== nothing && (ax.xlabel = xlabel_i)
    ylabel_i !== nothing && (ax.ylabel = ylabel_i)
    xlim_i !== nothing && CairoMakie.xlims!(ax, xlim_i...)
    ylim_i !== nothing && CairoMakie.ylims!(ax, ylim_i...)
    if xticks_i !== nothing
        if xticks_i isa Tuple && length(xticks_i) == 2
            ax.xticks = xticks_i
        elseif !isempty(xticks_i) && first(xticks_i) isa Tuple
            ax.xticks = ([t[1] for t in xticks_i], [t[2] for t in xticks_i])
        else
            ax.xticks = collect(xticks_i)
        end
    end
    if yticks_i !== nothing
        if yticks_i isa Tuple && length(yticks_i) == 2
            ax.yticks = yticks_i
        elseif !isempty(yticks_i) && first(yticks_i) isa Tuple
            ax.yticks = ([t[1] for t in yticks_i], [t[2] for t in yticks_i])
        else
            ax.yticks = collect(yticks_i)
        end
    end
    return nothing
end

function create_plot_matrix_figure_and_axes(;
    logscale_x::AbstractVector{Bool},
    logscale_y::AbstractVector{Bool},
    double_column::Bool,
    magnification::Real,
    top_xaxis::Bool,
    right_yaxis::Bool,
    rowgap,
    colgap,
)
    figsize = 2 .* apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = false,
        logscale_y = false,
    )

    fig = CairoMakie.Figure(size = figsize)
    if rowgap !== nothing
        fig.layout.default_rowgap = CairoMakie.Fixed(rowgap)
    end
    if colgap !== nothing
        fig.layout.default_colgap = CairoMakie.Fixed(colgap)
    end

    axs = [
        CairoMakie.Axis(fig[1, 1]; xscale = logscale_x[1] ? log10 : identity, yscale = logscale_y[1] ? log10 : identity),
        CairoMakie.Axis(fig[1, 2]; xscale = logscale_x[2] ? log10 : identity, yscale = logscale_y[2] ? log10 : identity),
        CairoMakie.Axis(fig[2, 1]; xscale = logscale_x[3] ? log10 : identity, yscale = logscale_y[3] ? log10 : identity),
        CairoMakie.Axis(fig[2, 2]; xscale = logscale_x[4] ? log10 : identity, yscale = logscale_y[4] ? log10 : identity),
    ]
    for i in eachindex(axs)
        apply_axis_scale_theme!(axs[i]; logscale_x = logscale_x[i], logscale_y = logscale_y[i])
    end

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
        for (ax, cell) in zip((axs[2], axs[4]), (fig[1, 2], fig[2, 2]))
            ax_left = CairoMakie.Axis(cell;
                xscale = ax.xscale[],
                yscale = ax.yscale[],
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
            apply_axis_scale_theme!(
                ax_left;
                logscale_x = ax.xscale[] !== identity,
                logscale_y = ax.yscale[] !== identity,
            )
            ax_left.yaxisposition = :left
            ax_left.rightspinevisible = false
            CairoMakie.linkxaxes!(ax, ax_left)
            CairoMakie.linkyaxes!(ax, ax_left)
        end
    end

    return fig, axs
end

function add_plot_matrix_colorbar!(
    fig,
    vmin,
    vmax;
    colormap,
    invert_color_scaling::Bool,
    log_color_scaling::Bool,
    colorbar_label,
    colorbar_ticks,
    colorbar_pos,
    colorbar_size,
    colorbar_side::Symbol,
    colorbar_label_pos::Symbol,
)
    cb_cmap = invert_color_scaling ? CairoMakie.Reverse(colormap) : colormap
    limits = log_color_scaling ? (10.0^vmin, 10.0^vmax) : (vmin, vmax)
    cb = if colorbar_pos === nothing
        if colorbar_side == :right
            CairoMakie.Colorbar(fig[1:2, 3], limits = limits, colormap = cb_cmap, scale = log_color_scaling ? log10 : identity)
        elseif colorbar_side == :left
            CairoMakie.Colorbar(fig[1:2, 0], limits = limits, colormap = cb_cmap, scale = log_color_scaling ? log10 : identity)
        elseif colorbar_side == :top
            CairoMakie.Colorbar(fig[0, 1:2], limits = limits, colormap = cb_cmap, vertical = false, scale = log_color_scaling ? log10 : identity)
        elseif colorbar_side == :bottom
            CairoMakie.Colorbar(fig[3, 1:2], limits = limits, colormap = cb_cmap, vertical = false, scale = log_color_scaling ? log10 : identity)
        else
            throw(ArgumentError("colorbar_side must be one of :left, :right, :top, :bottom"))
        end
    else
        positioned_cb = CairoMakie.Colorbar(fig[1:2, 3], limits = limits, colormap = cb_cmap, scale = log_color_scaling ? log10 : identity)
        positioned_cb.halign = colorbar_pos[1]
        positioned_cb.valign = colorbar_pos[2]
        positioned_cb.tellwidth = false
        positioned_cb.tellheight = false
        positioned_cb
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
    return cb
end

function load_observable_plot_matrix_observables(paths::Vector{String}, scalar::Symbol)
    fields = [:degree_hist_link, :ev_sym_link, :max_pathlen_hist, :cardinalities_hist]
    loaded = load_fields_from_paths(paths, fields, scalar)
    connectivity_link_hists = [dataset[1] for dataset in loaded]
    ev_sym_link = [dataset[2] for dataset in loaded]
    max_pathlen_hists = [dataset[3] for dataset in loaded]
    cardinalities_hists = [dataset[4] for dataset in loaded]
    return (; connectivity_link_hists, max_pathlen_hists, ev_sym_link, cardinalities_hists)
end

function load_observable_plot_matrix_observables(paths::Vector{String})
    fields = [:degree_hist_link, :ev_sym_link, :max_pathlen_hist, :cardinalities_hist]
    loaded = load_fields_from_paths(paths, fields)
    connectivity_link_hists = [dataset[1] for dataset in loaded]
    ev_sym_link = [dataset[2] for dataset in loaded]
    max_pathlen_hists = [dataset[3] for dataset in loaded]
    cardinalities_hists = [dataset[4] for dataset in loaded]
    return (; connectivity_link_hists, max_pathlen_hists, ev_sym_link, cardinalities_hists)
end

"""
    _observable_plot_matrix_from_data(
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

Render a 2×2 observable plot matrix from preloaded scalar-conditioned data.

Panel order:
1. connectivity_link histogram
2. max_pathlen histogram
3. ev_sym_link vector
4. cardinalities histogram

Returns `fig` or `(fig, axs)` when `return_axis=true`.

# Arguments
- `data`: Tuple `(connectivity_link, max_pathlen, ev_sym_link, cardinalities)`.
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
- `comp`: Optional tuple of comparison datasets in the same panel order.
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
function _observable_plot_matrix_from_data(
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
    logscale_x::AbstractVector{Bool} = fill(true, 4),
    logscale_y::AbstractVector{Bool} = fill(true, 4),
    double_column::Bool = true,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 0.0,
    colgap::Union{Nothing,Real} = 0.0,
    log_binning::Bool = false,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, Vector{CairoMakie.Axis}}}
    validate_plot_matrix_inputs(
        xlim,
        ylim,
        xlabel,
        ylabel;
        magnification = magnification,
        rowgap = rowgap,
        colgap = colgap,
        colorbar_size = colorbar_size,
    )
    logscale_x_panel = normalize_panel_bools(logscale_x, "logscale_x")
    logscale_y_panel = normalize_panel_bools(logscale_y, "logscale_y")
    h1, h2, v3, h4 = data
    scalar_bin_edges = compute_plot_matrix_scalar_bin_edges(
        data;
        num_bins = num_bins,
        log_binning = log_binning,
    )

    avg1 = [avg_hist_or_vec(h1[i]; num_bins = num_bins, bin_edges = scalar_bin_edges, log_binning = log_binning) for i in 1:length(h1)]
    avg2 = [avg_hist_or_vec(h2[i]; num_bins = num_bins, bin_edges = scalar_bin_edges, log_binning = log_binning) for i in 1:length(h2)]
    avg3 = [avg_hist_or_vec(v3[i]; num_bins = num_bins, bin_edges = scalar_bin_edges, log_binning = log_binning) for i in 1:length(v3)]
    avg4 = [avg_hist_or_vec(h4[i]; num_bins = num_bins, bin_edges = scalar_bin_edges, log_binning = log_binning) for i in 1:length(h4)]

    d1 = length(avg1) == 1 ? avg1[1] : vcat(avg1...)
    d2 = length(avg2) == 1 ? avg2[1] : vcat(avg2...)
    d3 = length(avg3) == 1 ? avg3[1] : vcat(avg3...)
    d4 = length(avg4) == 1 ? avg4[1] : vcat(avg4...)

    (; vmin, vmax, denom) = compute_color_scale([d1, d2, d3, d4]; log_color_scaling = log_binning)
    fig, axs = create_plot_matrix_figure_and_axes(;
        logscale_x = logscale_x_panel,
        logscale_y = logscale_y_panel,
        double_column = double_column,
        magnification = magnification,
        top_xaxis = top_xaxis,
        right_yaxis = right_yaxis,
        rowgap = rowgap,
        colgap = colgap,
    )

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

    xt = normalize_panel_ticks(xticks, "xticks")
    yt = normalize_panel_ticks(yticks, "yticks")
    plot_hist_or_vec_panel!(axs[1], d1, comp1, xlim[1], ylim[1], xlabel[1], ylabel[1], xt[1], yt[1];
        logscale_y = logscale_y_panel[1], invert_color_scaling = invert_color_scaling, invert_plot_order = invert_color_scaling, log_color_scaling = log_binning, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[2], d2, comp2, xlim[2], ylim[2], xlabel[2], ylabel[2], xt[2], yt[2];
        logscale_y = logscale_y_panel[2], invert_color_scaling = invert_color_scaling, invert_plot_order = invert_color_scaling, log_color_scaling = log_binning, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[3], d3, comp3, xlim[3], ylim[3], xlabel[3], ylabel[3], xt[3], yt[3];
        logscale_y = logscale_y_panel[3], invert_color_scaling = invert_color_scaling, invert_plot_order = invert_color_scaling, log_color_scaling = log_binning, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[4], d4, comp4, xlim[4], ylim[4], xlabel[4], ylabel[4], xt[4], yt[4];
        logscale_y = logscale_y_panel[4], invert_color_scaling = invert_color_scaling, invert_plot_order = invert_color_scaling, log_color_scaling = log_binning, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)

    add_plot_matrix_colorbar!(
        fig,
        vmin,
        vmax;
        colormap = colormap,
        invert_color_scaling = invert_color_scaling,
        log_color_scaling = log_binning,
        colorbar_label = colorbar_label,
        colorbar_ticks = colorbar_ticks,
        colorbar_pos = colorbar_pos,
        colorbar_size = colorbar_size,
        colorbar_side = colorbar_side,
        colorbar_label_pos = colorbar_label_pos,
    )

    CairoMakie.save(fig_path, fig)
    return return_axis ? (fig, axs) : fig
end

"""
    observable_plot_matrix(
        data_paths,
        comp_paths,
        scalar::Symbol,
        fig_path::String;
        xlim, ylim, xlabel, ylabel,
        sqrt_scalars = false,
        num_bins = nothing,
        ...
    )

Load and render a 2×2 observable plot matrix.

Panel order:
1. connectivity_link histogram
2. max_pathlen histogram
3. ev_sym_link vector
4. cardinalities histogram

The dataset observables are loaded in one bulk pass via `load_fields_from_paths`.
Comparison data is loaded the same way and reduced to one reference band/line per panel.
"""
function observable_plot_matrix(
    data_paths_in,
    comp_paths_in,
    scalar::Symbol,
    fig_path::String;
    xlim::AbstractVector{<:Union{Tuple{Float64,Float64},Nothing}},
    ylim::AbstractVector{<:Union{Tuple{Float64,Float64},Nothing}},
    xlabel::AbstractVector{<:Union{AbstractString,LaTeXStrings.LaTeXString,Nothing}},
    ylabel::AbstractVector{<:Union{AbstractString,LaTeXStrings.LaTeXString,Nothing}},
    sqrt_scalars::Bool = false,
    num_bins::Union{Nothing,Int} = nothing,
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
    logscale_x::AbstractVector{Bool} = fill(true, 4),
    logscale_y::AbstractVector{Bool} = fill(true, 4),
    double_column::Bool = true,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 0.0,
    colgap::Union{Nothing,Real} = 0.0,
    log_binning::Bool = false,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, Vector{CairoMakie.Axis}}}
    to_paths(x) = x isa AbstractString ? [String(x)] : String.(collect(x))
    data_paths = to_paths(data_paths_in)
    comp_paths = to_paths(comp_paths_in)

    data_loaded = load_observable_plot_matrix_observables(data_paths, scalar)
    comp_loaded = load_observable_plot_matrix_observables(comp_paths)

    normalized_connectivity_data = normalize_hists(data_loaded.connectivity_link_hists)
    normalized_connectivity_comp = normalize_hists(comp_loaded.connectivity_link_hists)
    normalized_max_pathlen_data = normalize_hists(data_loaded.max_pathlen_hists; normalization = 1)
    normalized_max_pathlen_comp = normalize_hists(comp_loaded.max_pathlen_hists; normalization = 1)
    normalized_cardinalities_data = normalize_hists(data_loaded.cardinalities_hists; normalize_x_axis_with_size = true, cset_size = data_loaded.size)
    normalized_cardinalities_comp = normalize_hists(comp_loaded.cardinalities_hists; normalize_x_axis_with_size = true, cset_size = comp_loaded.size)

    maybe_sqrt_pairs(pairs) = sqrt_scalars ? [(v, sqrt(Float64(s))) for (v, s) in pairs] : pairs
    normalized_connectivity_data = [maybe_sqrt_pairs(g) for g in normalized_connectivity_data]
    normalized_max_pathlen_data = [maybe_sqrt_pairs(g) for g in normalized_max_pathlen_data]
    ev_sym_link_data = [maybe_sqrt_pairs(g) for g in data_loaded.ev_sym_link]
    normalized_cardinalities_data = [maybe_sqrt_pairs(g) for g in normalized_cardinalities_data]

    data = (
        normalized_connectivity_data,
        normalized_max_pathlen_data,
        ev_sym_link_data,
        normalized_cardinalities_data,
    )
    comp = (
        normalized_connectivity_comp,
        normalized_max_pathlen_comp,
        comp_loaded.ev_sym_link,
        normalized_cardinalities_comp,
    )

    return _observable_plot_matrix_from_data(
        data,
        fig_path;
        xlim = xlim,
        ylim = ylim,
        xlabel = xlabel,
        ylabel = ylabel,
        num_bins = num_bins,
        colormap = colormap,
        invert_color_scaling = invert_color_scaling,
        colorbar_label = colorbar_label,
        colorbar_ticks = colorbar_ticks,
        colorbar_pos = colorbar_pos,
        colorbar_size = colorbar_size,
        colorbar_side = colorbar_side,
        colorbar_label_pos = colorbar_label_pos,
        comp = comp,
        comp_color = comp_color,
        comp_linewidth = comp_linewidth,
        xticks = xticks,
        yticks = yticks,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        double_column = double_column,
        magnification = magnification,
        plot_std = plot_std,
        right_yaxis = right_yaxis,
        top_xaxis = top_xaxis,
        rowgap = rowgap,
        colgap = colgap,
        log_binning = log_binning,
        return_axis = return_axis,
    )
end

function hist_hist_vec_hist_plot_matrix(args...; kwargs...)
    return _observable_plot_matrix_from_data(args...; kwargs...)
end

"""
    observable_distinguishability_plotmatrix(
        kind::String,
        scalar::Symbol,
        fig_path::String;
        size = 2048,
        ...
    )

Load and render a 2×2 plot matrix with the observables `connectivity_link`,
`max_pathlen`, and `ev_sym_link` in the first three panels and the
permutation-distinguishability p-value versus scalar in the fourth panel.
"""
function observable_distinguishability_plotmatrix(
    kind::String,
    scalar::Symbol,
    fig_path::String;
    size::Int = 2048,
    num_csets::Int=10000,
    comp_kind::Union{Nothing,String} = nothing,
    distinguishability::Symbol = :permutation,
    xlim::AbstractVector{<:Union{Tuple{Float64,Float64},Nothing}},
    ylim::AbstractVector{<:Union{Tuple{Float64,Float64},Nothing}},
    xlabel::AbstractVector{<:Union{AbstractString,LaTeXStrings.LaTeXString,Nothing}},
    ylabel::AbstractVector{<:Union{AbstractString,LaTeXStrings.LaTeXString,Nothing}},
    sqrt_scalars::Bool = false,
    num_bins::Union{Nothing,Int} = nothing,
    num_draws::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    energy_distance::Symbol = :Hellinger,
    mutual_information_k::Int = 5,
    mutual_information_pca_mode::Symbol = :cutoff,
    mutual_information_pca_dim::Int = 32,
    mutual_information_explained_variance::Real = 0.99,
    mutual_information_eigenvalue_rtol::Real = 1e-6,
    mutual_information_max_per_class::Union{Nothing,Int} = nothing,
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    projection_tolerance::Float64 = 1e-6,
    to_regularize_rel::Float64 = 0.01,
    tv_quantization_digits::Int = 8,
    check_bias::Bool = false,
    bias_num_splits::Int = 20,
    rng = Random.default_rng(),
    colormap = :magma,
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
    logscale_x::AbstractVector{Bool} = fill(true, 4),
    logscale_y::AbstractVector{Bool} = fill(true, 4),
    double_column::Bool = true,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 0.0,
    colgap::Union{Nothing,Real} = 0.0,
    log_binning::Bool = false,
    legendpos = :rb,
    progress::Bool = false,
    verbose::Bool = false,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, Vector{CairoMakie.Axis}}}
    validate_plot_matrix_inputs(
        xlim,
        ylim,
        xlabel,
        ylabel;
        magnification = magnification,
        rowgap = rowgap,
        colgap = colgap,
        colorbar_size = colorbar_size,
    )
    if !(size > 0)
        throw(DomainError(size, "size must be > 0"))
    end
    if !(n_perm >= 1)
        throw(DomainError(n_perm, "n_perm must be >= 1"))
    end
    logscale_x_panel = normalize_panel_bools(logscale_x, "logscale_x")
    logscale_y_panel = normalize_panel_bools(logscale_y, "logscale_y")
    if num_draws !== nothing && !(num_draws >= 1)
        throw(DomainError(num_draws, "num_draws must be >= 1 when provided"))
    end
    distinguishability_aliases = Dict(:mi => :mutual_information, :energy_distance => :energy)
    distinguishability = get(distinguishability_aliases, distinguishability, distinguishability)
    if distinguishability ∉ (:permutation, :tv, :energy, :mutual_information, :mahalanobis)
        throw(ArgumentError("distinguishability must be :permutation, :tv, :energy, :energy_distance, :mutual_information, :mi, or :mahalanobis"))
    end

    comp_name = isnothing(comp_kind) ? comparison_kind_for_observable_plot(kind) : comp_kind
    obs_paths = data_paths(["$(kind)_$(size)_$(num_csets)/statistics.jld2"])
    comp_paths = data_paths(["$(comp_name)_$(size)_10000/statistics.jld2"])

    data_loaded = load_observable_plot_matrix_observables(obs_paths, scalar)
    comp_loaded = load_observable_plot_matrix_observables(comp_paths)

    normalized_connectivity_data = normalize_hists(data_loaded.connectivity_link_hists)
    normalized_connectivity_comp = normalize_hists(comp_loaded.connectivity_link_hists)
    normalized_max_pathlen_data = normalize_hists(data_loaded.max_pathlen_hists; normalization = 1)
    normalized_max_pathlen_comp = normalize_hists(comp_loaded.max_pathlen_hists; normalization = 1)

    maybe_sqrt_pairs(pairs) = sqrt_scalars ? [(v, sqrt(Float64(s))) for (v, s) in pairs] : pairs
    normalized_connectivity_data = [maybe_sqrt_pairs(g) for g in normalized_connectivity_data]
    normalized_max_pathlen_data = [maybe_sqrt_pairs(g) for g in normalized_max_pathlen_data]
    ev_sym_link_data = [maybe_sqrt_pairs(g) for g in data_loaded.ev_sym_link]

    data = (
        normalized_connectivity_data,
        normalized_max_pathlen_data,
        ev_sym_link_data,
        normalized_connectivity_data,
    )
    scalar_bin_edges = compute_plot_matrix_scalar_bin_edges(
        data;
        num_bins = num_bins,
        log_binning = log_binning,
    )

    d1_groups = [avg_hist_or_vec(g; num_bins = num_bins, bin_edges = scalar_bin_edges, log_binning = log_binning) for g in normalized_connectivity_data]
    d2_groups = [avg_hist_or_vec(g; num_bins = num_bins, bin_edges = scalar_bin_edges, log_binning = log_binning) for g in normalized_max_pathlen_data]
    d3_groups = [avg_hist_or_vec(g; num_bins = num_bins, bin_edges = scalar_bin_edges, log_binning = log_binning) for g in ev_sym_link_data]
    d1 = length(d1_groups) == 1 ? d1_groups[1] : vcat(d1_groups...)
    d2 = length(d2_groups) == 1 ? d2_groups[1] : vcat(d2_groups...)
    d3 = length(d3_groups) == 1 ? d3_groups[1] : vcat(d3_groups...)

    connectivity_ref = flatten_loaded_values(normalized_connectivity_comp)
    max_pathlen_ref = flatten_loaded_values(normalized_max_pathlen_comp)
    ev_sym_link_ref = flatten_loaded_values(comp_loaded.ev_sym_link)
    comp1 = comp_avg_hist_or_vec(connectivity_ref; num_bins = num_bins)
    comp2 = comp_avg_hist_or_vec(max_pathlen_ref; num_bins = num_bins)
    comp3 = comp_avg_hist_or_vec(ev_sym_link_ref; num_bins = num_bins)

    distinguishability_series =
        if distinguishability == :permutation
            (
                connectivity_link = compute_permutation_scalar_series(
                    normalized_connectivity_data,
                    connectivity_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    num_draws = num_draws,
                    n_perm = n_perm,
                    distance = energy_distance,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
                max_pathlen = compute_permutation_scalar_series(
                    normalized_max_pathlen_data,
                    max_pathlen_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    num_draws = num_draws,
                    n_perm = n_perm,
                    distance = energy_distance,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
                ev_sym_link = compute_permutation_scalar_series(
                    ev_sym_link_data,
                    ev_sym_link_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    num_draws = num_draws,
                    n_perm = n_perm,
                    distance = energy_distance,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
            )
        elseif distinguishability == :tv
            (
                connectivity_link = compute_total_variation_scalar_series(
                    normalized_connectivity_data,
                    connectivity_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    tv_quantization_digits = tv_quantization_digits,
                    check_bias = check_bias,
                    bias_num_splits = bias_num_splits,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
                max_pathlen = compute_total_variation_scalar_series(
                    normalized_max_pathlen_data,
                    max_pathlen_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    tv_quantization_digits = tv_quantization_digits,
                    check_bias = check_bias,
                    bias_num_splits = bias_num_splits,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
                ev_sym_link = compute_total_variation_scalar_series(
                    ev_sym_link_data,
                    ev_sym_link_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    tv_quantization_digits = tv_quantization_digits,
                    check_bias = check_bias,
                    bias_num_splits = bias_num_splits,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
            )
        elseif distinguishability == :energy
            (
                connectivity_link = compute_energy_scalar_series(
                    normalized_connectivity_data,
                    connectivity_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    num_draws = num_draws,
                    distance = energy_distance,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
                max_pathlen = compute_energy_scalar_series(
                    normalized_max_pathlen_data,
                    max_pathlen_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    num_draws = num_draws,
                    distance = energy_distance,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
                ev_sym_link = compute_energy_scalar_series(
                    ev_sym_link_data,
                    ev_sym_link_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    num_draws = num_draws,
                    distance = energy_distance,
                    rng = rng,
                    progress = progress,
                    verbose = verbose,
                ),
            )
        elseif distinguishability == :mutual_information
            (
                connectivity_link = compute_mutual_information_scalar_series(
                    normalized_connectivity_data,
                    connectivity_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    k = mutual_information_k,
                    pca_mode = mutual_information_pca_mode,
                    pca_dim = mutual_information_pca_dim,
                    explained_variance = mutual_information_explained_variance,
                    eigenvalue_rtol = mutual_information_eigenvalue_rtol,
                    max_per_class = mutual_information_max_per_class,
                    rng = rng,
                    verbose = verbose,
                ),
                max_pathlen = compute_mutual_information_scalar_series(
                    normalized_max_pathlen_data,
                    max_pathlen_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    k = mutual_information_k,
                    pca_mode = mutual_information_pca_mode,
                    pca_dim = mutual_information_pca_dim,
                    explained_variance = mutual_information_explained_variance,
                    eigenvalue_rtol = mutual_information_eigenvalue_rtol,
                    max_per_class = mutual_information_max_per_class,
                    rng = rng,
                    verbose = verbose,
                ),
                ev_sym_link = compute_mutual_information_scalar_series(
                    ev_sym_link_data,
                    ev_sym_link_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    k = mutual_information_k,
                    pca_mode = mutual_information_pca_mode,
                    pca_dim = mutual_information_pca_dim,
                    explained_variance = mutual_information_explained_variance,
                    eigenvalue_rtol = mutual_information_eigenvalue_rtol,
                    max_per_class = mutual_information_max_per_class,
                    rng = rng,
                    verbose = verbose,
                ),
            )
        else
            (
                connectivity_link = compute_mahalanobis_scalar_series(
                    normalized_connectivity_data,
                    connectivity_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    regulator = regulator,
                    R = R,
                    q = q,
                    alpha = alpha,
                    rng = rng,
                    projection_tolerance = projection_tolerance,
                    to_regularize_rel = to_regularize_rel,
                    progress = progress,
                    verbose = verbose,
                ),
                max_pathlen = compute_mahalanobis_scalar_series(
                    normalized_max_pathlen_data,
                    max_pathlen_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    regulator = regulator,
                    R = R,
                    q = q,
                    alpha = alpha,
                    rng = rng,
                    projection_tolerance = projection_tolerance,
                    to_regularize_rel = to_regularize_rel,
                    progress = progress,
                    verbose = verbose,
                ),
                ev_sym_link = compute_mahalanobis_scalar_series(
                    ev_sym_link_data,
                    ev_sym_link_ref;
                    num_bins = num_bins,
                    bin_edges = scalar_bin_edges,
                    log_binning = log_binning,
                    regulator = regulator,
                    R = R,
                    q = q,
                    alpha = alpha,
                    rng = rng,
                    projection_tolerance = projection_tolerance,
                    to_regularize_rel = to_regularize_rel,
                    progress = progress,
                    verbose = verbose,
                ),
            )
        end

    if verbose
        for (observable, series) in pairs(distinguishability_series)
            for row in series
                if distinguishability == :permutation
                    println(
                        "observable_distinguishability_plotmatrix ",
                        observable,
                        " scalar=", row.scalar,
                        " D_obs=", row.D_obs,
                        " p_value=", row.p_value,
                        " z_emp=", row.z_emp,
                        " z_coll=", row.z_coll,
                        " std_Ts=", row.std_Ts,
                    )
                elseif distinguishability == :tv
                    println(
                        "observable_distinguishability_plotmatrix ",
                        observable,
                        " scalar=", row.scalar,
                        " D_tv=", row.D_tv,
                        " bayes_accuracy=", row.bayes_accuracy,
                        " tv_bias_mean=", row.tv_bias_mean,
                        " tv_bias_std=", row.tv_bias_std,
                    )
                elseif distinguishability == :energy
                    println(
                        "observable_distinguishability_plotmatrix ",
                        observable,
                        " scalar=", row.scalar,
                        " E=", row.E,
                        " D=", row.D,
                        isnothing(num_draws) ? "" : " std=$(row.std)",
                    )
                elseif distinguishability == :mutual_information
                    println(
                        "observable_distinguishability_plotmatrix ",
                        observable,
                        " scalar=", row.scalar,
                        " D_mi=", row.D,
                    )
                else
                    println(
                        "observable_distinguishability_plotmatrix ",
                        observable,
                        " scalar=", row.scalar,
                        " D_mahalanobis=", row.D,
                    )
                end
            end
        end
    end

    panel4_key =
        distinguishability == :permutation ? :p_value :
        distinguishability == :tv ? :bayes_accuracy :
        distinguishability == :energy ? :D :
        :D
    panel4_data = [
        (row.scalar, [getproperty(row, panel4_key)], zeros(1))
        for row in distinguishability_series.connectivity_link
    ]
    append!(panel4_data, ((row.scalar, [getproperty(row, panel4_key)], zeros(1)) for row in distinguishability_series.max_pathlen))
    append!(panel4_data, ((row.scalar, [getproperty(row, panel4_key)], zeros(1)) for row in distinguishability_series.ev_sym_link))

    (; vmin, vmax, denom) = compute_color_scale([d1, d2, d3, panel4_data]; log_color_scaling = log_binning)
    fig, axs = create_plot_matrix_figure_and_axes(;
        logscale_x = logscale_x_panel,
        logscale_y = logscale_y_panel,
        double_column = double_column,
        magnification = magnification,
        top_xaxis = top_xaxis,
        right_yaxis = right_yaxis,
        rowgap = rowgap,
        colgap = colgap,
    )

    xt = normalize_panel_ticks(xticks, "xticks")
    yt = normalize_panel_ticks(yticks, "yticks")
    plot_hist_or_vec_panel!(axs[1], d1, comp1, xlim[1], ylim[1], xlabel[1], ylabel[1], xt[1], yt[1];
        logscale_y = logscale_y_panel[1], invert_color_scaling = invert_color_scaling, invert_plot_order = invert_color_scaling, log_color_scaling = log_binning, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[2], d2, comp2, xlim[2], ylim[2], xlabel[2], ylabel[2], xt[2], yt[2];
        logscale_y = logscale_y_panel[2], invert_color_scaling = invert_color_scaling, invert_plot_order = invert_color_scaling, log_color_scaling = log_binning, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)
    plot_hist_or_vec_panel!(axs[3], d3, comp3, xlim[3], ylim[3], xlabel[3], ylabel[3], xt[3], yt[3];
        logscale_y = logscale_y_panel[3], invert_color_scaling = invert_color_scaling, invert_plot_order = invert_color_scaling, log_color_scaling = log_binning, plot_std = plot_std,
        vmin = vmin, denom = denom, colormap = colormap, comp_color = comp_color, comp_linewidth = comp_linewidth)

    ax4 = axs[4]
    apply_axis_metadata!(ax4, xlim[4], ylim[4], xlabel[4], ylabel[4], xt[4], yt[4])
    observable_labels = observable_distinguishability_symbol_labels()
    panel4_specs = [
        (:connectivity_link, distinguishability_series.connectivity_link, nothing),
        (:max_pathlen, distinguishability_series.max_pathlen, :dash),
        (:ev_sym_link, distinguishability_series.ev_sym_link, :dot),
    ]
    for (observable, series, linestyle) in panel4_specs
        xs = Float64[row.scalar for row in series]
        ys = Float64[getproperty(row, panel4_key) for row in series]
        CairoMakie.lines!(ax4, xs, ys; label = getproperty(observable_labels, observable), linestyle = linestyle)
    end
    CairoMakie.axislegend(ax4; position = legendpos)

    add_plot_matrix_colorbar!(
        fig,
        vmin,
        vmax;
        colormap = colormap,
        invert_color_scaling = invert_color_scaling,
        log_color_scaling = log_binning,
        colorbar_label = colorbar_label,
        colorbar_ticks = colorbar_ticks,
        colorbar_pos = colorbar_pos,
        colorbar_size = colorbar_size,
        colorbar_side = colorbar_side,
        colorbar_label_pos = colorbar_label_pos,
    )

    CairoMakie.save(fig_path, fig)
    return return_axis ? (fig, axs) : fig
end

comparison_kind_for_observable_plot(kind::AbstractString) =
    kind == "minkowski_quasicrystal" ? "minkowski_sprinkling" : "manifoldlike_simply_connected"

flatten_scalar_paired_values(groups::AbstractVector{<:AbstractVector}) =
    getindex.(length(groups) == 1 ? groups[1] : vcat(groups...), 1)

flatten_loaded_values(groups::AbstractVector{<:AbstractVector}) =
    length(groups) == 1 ? groups[1] : vcat(groups...)

function compute_mutual_information_scalar_series(
    data_groups::AbstractVector{<:AbstractVector},
    ref_values::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    log_binning::Bool = false,
    k::Int = 5,
    pca_mode::Symbol = :cutoff,
    pca_dim::Int = 32,
    explained_variance::Real = 0.99,
    eigenvalue_rtol::Real = 1e-6,
    max_per_class::Union{Nothing,Int} = nothing,
    rng = Random.default_rng(),
    verbose::Bool = false,
)
    pairs = length(data_groups) == 1 ? data_groups[1] : vcat(data_groups...)
    bins = group_scalar_pairs_for_plot(
        pairs;
        num_bins = num_bins,
        bin_edges = bin_edges,
        log_binning = log_binning,
    )
    seeds = rand(rng, UInt64, length(bins))
    out = Vector{NamedTuple}(undef, length(bins))
    for (i, (s, vals)) in enumerate(bins)
        res = distinguishability_mutual_information(
            vals,
            ref_values;
            k = k,
            pca_mode = pca_mode,
            pca_dim = pca_dim,
            explained_variance = explained_variance,
            eigenvalue_rtol = eigenvalue_rtol,
            max_per_class = max_per_class,
            rng = Random.Xoshiro(seeds[i]),
            verbose = verbose,
        )
        out[i] = (scalar = s, D = res.D_mi)
    end
    return out
end

function compute_mahalanobis_scalar_series(
    data_groups::AbstractVector{<:AbstractVector},
    ref_values::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    log_binning::Bool = false,
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    projection_tolerance::Float64 = 1e-6,
    to_regularize_rel::Float64 = 0.01,
    progress::Bool = false,
    verbose::Bool = false,
)
    pairs = length(data_groups) == 1 ? data_groups[1] : vcat(data_groups...)
    bins = group_scalar_pairs_for_plot(
        pairs;
        num_bins = num_bins,
        bin_edges = bin_edges,
        log_binning = log_binning,
    )
    seeds = rand(rng, UInt64, length(bins))
    out = Vector{NamedTuple}(undef, length(bins))
    for (i, (s, vals)) in enumerate(bins)
        res = mahalanobis_gap_distinguishability(
            vals,
            ref_values;
            regulator = regulator,
            R = R,
            q = q,
            alpha = alpha,
            rng = Random.Xoshiro(seeds[i]),
            symmetric = false,
            projection_tolerance = projection_tolerance,
            to_regularize_rel = to_regularize_rel,
            progress = progress,
            verbose = verbose,
        )
        out[i] = (scalar = s, D = res.D)
    end
    return out
end

function compute_energy_scalar_series(
    data_groups::AbstractVector{<:AbstractVector},
    ref_values::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    log_binning::Bool = false,
    num_draws::Union{Nothing,Int} = nothing,
    distance::Symbol = :Hellinger,
    rng = Random.default_rng(),
    progress::Bool = false,
    verbose::Bool = false,
)
    pairs = length(data_groups) == 1 ? data_groups[1] : vcat(data_groups...)
    bins = group_scalar_pairs_for_plot(
        pairs;
        num_bins = num_bins,
        bin_edges = bin_edges,
        log_binning = log_binning,
    )
    out = Vector{NamedTuple}(undef, length(bins))
    pm = progress ? ProgressMeter.Progress(length(bins); desc = "energy scalar bins") : nothing
    for (k, (s, vals)) in enumerate(bins)
        res = isnothing(num_draws) ?
            energy_based_histogram_distinguishability(vals, ref_values; distance = distance, verbose = verbose) :
            energy_based_histogram_distinguishability(vals, ref_values, num_draws; rng = rng, distance = distance, verbose = verbose)
        out[k] = isnothing(num_draws) ? (scalar = s, D = res.D, E = res.E) : (scalar = s, D = res.D, E = res.E, std = res.std)
        progress && ProgressMeter.next!(pm)
    end
    return out
end

function compute_permutation_scalar_series(
    data_groups::AbstractVector{<:AbstractVector},
    ref_values::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    log_binning::Bool = false,
    num_draws::Union{Nothing,Int} = nothing,
    n_perm::Int = 1000,
    distance::Symbol = :Hellinger,
    rng = Random.default_rng(),
    progress::Bool = false,
    verbose::Bool = false,
)
    pairs = length(data_groups) == 1 ? data_groups[1] : vcat(data_groups...)
    bins = group_scalar_pairs_for_plot(
        pairs;
        num_bins = num_bins,
        bin_edges = bin_edges,
        log_binning = log_binning,
    )
    out = Vector{NamedTuple}(undef, length(bins))
    pm = progress ? ProgressMeter.Progress(length(bins); desc = "perm scalar bins") : nothing
    for (k, (s, vals)) in enumerate(bins)
        res = isnothing(num_draws) ?
            histogram_distinguishability_permutation(vals, ref_values; n_perm = n_perm, rng = rng, progress = false, distance = distance, verbose = verbose) :
            histogram_distinguishability_permutation(vals, ref_values, num_draws; n_perm = n_perm, rng = rng, progress = false, distance = distance, verbose = verbose)
        out[k] = (scalar = s, D_obs = res.D_obs, p_value = res.p_value, z_emp = res.z_emp, z_coll = res.z_coll, std_Ts = res.std_Ts)
        progress && ProgressMeter.next!(pm)
    end
    return out
end

function compute_total_variation_scalar_series(
    data_groups::AbstractVector{<:AbstractVector},
    ref_values::AbstractVector;
    num_bins::Union{Nothing,Int} = nothing,
    bin_edges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    log_binning::Bool = false,
    tv_quantization_digits::Int = 8,
    check_bias::Bool = false,
    bias_num_splits::Int = 20,
    rng = Random.default_rng(),
    progress::Bool = false,
    verbose::Bool = false,
)
    pairs = length(data_groups) == 1 ? data_groups[1] : vcat(data_groups...)
    bins = group_scalar_pairs_for_plot(
        pairs;
        num_bins = num_bins,
        bin_edges = bin_edges,
        log_binning = log_binning,
    )
    out = Vector{NamedTuple}(undef, length(bins))
    pm = progress ? ProgressMeter.Progress(length(bins); desc = "tv scalar bins") : nothing
    for (k, (s, vals)) in enumerate(bins)
        res = distinguishability_total_variation(
            vals,
            ref_values;
            tv_quantization_digits = tv_quantization_digits,
            check_bias = check_bias,
            bias_num_splits = bias_num_splits,
            rng = rng,
            verbose = verbose,
        )
        out[k] = (
            scalar = s,
            D_tv = res.D_tv,
            bayes_accuracy = res.bayes_accuracy,
            tv_bias_mean = res.tv_bias_mean,
            tv_bias_std = res.tv_bias_std,
            D_tv_debiased = res.D_tv_debiased,
            bayes_accuracy_debiased = res.bayes_accuracy_debiased,
        )
        progress && ProgressMeter.next!(pm)
    end
    return out
end

function observable_distinguishability_symbol_labels()
    return (
        cardinalities = LaTeXStrings.L"\mathcal{S}_j",
        ev_sym_link = LaTeXStrings.L"\lambda_j",
        connectivity_link = LaTeXStrings.L"P^{\mathrm{link}}_j",
        max_pathlen = LaTeXStrings.L"\mathcal{H}_j",
    )
end

function graph_observable_field_spec(observable::Symbol)
    if observable == :cardinalities
        return (field = :cardinalities_hist, hist = true, normalization = :probability)
    elseif observable == :in_degree
        return (field = :in_degree_hist, hist = true, normalization = :probability)
    elseif observable == :out_degree
        return (field = :out_degree_hist, hist = true, normalization = :probability)
    elseif observable == :connectivity
        return (field = :degree_hist, hist = true, normalization = :probability)
    elseif observable == :in_degree_link
        return (field = :in_degree_hist_link, hist = true, normalization = :probability)
    elseif observable == :out_degree_link
        return (field = :out_degree_hist_link, hist = true, normalization = :probability)
    elseif observable == :connectivity_link
        return (field = :degree_hist_link, hist = true, normalization = :probability)
    elseif observable == :max_pathlen
        return (field = :max_pathlen_hist, hist = true, normalization = :probability)
    elseif observable == :ev_sym_link
        return (field = :ev_sym_link, hist = false, normalization = nothing)
    else
        throw(ArgumentError("unsupported graph observable $(observable)"))
    end
end

function graph_observable_scalar_label(scalar::Symbol)
    if scalar == :num_layers
        return LaTeXStrings.L"n_{\mathrm{layers}}"
    elseif scalar == :rel_size_KR
        return LaTeXStrings.L"\delta_{\mathrm{KR}}"
    elseif scalar == :rel_num_flips
        return LaTeXStrings.L"\delta_{\mathrm{flips}}"
    elseif scalar == :num_boundary_cuts
        return LaTeXStrings.L"n_{\mathrm{pants}}"
    elseif scalar == :genus
        return LaTeXStrings.L"g"
    end
    return string(scalar)
end

function default_kind_plot_matrix_title(kind::AbstractString)
    if kind == "grid"
        return "regular lattice causets"
    elseif kind == "minkowski_quasicrystal"
        return "quasicrystal causets"
    elseif kind == "layered"
        return "layered causets"
    elseif kind == "random"
        return "fixed-connectivity causets"
    elseif kind == "merged_link_prob_tenth"
        return "spacetimes with KR insertions"
    elseif kind == "destroyed"
        return "spacetimes with (worm-)holes"
    elseif kind == "manifoldlike_pants"
        return "generalized-pants spacetimes"
    elseif kind == "manifoldlike_genus"
        return "spacetimes with handles"
    end
    return replace(kind, "_" => " ")
end

function normalize_kind_plot_matrix_meta(values, name::AbstractString; n::Int = 8, default = nothing)
    if values === nothing
        return fill(default, n)
    end
    normalized = collect(values)
    if length(normalized) != n
        throw(ArgumentError("$name must have length $n"))
    end
    return normalized
end

function graph_observable_kind_plot_matrix_panel_meta(;
    xlim_grid = nothing,
    xlim_quasicrystal = nothing,
    xlim_layered = nothing,
    xlim_random = nothing,
    xlim_merged = nothing,
    xlim_destroyed = nothing,
    xlim_pants = nothing,
    xlim_genus = nothing,
    ylim_grid = nothing,
    ylim_quasicrystal = nothing,
    ylim_layered = nothing,
    ylim_random = nothing,
    ylim_merged = nothing,
    ylim_destroyed = nothing,
    ylim_pants = nothing,
    ylim_genus = nothing,
    xlabel_grid = nothing,
    xlabel_quasicrystal = nothing,
    xlabel_layered = nothing,
    xlabel_random = nothing,
    xlabel_merged = nothing,
    xlabel_destroyed = nothing,
    xlabel_pants = nothing,
    xlabel_genus = nothing,
    ylabel_grid = nothing,
    ylabel_quasicrystal = nothing,
    ylabel_layered = nothing,
    ylabel_random = nothing,
    ylabel_merged = nothing,
    ylabel_destroyed = nothing,
    ylabel_pants = nothing,
    ylabel_genus = nothing,
    xticks_grid = nothing,
    xticks_quasicrystal = nothing,
    xticks_layered = nothing,
    xticks_random = nothing,
    xticks_merged = nothing,
    xticks_destroyed = nothing,
    xticks_pants = nothing,
    xticks_genus = nothing,
    yticks_grid = nothing,
    yticks_quasicrystal = nothing,
    yticks_layered = nothing,
    yticks_random = nothing,
    yticks_merged = nothing,
    yticks_destroyed = nothing,
    yticks_pants = nothing,
    yticks_genus = nothing,
    colorbar_ticks_layered = nothing,
    colorbar_ticks_merged = nothing,
    colorbar_ticks_destroyed = nothing,
    colorbar_ticks_pants = nothing,
    colorbar_ticks_genus = nothing,
)
    return (
        xlim = [xlim_grid, xlim_quasicrystal, xlim_layered, xlim_random, xlim_merged, xlim_destroyed, xlim_pants, xlim_genus],
        ylim = [ylim_grid, ylim_quasicrystal, ylim_layered, ylim_random, ylim_merged, ylim_destroyed, ylim_pants, ylim_genus],
        xlabel = [xlabel_grid, xlabel_quasicrystal, xlabel_layered, xlabel_random, xlabel_merged, xlabel_destroyed, xlabel_pants, xlabel_genus],
        ylabel = [ylabel_grid, ylabel_quasicrystal, ylabel_layered, ylabel_random, ylabel_merged, ylabel_destroyed, ylabel_pants, ylabel_genus],
        xticks = [xticks_grid, xticks_quasicrystal, xticks_layered, xticks_random, xticks_merged, xticks_destroyed, xticks_pants, xticks_genus],
        yticks = [yticks_grid, yticks_quasicrystal, yticks_layered, yticks_random, yticks_merged, yticks_destroyed, yticks_pants, yticks_genus],
        colorbar_ticks = [nothing, nothing, colorbar_ticks_layered, nothing, colorbar_ticks_merged, colorbar_ticks_destroyed, colorbar_ticks_pants, colorbar_ticks_genus],
    )
end

function load_graph_observable_kind_panel(
    paths_in,
    observable::Symbol;
    size::Int,
    scalar::Union{Nothing,Symbol} = nothing,
    comp_paths_in = nothing,
    num_bins::Union{Nothing,Int} = nothing,
    log_binning::Bool = false,
    verbose::Bool = false,
)
    spec = graph_observable_field_spec(observable)
    paths = paths_in isa AbstractString ? [String(paths_in)] : String.(collect(paths_in))
    normalize_x_axis_with_size = observable === :cardinalities
    xscale = observable in (:cardinalities, :ev_sym_link) ? inv(Float64(size)) : 1.0
    yscale = 1.0
    values =
        if spec.hist
            isnothing(scalar) ?
                load_histograms_from_paths(paths, spec.field; verbose = verbose)[1] :
                load_histograms_from_paths(paths, spec.field, scalar; verbose = verbose)[1]
        else
            isnothing(scalar) ?
                load_fields_from_paths(paths, [spec.field]; verbose = verbose)[1][1] :
                load_fields_from_paths(paths, [spec.field], scalar; verbose = verbose)[1][1]
        end

    prepared =
        if spec.hist
            normalize_hists([values]; normalization = spec.normalization, normalize_x_axis_with_size = normalize_x_axis_with_size, cset_size = size)[1]
        else
            values
        end

    comp_mean_std = nothing
    if comp_paths_in !== nothing
        comp_paths = comp_paths_in isa AbstractString ? [String(comp_paths_in)] : String.(collect(comp_paths_in))
        comp_values =
            if spec.hist
                load_histograms_from_paths(comp_paths, spec.field; verbose = verbose)[1]
            else
                load_fields_from_paths(comp_paths, [spec.field]; verbose = verbose)[1][1]
            end
        comp_prepared =
            if spec.hist
                normalize_hists([comp_values]; normalization = spec.normalization, normalize_x_axis_with_size = normalize_x_axis_with_size, cset_size = size)[1]
            else
                comp_values
            end
        comp_mean_std = comp_avg_hist_or_vec(comp_prepared; num_bins = num_bins)
    end

    if isnothing(scalar)
        mean_std = spec.hist ? average_histogram_with_std(prepared) : average_vectors_with_std(prepared)
        data = [(1.0, mean_std[1], mean_std[2])]
        return (data = data, scalar = nothing, comp = comp_mean_std, xscale = xscale, yscale = yscale)
    end

    bin_edges = compute_plot_matrix_scalar_bin_edges((prepared,); num_bins = num_bins, log_binning = log_binning)
    data = avg_hist_or_vec(prepared; num_bins = num_bins, bin_edges = bin_edges, log_binning = log_binning)
    return (data = data, scalar = scalar, comp = comp_mean_std, xscale = xscale, yscale = yscale)
end

function load_graph_observable_kind_plot_matrix_data(
    observable::Symbol;
    size::Int = 2048,
    grid_kind::String = "grid",
    quasicrystal_kind::String = "minkowski_quasicrystal",
    layered_kind::String = "layered",
    random_kind::String = "random",
    merged_kind::String = "merged_link_prob_tenth",
    destroyed_kind::String = "destroyed",
    pants_kind::String = "manifoldlike_pants",
    genus_kind::String = "manifoldlike_genus",
    grid_paths = data_paths(["grid_$(size)_10000/statistics.jld2"]),
    quasicrystal_paths = data_paths(["minkowski_quasicrystal_$(size)_10000/statistics.jld2"]),
    layered_paths = data_paths(["layered_$(size)_10000/statistics.jld2"]),
    random_paths = data_paths(["random_$(size)_10000/statistics.jld2"]),
    merged_paths = data_paths(["merged_link_prob_tenth_$(size)_30000/statistics.jld2"]),
    destroyed_paths = data_paths(["destroyed_$(size)_100000/statistics.jld2"]),
    pants_paths = data_paths(["manifoldlike_pants_$(size)_10000/statistics.jld2"]),
    genus_paths = data_paths(["manifoldlike_genus_$(size)_10000/statistics.jld2"]),
    standard_comp_paths = data_paths(["manifoldlike_simply_connected_$(size)_10000/statistics.jld2"]),
    quasicrystal_comp_paths = data_paths(["minkowski_sprinkling_$(size)_10000/statistics.jld2"]),
    num_bins::Union{Nothing,Int} = nothing,
    log_colorbar::Union{Nothing,Bool} = nothing,
    log_colorbar_layered::Union{Nothing,Bool} = nothing,
    log_colorbar_merged::Union{Nothing,Bool} = nothing,
    log_colorbar_destroyed::Union{Nothing,Bool} = nothing,
    verbose::Bool = false,
)
    if !(size > 0)
        throw(DomainError(size, "size must be > 0"))
    end

    panel_specs = [
        (kind = grid_kind, paths = grid_paths, comp_paths = standard_comp_paths, scalar = nothing, invert = false, default_log_binning = false),
        (kind = quasicrystal_kind, paths = quasicrystal_paths, comp_paths = quasicrystal_comp_paths, scalar = nothing, invert = false, default_log_binning = false),
        (kind = layered_kind, paths = layered_paths, comp_paths = standard_comp_paths, scalar = :num_layers, invert = false, default_log_binning = false),
        (kind = random_kind, paths = random_paths, comp_paths = standard_comp_paths, scalar = nothing, invert = false, default_log_binning = false),
        (kind = merged_kind, paths = merged_paths, comp_paths = standard_comp_paths, scalar = :rel_size_KR, invert = true, default_log_binning = true),
        (kind = destroyed_kind, paths = destroyed_paths, comp_paths = standard_comp_paths, scalar = :rel_num_flips, invert = true, default_log_binning = true),
        (kind = pants_kind, paths = pants_paths, comp_paths = standard_comp_paths, scalar = :num_boundary_cuts, invert = true, default_log_binning = false),
        (kind = genus_kind, paths = genus_paths, comp_paths = standard_comp_paths, scalar = :genus, invert = true, default_log_binning = false),
    ]

    requested_panel_log_binning = [
        nothing,
        nothing,
        log_colorbar_layered,
        nothing,
        log_colorbar_merged,
        log_colorbar_destroyed,
        nothing,
        nothing,
    ]
    panel_log_binning = [
        isnothing(requested_panel_log_binning[i]) ?
            (isnothing(log_colorbar) ? panel_specs[i].default_log_binning : log_colorbar) :
            requested_panel_log_binning[i]
        for i in eachindex(panel_specs)
    ]

    panels = [begin
        load_graph_observable_kind_panel(
            spec.paths,
            observable;
            size = size,
            scalar = spec.scalar,
            comp_paths_in = spec.comp_paths,
            num_bins = num_bins,
            log_binning = false,
            verbose = verbose,
        )
    end for (i, spec) in enumerate(panel_specs)]

    return (
        observable = observable,
        size = size,
        panel_specs = panel_specs,
        panels = panels,
        num_bins = num_bins,
        panel_log_colorbar = panel_log_binning,
        log_colorbar = log_colorbar,
    )
end

"""
    graph_observable_kind_plot_matrix(
        observable::Symbol,
        fig_path::String;
        size = 2048,
        ...
    )

Render a 4×2 kind-comparison matrix for one graph observable. Panel order:
`grid | quasicrystal`, `layered | random`, `merged | destroyed`, `pants | genus`.

Kinds with scalar metadata are color-coded:
- `layered` by `:num_layers` using `:viridis`
- `merged` by `:rel_size_KR` using inverted `:magma`
- `destroyed` by `:rel_num_flips` using inverted `:magma`
- `pants` by `:num_boundary_cuts` using inverted `:magma`
- `genus` by `:genus` using inverted `:magma`
"""
function graph_observable_kind_plot_matrix(
    observable::Symbol,
    fig_path::String;
    size::Int = 2048,
    grid_kind::String = "grid",
    quasicrystal_kind::String = "minkowski_quasicrystal",
    layered_kind::String = "layered",
    random_kind::String = "random",
    merged_kind::String = "merged_link_prob_tenth",
    destroyed_kind::String = "destroyed",
    pants_kind::String = "manifoldlike_pants",
    genus_kind::String = "manifoldlike_genus",
    grid_paths = data_paths(["grid_$(size)_10000/statistics.jld2"]),
    quasicrystal_paths = data_paths(["minkowski_quasicrystal_$(size)_10000/statistics.jld2"]),
    layered_paths = data_paths(["layered_$(size)_10000/statistics.jld2"]),
    random_paths = data_paths(["random_$(size)_10000/statistics.jld2"]),
    merged_paths = data_paths(["merged_link_prob_tenth_$(size)_30000/statistics.jld2"]),
    destroyed_paths = data_paths(["destroyed_$(size)_100000/statistics.jld2"]),
    pants_paths = data_paths(["manifoldlike_pants_$(size)_10000/statistics.jld2"]),
    genus_paths = data_paths(["manifoldlike_genus_$(size)_10000/statistics.jld2"]),
    standard_comp_paths = data_paths(["manifoldlike_simply_connected_$(size)_10000/statistics.jld2"]),
    quasicrystal_comp_paths = data_paths(["minkowski_sprinkling_$(size)_10000/statistics.jld2"]),
    xlim_grid = nothing,
    title_grid = nothing,
    xlim_quasicrystal = nothing,
    title_quasicrystal = nothing,
    xlim_layered = nothing,
    title_layered = nothing,
    xlim_random = nothing,
    title_random = nothing,
    xlim_merged = nothing,
    title_merged = nothing,
    xlim_destroyed = nothing,
    title_destroyed = nothing,
    xlim_pants = nothing,
    title_pants = nothing,
    xlim_genus = nothing,
    title_genus = nothing,
    ylim_grid = nothing,
    ylim_quasicrystal = nothing,
    ylim_layered = nothing,
    ylim_random = nothing,
    ylim_merged = nothing,
    ylim_destroyed = nothing,
    ylim_pants = nothing,
    ylim_genus = nothing,
    xlabel_grid = nothing,
    xlabel_quasicrystal = nothing,
    xlabel_layered = nothing,
    xlabel_random = nothing,
    xlabel_merged = nothing,
    xlabel_destroyed = nothing,
    xlabel_pants = nothing,
    xlabel_genus = nothing,
    ylabel_grid = nothing,
    ylabel_quasicrystal = nothing,
    ylabel_layered = nothing,
    ylabel_random = nothing,
    ylabel_merged = nothing,
    ylabel_destroyed = nothing,
    ylabel_pants = nothing,
    ylabel_genus = nothing,
    xticks_grid = nothing,
    xticks_quasicrystal = nothing,
    xticks_layered = nothing,
    xticks_random = nothing,
    xticks_merged = nothing,
    xticks_destroyed = nothing,
    xticks_pants = nothing,
    xticks_genus = nothing,
    yticks_grid = nothing,
    yticks_quasicrystal = nothing,
    yticks_layered = nothing,
    yticks_random = nothing,
    yticks_merged = nothing,
    yticks_destroyed = nothing,
    yticks_pants = nothing,
    yticks_genus = nothing,
    colorbar_ticks_layered = nothing,
    colorbar_ticks_merged = nothing,
    colorbar_ticks_destroyed = nothing,
    colorbar_ticks_pants = nothing,
    colorbar_ticks_genus = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    double_column::Bool = true,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 8.0,
    colgap::Union{Nothing,Real} = 12.0,
    num_bins::Union{Nothing,Int} = nothing,
    log_colorbar::Union{Nothing,Bool} = nothing,
    log_colorbar_layered::Union{Nothing,Bool} = nothing,
    log_colorbar_merged::Union{Nothing,Bool} = nothing,
    log_colorbar_destroyed::Union{Nothing,Bool} = nothing,
    invert_color_scaling_layered::Union{Nothing,Bool} = nothing,
    invert_color_scaling_deffect::Union{Nothing,Bool} = nothing,
    invert_color_scaling_topo::Union{Nothing,Bool} = nothing,
    invert_plot_order_layered::Bool = false,
    invert_plot_order_deffect::Bool = false,
    invert_plot_order_topo::Bool = false,
    comp_color = :black,
    comp_linewidth::Union{Nothing,Real} = nothing,
    layered_colormap = :viridis,
    deffect_colormap = :tarn,
    topo_colormap = :delta,
    return_axis::Bool = false,
    verbose::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, Matrix{CairoMakie.Axis}}}
    loaded = load_graph_observable_kind_plot_matrix_data(
        observable;
        size = size,
        grid_kind = grid_kind,
        quasicrystal_kind = quasicrystal_kind,
        layered_kind = layered_kind,
        random_kind = random_kind,
        merged_kind = merged_kind,
        destroyed_kind = destroyed_kind,
        pants_kind = pants_kind,
        genus_kind = genus_kind,
        grid_paths = grid_paths,
        quasicrystal_paths = quasicrystal_paths,
        layered_paths = layered_paths,
        random_paths = random_paths,
        merged_paths = merged_paths,
        destroyed_paths = destroyed_paths,
        pants_paths = pants_paths,
        genus_paths = genus_paths,
        standard_comp_paths = standard_comp_paths,
        quasicrystal_comp_paths = quasicrystal_comp_paths,
        num_bins = num_bins,
        log_colorbar = log_colorbar,
        log_colorbar_layered = log_colorbar_layered,
        log_colorbar_merged = log_colorbar_merged,
        log_colorbar_destroyed = log_colorbar_destroyed,
        verbose = verbose,
    )
    return graph_observable_kind_plot_matrix(
        loaded,
        fig_path;
        title_grid = title_grid,
        title_quasicrystal = title_quasicrystal,
        title_layered = title_layered,
        title_random = title_random,
        title_merged = title_merged,
        title_destroyed = title_destroyed,
        title_pants = title_pants,
        title_genus = title_genus,
        xlim_grid = xlim_grid,
        xlim_quasicrystal = xlim_quasicrystal,
        xlim_layered = xlim_layered,
        xlim_random = xlim_random,
        xlim_merged = xlim_merged,
        xlim_destroyed = xlim_destroyed,
        xlim_pants = xlim_pants,
        xlim_genus = xlim_genus,
        ylim_grid = ylim_grid,
        ylim_quasicrystal = ylim_quasicrystal,
        ylim_layered = ylim_layered,
        ylim_random = ylim_random,
        ylim_merged = ylim_merged,
        ylim_destroyed = ylim_destroyed,
        ylim_pants = ylim_pants,
        ylim_genus = ylim_genus,
        xlabel_grid = xlabel_grid,
        xlabel_quasicrystal = xlabel_quasicrystal,
        xlabel_layered = xlabel_layered,
        xlabel_random = xlabel_random,
        xlabel_merged = xlabel_merged,
        xlabel_destroyed = xlabel_destroyed,
        xlabel_pants = xlabel_pants,
        xlabel_genus = xlabel_genus,
        ylabel_grid = ylabel_grid,
        ylabel_quasicrystal = ylabel_quasicrystal,
        ylabel_layered = ylabel_layered,
        ylabel_random = ylabel_random,
        ylabel_merged = ylabel_merged,
        ylabel_destroyed = ylabel_destroyed,
        ylabel_pants = ylabel_pants,
        ylabel_genus = ylabel_genus,
        xticks_grid = xticks_grid,
        xticks_quasicrystal = xticks_quasicrystal,
        xticks_layered = xticks_layered,
        xticks_random = xticks_random,
        xticks_merged = xticks_merged,
        xticks_destroyed = xticks_destroyed,
        xticks_pants = xticks_pants,
        xticks_genus = xticks_genus,
        yticks_grid = yticks_grid,
        yticks_quasicrystal = yticks_quasicrystal,
        yticks_layered = yticks_layered,
        yticks_random = yticks_random,
        yticks_merged = yticks_merged,
        yticks_destroyed = yticks_destroyed,
        yticks_pants = yticks_pants,
        yticks_genus = yticks_genus,
        colorbar_ticks_layered = colorbar_ticks_layered,
        colorbar_ticks_merged = colorbar_ticks_merged,
        colorbar_ticks_destroyed = colorbar_ticks_destroyed,
        colorbar_ticks_pants = colorbar_ticks_pants,
        colorbar_ticks_genus = colorbar_ticks_genus,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        double_column = double_column,
        magnification = magnification,
        plot_std = plot_std,
        right_yaxis = right_yaxis,
        top_xaxis = top_xaxis,
        rowgap = rowgap,
        colgap = colgap,
        log_colorbar = log_colorbar,
        log_colorbar_layered = log_colorbar_layered,
        log_colorbar_merged = log_colorbar_merged,
        log_colorbar_destroyed = log_colorbar_destroyed,
        invert_color_scaling_layered = invert_color_scaling_layered,
        invert_color_scaling_deffect = invert_color_scaling_deffect,
        invert_color_scaling_topo = invert_color_scaling_topo,
        invert_plot_order_layered = invert_plot_order_layered,
        invert_plot_order_deffect = invert_plot_order_deffect,
        invert_plot_order_topo = invert_plot_order_topo,
        layered_colormap = layered_colormap,
        deffect_colormap = deffect_colormap,
        topo_colormap = topo_colormap,
        comp_color = comp_color,
        comp_linewidth = comp_linewidth,
        return_axis = return_axis,
    )
end

function graph_observable_kind_plot_matrix(
    loaded::NamedTuple,
    fig_path::String;
    title_grid = nothing,
    title_quasicrystal = nothing,
    title_layered = nothing,
    title_random = nothing,
    title_merged = nothing,
    title_destroyed = nothing,
    title_pants = nothing,
    title_genus = nothing,
    xlim_grid = nothing,
    xlim_quasicrystal = nothing,
    xlim_layered = nothing,
    xlim_random = nothing,
    xlim_merged = nothing,
    xlim_destroyed = nothing,
    xlim_pants = nothing,
    xlim_genus = nothing,
    ylim_grid = nothing,
    ylim_quasicrystal = nothing,
    ylim_layered = nothing,
    ylim_random = nothing,
    ylim_merged = nothing,
    ylim_destroyed = nothing,
    ylim_pants = nothing,
    ylim_genus = nothing,
    xlabel_grid = nothing,
    xlabel_quasicrystal = nothing,
    xlabel_layered = nothing,
    xlabel_random = nothing,
    xlabel_merged = nothing,
    xlabel_destroyed = nothing,
    xlabel_pants = nothing,
    xlabel_genus = nothing,
    ylabel_grid = nothing,
    ylabel_quasicrystal = nothing,
    ylabel_layered = nothing,
    ylabel_random = nothing,
    ylabel_merged = nothing,
    ylabel_destroyed = nothing,
    ylabel_pants = nothing,
    ylabel_genus = nothing,
    xticks_grid = nothing,
    xticks_quasicrystal = nothing,
    xticks_layered = nothing,
    xticks_random = nothing,
    xticks_merged = nothing,
    xticks_destroyed = nothing,
    xticks_pants = nothing,
    xticks_genus = nothing,
    yticks_grid = nothing,
    yticks_quasicrystal = nothing,
    yticks_layered = nothing,
    yticks_random = nothing,
    yticks_merged = nothing,
    yticks_destroyed = nothing,
    yticks_pants = nothing,
    yticks_genus = nothing,
    colorbar_ticks_layered = nothing,
    colorbar_ticks_merged = nothing,
    colorbar_ticks_destroyed = nothing,
    colorbar_ticks_pants = nothing,
    colorbar_ticks_genus = nothing,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    double_column::Bool = true,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 8.0,
    colgap::Union{Nothing,Real} = 12.0,
    log_colorbar::Union{Nothing,Bool} = nothing,
    log_colorbar_layered::Union{Nothing,Bool} = nothing,
    log_colorbar_merged::Union{Nothing,Bool} = nothing,
    log_colorbar_destroyed::Union{Nothing,Bool} = nothing,
    invert_color_scaling_layered::Union{Nothing,Bool} = nothing,
    invert_color_scaling_deffect::Union{Nothing,Bool} = nothing,
    invert_color_scaling_topo::Union{Nothing,Bool} = nothing,
    invert_plot_order_layered::Bool = false,
    invert_plot_order_deffect::Bool = false,
    invert_plot_order_topo::Bool = false,
    layered_colormap = :viridis,
    deffect_colormap = :tarn,
    topo_colormap = :delta,
    comp_color = :black,
    comp_linewidth::Union{Nothing,Real} = nothing,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, Matrix{CairoMakie.Axis}}}
    if !(magnification > 0)
        throw(DomainError(magnification, "magnification must be > 0"))
    end
    rowgap !== nothing && rowgap < 0 && throw(DomainError(rowgap, "rowgap must be >= 0"))
    colgap !== nothing && colgap < 0 && throw(DomainError(colgap, "colgap must be >= 0"))

    observable_spec = graph_observable_field_spec(loaded.observable)
    panel_specs = loaded.panel_specs
    panels = loaded.panels
    loaded_panel_log_binning = loaded.panel_log_colorbar
    requested_panel_log_binning = [
        nothing,
        nothing,
        log_colorbar_layered,
        nothing,
        log_colorbar_merged,
        log_colorbar_destroyed,
        nothing,
        nothing,
    ]
    panel_log_binning = [
        isnothing(requested_panel_log_binning[i]) ?
            (isnothing(log_colorbar) ? loaded_panel_log_binning[i] : log_colorbar) :
            requested_panel_log_binning[i]
        for i in eachindex(loaded_panel_log_binning)
    ]
    panel_invert_color_scaling = [
        false,
        false,
        isnothing(invert_color_scaling_layered) ? panel_specs[3].invert : invert_color_scaling_layered,
        false,
        isnothing(invert_color_scaling_deffect) ? panel_specs[5].invert : invert_color_scaling_deffect,
        isnothing(invert_color_scaling_deffect) ? panel_specs[6].invert : invert_color_scaling_deffect,
        isnothing(invert_color_scaling_topo) ? panel_specs[7].invert : invert_color_scaling_topo,
        isnothing(invert_color_scaling_topo) ? panel_specs[8].invert : invert_color_scaling_topo,
    ]
    panel_invert_plot_order = [
        false,
        false,
        invert_plot_order_layered,
        false,
        invert_plot_order_deffect,
        invert_plot_order_deffect,
        invert_plot_order_topo,
        invert_plot_order_topo,
    ]

    panel_meta = graph_observable_kind_plot_matrix_panel_meta(
        xlim_grid = xlim_grid,
        xlim_quasicrystal = xlim_quasicrystal,
        xlim_layered = xlim_layered,
        xlim_random = xlim_random,
        xlim_merged = xlim_merged,
        xlim_destroyed = xlim_destroyed,
        xlim_pants = xlim_pants,
        xlim_genus = xlim_genus,
        ylim_grid = ylim_grid,
        ylim_quasicrystal = ylim_quasicrystal,
        ylim_layered = ylim_layered,
        ylim_random = ylim_random,
        ylim_merged = ylim_merged,
        ylim_destroyed = ylim_destroyed,
        ylim_pants = ylim_pants,
        ylim_genus = ylim_genus,
        xlabel_grid = xlabel_grid,
        xlabel_quasicrystal = xlabel_quasicrystal,
        xlabel_layered = xlabel_layered,
        xlabel_random = xlabel_random,
        xlabel_merged = xlabel_merged,
        xlabel_destroyed = xlabel_destroyed,
        xlabel_pants = xlabel_pants,
        xlabel_genus = xlabel_genus,
        ylabel_grid = ylabel_grid,
        ylabel_quasicrystal = ylabel_quasicrystal,
        ylabel_layered = ylabel_layered,
        ylabel_random = ylabel_random,
        ylabel_merged = ylabel_merged,
        ylabel_destroyed = ylabel_destroyed,
        ylabel_pants = ylabel_pants,
        ylabel_genus = ylabel_genus,
        xticks_grid = xticks_grid,
        xticks_quasicrystal = xticks_quasicrystal,
        xticks_layered = xticks_layered,
        xticks_random = xticks_random,
        xticks_merged = xticks_merged,
        xticks_destroyed = xticks_destroyed,
        xticks_pants = xticks_pants,
        xticks_genus = xticks_genus,
        yticks_grid = yticks_grid,
        yticks_quasicrystal = yticks_quasicrystal,
        yticks_layered = yticks_layered,
        yticks_random = yticks_random,
        yticks_merged = yticks_merged,
        yticks_destroyed = yticks_destroyed,
        yticks_pants = yticks_pants,
        yticks_genus = yticks_genus,
        colorbar_ticks_layered = colorbar_ticks_layered,
        colorbar_ticks_merged = colorbar_ticks_merged,
        colorbar_ticks_destroyed = colorbar_ticks_destroyed,
        colorbar_ticks_pants = colorbar_ticks_pants,
        colorbar_ticks_genus = colorbar_ticks_genus,
    )
    title_n = [
        isnothing(title_grid) ? default_kind_plot_matrix_title(panel_specs[1].kind) : title_grid,
        isnothing(title_quasicrystal) ? default_kind_plot_matrix_title(panel_specs[2].kind) : title_quasicrystal,
        isnothing(title_layered) ? default_kind_plot_matrix_title(panel_specs[3].kind) : title_layered,
        isnothing(title_random) ? default_kind_plot_matrix_title(panel_specs[4].kind) : title_random,
        isnothing(title_merged) ? default_kind_plot_matrix_title(panel_specs[5].kind) : title_merged,
        isnothing(title_destroyed) ? default_kind_plot_matrix_title(panel_specs[6].kind) : title_destroyed,
        isnothing(title_pants) ? default_kind_plot_matrix_title(panel_specs[7].kind) : title_pants,
        isnothing(title_genus) ? default_kind_plot_matrix_title(panel_specs[8].kind) : title_genus,
    ]
    xlim_n = panel_meta.xlim
    ylim_n = panel_meta.ylim
    xlabel_n = Any[panel_meta.xlabel...]
    default_bottom_xlabel = loaded.observable in (:cardinalities, :ev_sym_link) ? L"j/n" : L"j"
    for idx in 7:8
        isnothing(xlabel_n[idx]) && (xlabel_n[idx] = default_bottom_xlabel)
    end
    ylabel_n = panel_meta.ylabel
    xticks_n = panel_meta.xticks
    yticks_n = panel_meta.yticks
    colorbar_ticks_n = panel_meta.colorbar_ticks

    base_size = apply_paper_theme!(double_column = double_column, magnification = magnification)
    row_has_colorbar = [
        any(!isnothing(panel_specs[2 * (row - 1) + col].scalar) for col in 1:2)
        for row in 1:4
    ]
    base_row_height = 0.54 * base_size[2]
    colorbar_row_extra = 70
    row_heights = [
        row_has_colorbar[row] ? base_row_height + colorbar_row_extra : base_row_height
        for row in 1:4
    ]
    outer_rowgap = isnothing(rowgap) ? 0.0 : float(rowgap)
    top_padding = 100
    bottom_padding = 100
    fig_height = sum(row_heights) + 3 * outer_rowgap + top_padding + bottom_padding
    fig = CairoMakie.Figure(
        size = (base_size[1], fig_height),
        figure_padding = (20, 20, top_padding, bottom_padding),
    )
    axs = Matrix{CairoMakie.Axis}(undef, 4, 2)
    panel_layouts = Matrix{CairoMakie.GridLayout}(undef, 4, 2)

    for row in 1:4
        for col in 1:2
            idx = 2 * (row - 1) + col
            panel_layout = CairoMakie.GridLayout(fig[row, col])
            CairoMakie.rowgap!(panel_layout, 4)
            ax = CairoMakie.Axis(
                panel_layout[1, 1];
                xscale = logscale_x ? log10 : identity,
                yscale = logscale_y ? log10 : identity,
            )
            apply_axis_scale_theme!(ax; logscale_x = logscale_x, logscale_y = logscale_y)
            ax.title = title_n[idx]
            if top_xaxis && row == 1
                ax.xaxisposition = :top
                ax.xticklabelalign = (:center, :bottom)
                ax_bottom = CairoMakie.Axis(
                    panel_layout[1, 1];
                    xscale = ax.xscale[],
                    yscale = ax.yscale[],
                    xlabelvisible = false,
                    xticklabelsvisible = false,
                    yticklabelsvisible = false,
                    yticksvisible = false,
                    ylabelvisible = false,
                    xgridvisible = false,
                    xminorgridvisible = false,
                    ygridvisible = false,
                    yminorgridvisible = false,
                    backgroundcolor = :transparent,
                )
                apply_axis_scale_theme!(
                    ax_bottom;
                    logscale_x = ax.xscale[] !== identity,
                    logscale_y = ax.yscale[] !== identity,
                )
                ax_bottom.xaxisposition = :bottom
                ax_bottom.topspinevisible = false
            end
            if right_yaxis && col == 2
                ax.yaxisposition = :right
                ax.yticklabelalign = (:left, :center)
                ax.flip_ylabel = true
            end
            panel_layouts[row, col] = panel_layout
            axs[row, col] = ax
        end
    end

    rowgap !== nothing && CairoMakie.rowgap!(fig.layout, rowgap)
    CairoMakie.rowgap!(fig.layout, 2, -10)
    CairoMakie.rowgap!(fig.layout, 3, -10)
    colgap !== nothing && CairoMakie.colgap!(fig.layout, colgap)
    for row in 1:4
        CairoMakie.rowsize!(fig.layout, row, CairoMakie.Fixed(row_heights[row]))
    end

    for row in 1:4
        for col in 1:2
            idx = 2 * (row - 1) + col
            spec = panel_specs[idx]
            panel = panels[idx]
            ax = axs[row, col]
            panel_xscale =
                if loaded.observable in (:cardinalities, :ev_sym_link)
                    inv(Float64(loaded.size))
                else
                    hasproperty(panel, :xscale) ? panel.xscale : 1.0
                end
            panel_yscale = loaded.observable == :cardinalities ? 1.0 :
                (hasproperty(panel, :yscale) ? panel.yscale : 1.0)
            panel_data = scale_panel_series(panel.data, panel_yscale)
            panel_comp = scale_mean_std(panel.comp, panel_yscale)
            panel_colormap =
                if idx == 3
                    layered_colormap
                elseif idx == 5 || idx == 6
                    deffect_colormap
                elseif idx == 7 || idx == 8
                    topo_colormap
                else
                    :viridis
                end
            if isnothing(spec.scalar)
                plot_hist_or_vec_panel!(
                    ax,
                    panel_data,
                    panel_comp,
                    xlim_n[idx],
                    ylim_n[idx],
                    xlabel_n[idx],
                    ylabel_n[idx],
                    xticks_n[idx],
                    yticks_n[idx];
                    logscale_y = logscale_y,
                    invert_color_scaling = false,
                    invert_plot_order = false,
                    log_color_scaling = false,
                    plot_std = plot_std,
                    vmin = 0.0,
                    denom = 1.0,
                    colormap = panel_colormap,
                    comp_color = comp_color,
                    comp_linewidth = comp_linewidth,
                    xscale = panel_xscale,
                )
            else
                color_scale = compute_color_scale([panel.data]; log_color_scaling = panel_log_binning[idx])
                plot_hist_or_vec_panel!(
                    ax,
                    panel_data,
                    panel_comp,
                    xlim_n[idx],
                    ylim_n[idx],
                    xlabel_n[idx],
                    ylabel_n[idx],
                    xticks_n[idx],
                    yticks_n[idx];
                    logscale_y = logscale_y,
                    invert_color_scaling = panel_invert_color_scaling[idx],
                    invert_plot_order = panel_invert_plot_order[idx],
                    log_color_scaling = panel_log_binning[idx],
                    plot_std = plot_std,
                    vmin = color_scale.vmin,
                    denom = color_scale.denom,
                    colormap = panel_colormap,
                    comp_color = comp_color,
                    comp_linewidth = comp_linewidth,
                    xscale = panel_xscale,
                )
                cb_layout = CairoMakie.GridLayout(panel_layouts[row, col][2, 1])
                CairoMakie.rowgap!(panel_layouts[row, col], 0)
                CairoMakie.rowsize!(panel_layouts[row, col], 2, CairoMakie.Fixed(30))
                cb = CairoMakie.Colorbar(
                    cb_layout[1, 1],
                    limits = (color_scale.raw_vmin, color_scale.raw_vmax),
                    colormap = panel_invert_color_scaling[idx] ? CairoMakie.Reverse(panel_colormap) : panel_colormap,
                    vertical = false,
                    scale = panel_log_binning[idx] ? log10 : identity,
                )
                cb.flipaxis = false
                cb.height = 5
                cb.ticksize = 3
                cb.minorticksize = 2
                cb.ticklabelpad = row == 3 ? 4 : 2
                cb.label = graph_observable_scalar_label(spec.scalar)
                cb.labelpadding = row == 3 ? 0 : -5
                if colorbar_ticks_n[idx] !== nothing
                    cb.ticks = colorbar_ticks_n[idx]
                end
            end
        end
    end

    CairoMakie.save(fig_path, fig)
    return return_axis ? (fig, axs) : fig
end

function load_graph_observable_average_std(
    paths_in,
    observable::Symbol;
    size::Union{Nothing,Integer} = nothing,
    normalize_x_by_size::Bool = false,
    verbose::Bool = false,
)
    spec = graph_observable_field_spec(observable)
    paths = paths_in isa AbstractString ? [String(paths_in)] : String.(collect(paths_in))
    if spec.hist
        if normalize_x_by_size
            size === nothing && throw(ArgumentError("size is required when normalize_x_by_size = true for histogram observables"))
        end
        values = load_histograms_from_paths(paths, spec.field; verbose = verbose)[1]
        prepared = normalize_hists(
            [values];
            normalization = spec.normalization,
            normalize_x_axis_with_size = normalize_x_by_size,
            cset_size = size,
        )[1]
        mean, std = average_histogram_with_std(prepared)
    else
        values = load_fields_from_paths(paths, [spec.field]; verbose = verbose)[1][1]
        mean, std = average_vectors_with_std(values)
    end
    return (mean = mean, std = std)
end

function default_size_boundary_dimension_plot_matrix_title(panel::Symbol)
    if panel == :size
        return "sizes"
    elseif panel == :boundary
        return "boundaries"
    elseif panel == :dimension
        return "dimensionality"
    else
        return String(panel)
    end
end

function default_graph_observable_ylabel(observable::Symbol)
    if observable == :connectivity_link
        return L"P_{j}"
    elseif observable == :cardinalities
        return L"\mathcal{S}_{j}"
    elseif observable == :max_pathlen
        return L"\mathcal{H}_{j}"
    elseif observable == :ev_sym_link
        return L"\lambda_j"
    else
        return nothing
    end
end

function rich_title_legend(panel::Symbol, series_labels::AbstractVector{<:AbstractString}, series_colors)
    if panel == :size
        parts = Any["sizes n = "]
    elseif panel == :boundary
        parts = Any["boundaries: "]
    elseif panel == :dimension
        parts = Any["dimensionality d = "]
    else
        parts = Any[""]
    end
    for i in eachindex(series_labels)
        i > 1 && push!(parts, ", ")
        push!(parts, CairoMakie.rich(series_labels[i], color = series_colors[i]))
    end
    return CairoMakie.rich(parts...)
end

function plot_labeled_average_std_panel!(
    ax,
    series::AbstractVector;
    xlim_i,
    ylim_i,
    xlabel_i,
    ylabel_i,
    xticks_i,
    yticks_i,
    logscale_x::Bool,
    logscale_y::Bool,
    plot_std::Bool,
    show_legend::Bool,
    legend_position,
    legend_columns::Int,
)
    apply_axis_metadata!(ax, xlim_i, ylim_i, xlabel_i, ylabel_i, xticks_i, yticks_i)

    eps = if logscale_y
        if ylim_i !== nothing
            ylim_i[1] * 1e-3
        else
            minpos = minimum(v for s in series for v in s.mean if v > 0)
            minpos * 1e-3
        end
    else
        -Inf
    end

    for s in series
        x = hasproperty(s, :x) ? collect(s.x) : collect(1:length(s.mean))
        mean = s.mean
        ylo = s.mean .- s.std
        yhi = s.mean .+ s.std
        if logscale_x
            maskx = x .> 0
            x = x[maskx]
            mean = mean[maskx]
            ylo = ylo[maskx]
            yhi = yhi[maskx]
        end
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
            CairoMakie.band!(ax, x, ylo, yhi; color = (s.color, 0.18))
        end
        CairoMakie.lines!(ax, x, mean; color = s.color, label = s.label)
    end

    if show_legend
        if legend_position isa Tuple && length(legend_position) == 2
            leg = CairoMakie.axislegend(ax; position = :cc, nbanks = legend_columns)
            leg.halign = legend_position[1]
            leg.valign = legend_position[2]
        else
            CairoMakie.axislegend(ax; position = legend_position, nbanks = legend_columns)
        end
    end
    return nothing
end

function size_boundary_dimension_series_for_panel(
    panel_series,
    colors,
    panel::Symbol,
    loaded,
    normalize_x_by_size::Bool,
    invert_plot_order::Bool,
)
    is_hist = graph_observable_field_spec(loaded.observable).hist
    series = [
        begin
            size_denom = if panel == :size
                s.size
            elseif panel == :boundary
                loaded.boundary_size
            else
                loaded.dimension_size
            end
            (; label = s.label,
               mean = normalize_x_by_size && is_hist ? s.mean[2:end] : s.mean,
               std = normalize_x_by_size && is_hist ? s.std[2:end] : s.std,
               color = colors[j],
               x = if normalize_x_by_size
                   if is_hist
                       collect(1:length(s.mean)-1) ./ size_denom
                   else
                       collect(1:length(s.mean)) ./ size_denom
                   end
               else
                   if is_hist
                       collect(0:length(s.mean)-1)
                   else
                       collect(1:length(s.mean))
                   end
               end)
        end
        for (j, s) in enumerate(panel_series)
    ]
    return invert_plot_order ? reverse(series) : series
end

function normalize_size_boundary_dimension_num_csets(
    sizes::AbstractVector{<:Integer},
    num_csets::Union{Integer,AbstractVector{<:Integer}},
)
    nsizes = length(sizes)
    if num_csets isa Integer
        size_num_csets = fill(Int(num_csets), nsizes)
        shared = Int(num_csets)
        return (;
            size_num_csets,
            boundary_manifold_num_csets = shared,
            boundary_sprinkling_num_csets = shared,
            dimension_2d_num_csets = shared,
            dimension_3d_num_csets = shared,
        )
    end

    vals = Int.(collect(num_csets))
    expected = nsizes + 4
    if length(vals) != expected
        throw(ArgumentError("vector num_csets must have length $(expected): $(nsizes) size entries plus 4 comparison entries"))
    end
    return (;
        size_num_csets = vals[1:nsizes],
        boundary_manifold_num_csets = vals[nsizes + 1],
        boundary_sprinkling_num_csets = vals[nsizes + 2],
        dimension_2d_num_csets = vals[nsizes + 3],
        dimension_3d_num_csets = vals[nsizes + 4],
    )
end

function load_graph_observable_size_boundary_dimension_plot_matrix_data(
    observable::Symbol;
    sizes::AbstractVector{<:Integer} = [256, 300, 400, 512, 600, 700, 800, 900, 1024, 1200, 1500, 1800, 2048],
    num_csets::Union{Integer,AbstractVector{<:Integer}} = 10000,
    size_kind::String = "manifoldlike_simply_connected",
    size_paths = nothing,
    boundary_manifold_kind::String = "manifoldlike_simply_connected",
    boundary_sprinkling_kind::String = "minkowski_sprinkling",
    boundary_size::Int = 2048,
    boundary_manifold_paths = nothing,
    boundary_sprinkling_paths = nothing,
    dimension_2d_kind::String = "manifoldlike_simply_connected",
    dimension_3d_kind::String = "manifoldlike_simply_connected_3D",
    dimension_size::Int = 2048,
    dimension_2d_paths = nothing,
    dimension_3d_paths = nothing,
    normalize_x_by_size::Bool = false,
    verbose::Bool = false,
)
    isempty(sizes) && throw(ArgumentError("sizes must be non-empty"))
    counts = normalize_size_boundary_dimension_num_csets(sizes, num_csets)
    size_paths === nothing && (size_paths = [
        data_paths(["$(size_kind)_$(s)_$(counts.size_num_csets[i])/statistics.jld2"])
        for (i, s) in enumerate(sizes)
    ])
    boundary_manifold_paths === nothing && (boundary_manifold_paths = data_paths([
        "$(boundary_manifold_kind)_$(boundary_size)_$(counts.boundary_manifold_num_csets)/statistics.jld2"
    ]))
    boundary_sprinkling_paths === nothing && (boundary_sprinkling_paths = data_paths([
        "$(boundary_sprinkling_kind)_$(boundary_size)_$(counts.boundary_sprinkling_num_csets)/statistics.jld2"
    ]))
    dimension_2d_paths === nothing && (dimension_2d_paths = data_paths([
        "$(dimension_2d_kind)_$(dimension_size)_$(counts.dimension_2d_num_csets)/statistics.jld2"
    ]))
    dimension_3d_paths === nothing && (dimension_3d_paths = data_paths([
        "$(dimension_3d_kind)_$(dimension_size)_$(counts.dimension_3d_num_csets)/statistics.jld2"
    ]))

    size_labels = ["n=$(s)" for s in sizes]
    size_series = [
        (; label = size_labels[i], size = Int(sizes[i]), load_graph_observable_average_std(size_paths[i], observable; size = Int(sizes[i]), normalize_x_by_size = normalize_x_by_size, verbose = verbose)...)
        for i in eachindex(sizes)
    ]
    boundary_series = [
        (; label = "box", load_graph_observable_average_std(boundary_manifold_paths, observable; size = boundary_size, normalize_x_by_size = normalize_x_by_size, verbose = verbose)...),
        (; label = "diamond", load_graph_observable_average_std(boundary_sprinkling_paths, observable; size = boundary_size, normalize_x_by_size = normalize_x_by_size, verbose = verbose)...),
    ]
    dimension_series = [
        (; label = L"d=2", load_graph_observable_average_std(dimension_2d_paths, observable; size = dimension_size, normalize_x_by_size = normalize_x_by_size, verbose = verbose)...),
        (; label = L"d=3", load_graph_observable_average_std(dimension_3d_paths, observable; size = dimension_size, normalize_x_by_size = normalize_x_by_size, verbose = verbose)...),
    ]

    return (
        observable = observable,
        sizes = Int.(sizes),
        boundary_size = Int(boundary_size),
        dimension_size = Int(dimension_size),
        panel_specs = [
            (panel = :size,),
            (panel = :boundary,),
            (panel = :dimension,),
        ],
        panels = [size_series, boundary_series, dimension_series],
    )
end

function graph_observable_size_boundary_dimension_plot_matrix(
    observable::Symbol,
    fig_path::String;
    sizes::AbstractVector{<:Integer} = [256, 300, 400, 512, 600, 700, 800, 900, 1024, 1200, 1500, 1800, 2048],
    num_csets::Union{Integer,AbstractVector{<:Integer}} = 10000,
    size_kind::String = "manifoldlike_simply_connected",
    size_paths = nothing,
    boundary_manifold_kind::String = "manifoldlike_simply_connected",
    boundary_sprinkling_kind::String = "minkowski_sprinkling",
    boundary_size::Int = 2048,
    boundary_manifold_paths = nothing,
    boundary_sprinkling_paths = nothing,
    dimension_2d_kind::String = "manifoldlike_simply_connected",
    dimension_3d_kind::String = "manifoldlike_simply_connected_3D",
    dimension_size::Int = 2048,
    dimension_2d_paths = nothing,
    dimension_3d_paths = nothing,
    title_size = nothing,
    title_boundary = nothing,
    title_dimension = nothing,
    xlim_size = nothing,
    xlim_boundary = nothing,
    xlim_dimension = nothing,
    ylim_size = nothing,
    ylim_boundary = nothing,
    ylim_dimension = nothing,
    xlabel_size = L"j",
    xlabel_boundary = L"j",
    xlabel_dimension = L"j",
    normalize_x_by_size::Bool = false,
    ylabel = nothing,
    xticks_size = nothing,
    xticks_boundary = nothing,
    xticks_dimension = nothing,
    yticks_size = nothing,
    yticks_boundary = nothing,
    yticks_dimension = nothing,
    yticklabels_left::Bool = true,
    yticklabels_middle::Bool = true,
    yticklabels_right::Bool = true,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    double_column::Bool = true,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 8.0,
    colgap::Union{Nothing,Real} = 12.0,
    legend_position_size = :rt,
    legend_position_boundary = :rt,
    legend_position_dimension = :rt,
    legend_columns_size::Int = 1,
    legend_columns_boundary::Int = 1,
    legend_columns_dimension::Int = 1,
    invert_plot_order_size::Bool = false,
    invert_plot_order_boundary::Bool = false,
    invert_plot_order_dimension::Bool = false,
    title_legend::Bool = false,
    return_axis::Bool = false,
    verbose::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, Vector{CairoMakie.Axis}}}
    loaded = load_graph_observable_size_boundary_dimension_plot_matrix_data(
        observable;
        sizes = sizes,
        num_csets = num_csets,
        size_kind = size_kind,
        size_paths = size_paths,
        boundary_manifold_kind = boundary_manifold_kind,
        boundary_sprinkling_kind = boundary_sprinkling_kind,
        boundary_size = boundary_size,
        boundary_manifold_paths = boundary_manifold_paths,
        boundary_sprinkling_paths = boundary_sprinkling_paths,
        dimension_2d_kind = dimension_2d_kind,
        dimension_3d_kind = dimension_3d_kind,
        dimension_size = dimension_size,
        dimension_2d_paths = dimension_2d_paths,
        dimension_3d_paths = dimension_3d_paths,
        normalize_x_by_size = normalize_x_by_size,
        verbose = verbose,
    )
    return graph_observable_size_boundary_dimension_plot_matrix(
        loaded,
        fig_path;
        title_size = title_size,
        title_boundary = title_boundary,
        title_dimension = title_dimension,
        xlim_size = xlim_size,
        xlim_boundary = xlim_boundary,
        xlim_dimension = xlim_dimension,
        ylim_size = ylim_size,
        ylim_boundary = ylim_boundary,
        ylim_dimension = ylim_dimension,
        xlabel_size = xlabel_size,
        xlabel_boundary = xlabel_boundary,
        xlabel_dimension = xlabel_dimension,
        normalize_x_by_size = normalize_x_by_size,
        ylabel = ylabel,
        xticks_size = xticks_size,
        xticks_boundary = xticks_boundary,
        xticks_dimension = xticks_dimension,
        yticks_size = yticks_size,
        yticks_boundary = yticks_boundary,
        yticks_dimension = yticks_dimension,
        yticklabels_left = yticklabels_left,
        yticklabels_middle = yticklabels_middle,
        yticklabels_right = yticklabels_right,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        double_column = double_column,
        magnification = magnification,
        plot_std = plot_std,
        right_yaxis = right_yaxis,
        top_xaxis = top_xaxis,
        rowgap = rowgap,
        colgap = colgap,
        legend_position_size = legend_position_size,
        legend_position_boundary = legend_position_boundary,
        legend_position_dimension = legend_position_dimension,
        legend_columns_size = legend_columns_size,
        legend_columns_boundary = legend_columns_boundary,
        legend_columns_dimension = legend_columns_dimension,
        invert_plot_order_size = invert_plot_order_size,
        invert_plot_order_boundary = invert_plot_order_boundary,
        invert_plot_order_dimension = invert_plot_order_dimension,
        title_legend = title_legend,
        return_axis = return_axis,
    )
end

function graph_observable_size_boundary_dimension_plot_matrix(
    loaded::NamedTuple,
    fig_path::String;
    title_size = nothing,
    title_boundary = nothing,
    title_dimension = nothing,
    xlim_size = nothing,
    xlim_boundary = nothing,
    xlim_dimension = nothing,
    ylim_size = nothing,
    ylim_boundary = nothing,
    ylim_dimension = nothing,
    xlabel_size = nothing,
    xlabel_boundary = nothing,
    xlabel_dimension = nothing,
    normalize_x_by_size::Bool = false,
    ylabel = nothing,
    xticks_size = nothing,
    xticks_boundary = nothing,
    xticks_dimension = nothing,
    yticks_size = nothing,
    yticks_boundary = nothing,
    yticks_dimension = nothing,
    yticklabels_left::Bool = true,
    yticklabels_middle::Bool = true,
    yticklabels_right::Bool = true,
    logscale_x::Bool = true,
    logscale_y::Bool = true,
    double_column::Bool = true,
    magnification::Real = 1.0,
    plot_std::Bool = true,
    right_yaxis::Bool = true,
    top_xaxis::Bool = true,
    rowgap::Union{Nothing,Real} = 8.0,
    colgap::Union{Nothing,Real} = 12.0,
    legend_position_size = :rt,
    legend_position_boundary = :rt,
    legend_position_dimension = :rt,
    legend_columns_size::Int = 1,
    legend_columns_boundary::Int = 1,
    legend_columns_dimension::Int = 1,
    invert_plot_order_size::Bool = false,
    invert_plot_order_boundary::Bool = false,
    invert_plot_order_dimension::Bool = false,
    title_legend::Bool = false,
    return_axis::Bool = false,
)::Union{CairoMakie.Figure, Tuple{CairoMakie.Figure, Vector{CairoMakie.Axis}}}
    if !(magnification > 0)
        throw(DomainError(magnification, "magnification must be > 0"))
    end
    rowgap !== nothing && rowgap < 0 && throw(DomainError(rowgap, "rowgap must be >= 0"))
    colgap !== nothing && colgap < 0 && throw(DomainError(colgap, "colgap must be >= 0"))

    panel_specs = loaded.panel_specs
    panels = loaded.panels
    min_positive_normalized_x = [
        minimum(1 / s.size for s in panels[1]),
        1 / loaded.boundary_size,
        1 / loaded.dimension_size,
    ]
    title_n = [
        isnothing(title_size) ? default_size_boundary_dimension_plot_matrix_title(panel_specs[1].panel) : title_size,
        isnothing(title_boundary) ? default_size_boundary_dimension_plot_matrix_title(panel_specs[2].panel) : title_boundary,
        isnothing(title_dimension) ? default_size_boundary_dimension_plot_matrix_title(panel_specs[3].panel) : title_dimension,
    ]
    xlim_n = if normalize_x_by_size
        [
            isnothing(xlim_size) ? ((logscale_x ? min_positive_normalized_x[1] : 0.0), 1.0) : xlim_size,
            isnothing(xlim_boundary) ? ((logscale_x ? min_positive_normalized_x[2] : 0.0), 1.0) : xlim_boundary,
            isnothing(xlim_dimension) ? ((logscale_x ? min_positive_normalized_x[3] : 0.0), 1.0) : xlim_dimension,
        ]
    else
        [xlim_size, xlim_boundary, xlim_dimension]
    end
    if logscale_x
        xlim_n = [
            if isnothing(xl)
                nothing
            else
                lo, hi = xl
                lo <= 0 && normalize_x_by_size ? (min_positive_normalized_x[i], hi) : xl
            end
            for (i, xl) in enumerate(xlim_n)
        ]
    end
    ylim_n = [ylim_size, ylim_boundary, ylim_dimension]
    default_xlabel = normalize_x_by_size ? L"j / n" : L"j"
    xlabel_n = [
        isnothing(xlabel_size) ? default_xlabel : xlabel_size,
        isnothing(xlabel_boundary) ? default_xlabel : xlabel_boundary,
        isnothing(xlabel_dimension) ? default_xlabel : xlabel_dimension,
    ]
    ylabel_left = isnothing(ylabel) ? default_graph_observable_ylabel(loaded.observable) : ylabel
    ylabel_n = [ylabel_left, nothing, nothing]
    xticks_n = [xticks_size, xticks_boundary, xticks_dimension]
    yticks_n = [yticks_size, yticks_boundary, yticks_dimension]
    yticklabels_n = [yticklabels_left, yticklabels_middle, yticklabels_right]
    legend_positions = [legend_position_size, legend_position_boundary, legend_position_dimension]
    legend_columns = [legend_columns_size, legend_columns_boundary, legend_columns_dimension]
    invert_plot_order_n = [invert_plot_order_size, invert_plot_order_boundary, invert_plot_order_dimension]

    panel_size = apply_paper_theme!(double_column = double_column, third_line_size = true, magnification = magnification)
    fig_width = 3 * panel_size[1] + 2 * (isnothing(colgap) ? 0.0 : float(colgap))
    fig_height = panel_size[2]
    fig = CairoMakie.Figure(size = (fig_width, fig_height))
    axs = Vector{CairoMakie.Axis}(undef, 3)

    rowgap !== nothing && CairoMakie.rowgap!(fig.layout, rowgap)
    colgap !== nothing && CairoMakie.colgap!(fig.layout, colgap)

    for idx in 1:3
        ax = CairoMakie.Axis(
            fig[1, idx];
            xscale = logscale_x ? log10 : identity,
            yscale = logscale_y ? log10 : identity,
        )
        apply_axis_scale_theme!(ax; logscale_x = logscale_x, logscale_y = logscale_y)
        ax.title = title_n[idx]
        ax.xaxisposition = :bottom
        if right_yaxis && idx == 3
            ax.yaxisposition = :right
            ax.yticklabelalign = (:left, :center)
            ax.flip_ylabel = true
        end
        ax.yticklabelsvisible = yticklabels_n[idx]
        axs[idx] = ax
        if right_yaxis && idx == 3
            ax_left = CairoMakie.Axis(
                fig[1, idx];
                xscale = ax.xscale[],
                yscale = ax.yscale[],
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
            apply_axis_scale_theme!(
                ax_left;
                logscale_x = ax.xscale[] !== identity,
                logscale_y = ax.yscale[] !== identity,
            )
            ax_left.yaxisposition = :left
            ax_left.rightspinevisible = false
            CairoMakie.linkxaxes!(ax, ax_left)
            CairoMakie.linkyaxes!(ax, ax_left)
        end
    end

    theme_colors_obs = CairoMakie.theme(:palette).color
    theme_colors = theme_colors_obs isa Observables.Observable ? Observables.to_value(theme_colors_obs) : theme_colors_obs
    size_n = length(panels[1])
    size_colors = [theme_colors[mod1(i, length(theme_colors))] for i in 1:size_n]
    boundary_colors = [theme_colors[1], theme_colors[2]]
    dimension_colors = [theme_colors[1], theme_colors[2]]
    panel_colors = [size_colors, boundary_colors, dimension_colors]

    for idx in 1:3
        series = size_boundary_dimension_series_for_panel(
            panels[idx],
            panel_colors[idx],
            panel_specs[idx].panel,
            loaded,
            normalize_x_by_size,
            invert_plot_order_n[idx],
        )
        if title_legend
            title_labels =
                if panel_specs[idx].panel == :size
                    ["$(s.size)" for s in panels[idx]]
                elseif panel_specs[idx].panel == :boundary
                    [String(s.label) for s in panels[idx]]
                else
                    ["2", "3"]
                end
            axs[idx].title = rich_title_legend(panel_specs[idx].panel, title_labels, panel_colors[idx])
        end
        plot_labeled_average_std_panel!(
            axs[idx],
            series;
            xlim_i = xlim_n[idx],
            ylim_i = ylim_n[idx],
            xlabel_i = xlabel_n[idx],
            ylabel_i = ylabel_n[idx],
            xticks_i = xticks_n[idx],
            yticks_i = yticks_n[idx],
            logscale_x = logscale_x,
            logscale_y = logscale_y,
            plot_std = plot_std,
            show_legend = !title_legend,
            legend_position = legend_positions[idx],
            legend_columns = legend_columns[idx],
        )
    end

    CairoMakie.save(fig_path, fig)
    return return_axis ? (fig, axs) : fig
end

"""
    distinguishability_plot_row(
        kind::String,
        scalar::Symbol,
        fig_path::String;
        size = 2048,
        ...
    )

Compute and plot scalar-conditioned distinguishability curves for one dataset kind
against its comparison dataset in a 1×3 row.

Panels:
1. energy-based distinguishability
2. mutual-information distinguishability
3. mahalanobis distinguishability

Each panel overlays the observables `cardinalities`, `ev_sym_link`,
`connectivity_link`, and `max_pathlen`. A shared legend is placed to the right
of the full row.
"""
function distinguishability_plot_row(
    kind::String,
    scalar::Symbol,
    fig_path::String;
    size::Int = 2048,
    comp_kind::Union{Nothing,String} = nothing,
    sqrt_scalars::Bool = false,
    num_bins::Union{Nothing,Int} = nothing,
    xlim::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    ylim_energy::Union{Nothing,Tuple{Float64,Float64}} = (0.0, 1.0),
    ylim_mutual_information::Union{Nothing,Tuple{Float64,Float64}} = (0.0, 1.0),
    ylim_mahalanobis::Union{Nothing,Tuple{Float64,Float64}} = (0.0, 1.0),
    xticks = nothing,
    yticks_energy = nothing,
    yticks_mutual_information = nothing,
    yticks_mahalanobis = nothing,
    regulator::Float64 = 0.0,
    R::Int = 1000,
    q::Float64 = 0.0,
    alpha::Float64 = 0.05,
    rng = Random.default_rng(),
    projection_tolerance::Float64 = 1e-6,
    to_regularize_rel::Float64 = 0.01,
    progress::Bool = false,
    energy_distance::Symbol = :Hellinger,
    mutual_information_k::Int = 5,
    mutual_information_pca_mode::Symbol = :cutoff,
    mutual_information_pca_dim::Int = 32,
    mutual_information_explained_variance::Real = 0.99,
    mutual_information_eigenvalue_rtol::Real = 1e-6,
    mutual_information_max_per_class::Union{Nothing,Int} = nothing,
    magnification::Real = 1.0,
    verbose::Bool = false,
)::CairoMakie.Figure
    if !(size > 0)
        throw(DomainError(size, "size must be > 0"))
    end
    if !(magnification > 0)
        throw(DomainError(magnification, "magnification must be > 0"))
    end

    comp_name = isnothing(comp_kind) ? comparison_kind_for_observable_plot(kind) : comp_kind
    obs_paths = data_paths(["$(kind)_$(size)_10000/statistics.jld2"])
    comp_paths = data_paths(["$(comp_name)_$(size)_10000/statistics.jld2"])

    data_loaded = load_observable_plot_matrix_observables(obs_paths, scalar)
    comp_loaded = load_observable_plot_matrix_observables(comp_paths, scalar)

    normalized_cardinalities_data = normalize_hists(data_loaded.cardinalities_hists; normalize_x_axis_with_size = true, cset_size = size)
    normalized_cardinalities_comp = normalize_hists(comp_loaded.cardinalities_hists; normalize_x_axis_with_size = true, cset_size = size)
    normalized_connectivity_data = normalize_hists(data_loaded.connectivity_link_hists)
    normalized_connectivity_comp = normalize_hists(comp_loaded.connectivity_link_hists)
    normalized_max_pathlen_data = normalize_hists(data_loaded.max_pathlen_hists; normalization = 1)
    normalized_max_pathlen_comp = normalize_hists(comp_loaded.max_pathlen_hists; normalization = 1)

    maybe_sqrt_pairs(pairs) = sqrt_scalars ? [(v, sqrt(Float64(s))) for (v, s) in pairs] : pairs
    normalized_cardinalities_data = [maybe_sqrt_pairs(g) for g in normalized_cardinalities_data]
    normalized_connectivity_data = [maybe_sqrt_pairs(g) for g in normalized_connectivity_data]
    normalized_max_pathlen_data = [maybe_sqrt_pairs(g) for g in normalized_max_pathlen_data]
    ev_sym_link_data = [maybe_sqrt_pairs(g) for g in data_loaded.ev_sym_link]

    observable_groups = (
        cardinalities = normalized_cardinalities_data,
        ev_sym_link = ev_sym_link_data,
        connectivity_link = normalized_connectivity_data,
        max_pathlen = normalized_max_pathlen_data,
    )
    observable_refs = (
        cardinalities = flatten_loaded_values(normalized_cardinalities_comp),
        ev_sym_link = flatten_loaded_values(comp_loaded.ev_sym_link),
        connectivity_link = flatten_loaded_values(normalized_connectivity_comp),
        max_pathlen = flatten_loaded_values(normalized_max_pathlen_comp),
    )

    energy_series = (
        cardinalities = scalar_bin_distinguishability(observable_groups.cardinalities, observable_refs.cardinalities; num_bins = num_bins, distance = energy_distance, verbose = verbose),
        ev_sym_link = scalar_bin_distinguishability(observable_groups.ev_sym_link, observable_refs.ev_sym_link; num_bins = num_bins, distance = energy_distance, verbose = verbose),
        connectivity_link = scalar_bin_distinguishability(observable_groups.connectivity_link, observable_refs.connectivity_link; num_bins = num_bins, distance = energy_distance, verbose = verbose),
        max_pathlen = scalar_bin_distinguishability(observable_groups.max_pathlen, observable_refs.max_pathlen; num_bins = num_bins, distance = energy_distance, verbose = verbose),
    )
    mi_series = (
        cardinalities = compute_mutual_information_scalar_series(observable_groups.cardinalities, observable_refs.cardinalities;
            num_bins = num_bins, k = mutual_information_k, pca_mode = mutual_information_pca_mode,
            pca_dim = mutual_information_pca_dim, explained_variance = mutual_information_explained_variance,
            eigenvalue_rtol = mutual_information_eigenvalue_rtol, max_per_class = mutual_information_max_per_class,
            rng = rng, verbose = verbose),
        ev_sym_link = compute_mutual_information_scalar_series(observable_groups.ev_sym_link, observable_refs.ev_sym_link;
            num_bins = num_bins, k = mutual_information_k, pca_mode = mutual_information_pca_mode,
            pca_dim = mutual_information_pca_dim, explained_variance = mutual_information_explained_variance,
            eigenvalue_rtol = mutual_information_eigenvalue_rtol, max_per_class = mutual_information_max_per_class,
            rng = rng, verbose = verbose),
        connectivity_link = compute_mutual_information_scalar_series(observable_groups.connectivity_link, observable_refs.connectivity_link;
            num_bins = num_bins, k = mutual_information_k, pca_mode = mutual_information_pca_mode,
            pca_dim = mutual_information_pca_dim, explained_variance = mutual_information_explained_variance,
            eigenvalue_rtol = mutual_information_eigenvalue_rtol, max_per_class = mutual_information_max_per_class,
            rng = rng, verbose = verbose),
        max_pathlen = compute_mutual_information_scalar_series(observable_groups.max_pathlen, observable_refs.max_pathlen;
            num_bins = num_bins, k = mutual_information_k, pca_mode = mutual_information_pca_mode,
            pca_dim = mutual_information_pca_dim, explained_variance = mutual_information_explained_variance,
            eigenvalue_rtol = mutual_information_eigenvalue_rtol, max_per_class = mutual_information_max_per_class,
            rng = rng, verbose = verbose),
    )
    mahalanobis_series = (
        cardinalities = compute_mahalanobis_scalar_series(observable_groups.cardinalities, observable_refs.cardinalities;
            num_bins = num_bins, regulator = regulator, R = R, q = q, alpha = alpha, rng = rng,
            projection_tolerance = projection_tolerance, to_regularize_rel = to_regularize_rel,
            progress = progress, verbose = verbose),
        ev_sym_link = compute_mahalanobis_scalar_series(observable_groups.ev_sym_link, observable_refs.ev_sym_link;
            num_bins = num_bins, regulator = regulator, R = R, q = q, alpha = alpha, rng = rng,
            projection_tolerance = projection_tolerance, to_regularize_rel = to_regularize_rel,
            progress = progress, verbose = verbose),
        connectivity_link = compute_mahalanobis_scalar_series(observable_groups.connectivity_link, observable_refs.connectivity_link;
            num_bins = num_bins, regulator = regulator, R = R, q = q, alpha = alpha, rng = rng,
            projection_tolerance = projection_tolerance, to_regularize_rel = to_regularize_rel,
            progress = progress, verbose = verbose),
        max_pathlen = compute_mahalanobis_scalar_series(observable_groups.max_pathlen, observable_refs.max_pathlen;
            num_bins = num_bins, regulator = regulator, R = R, q = q, alpha = alpha, rng = rng,
            projection_tolerance = projection_tolerance, to_regularize_rel = to_regularize_rel,
            progress = progress, verbose = verbose),
    )

    base_size = apply_paper_theme!(double_column = true, magnification = magnification)
    fig = CairoMakie.Figure(size = (1.35 * base_size[1], base_size[2]))
    axs = [
        CairoMakie.Axis(fig[1, 1]),
        CairoMakie.Axis(fig[1, 2]),
        CairoMakie.Axis(fig[1, 3]),
    ]

    metric_specs = [
        (axs[1], energy_series, LaTeXStrings.L"D_{\mathrm{energy}}", ylim_energy, yticks_energy),
        (axs[2], mi_series, LaTeXStrings.L"D_{\mathrm{MI}}", ylim_mutual_information, yticks_mutual_information),
        (axs[3], mahalanobis_series, LaTeXStrings.L"D_{\mathrm{Mah}}", ylim_mahalanobis, yticks_mahalanobis),
    ]
    observable_labels = observable_distinguishability_symbol_labels()
    observable_order = [:cardinalities, :ev_sym_link, :connectivity_link, :max_pathlen]

    legend_handles = Any[]
    legend_labels = Any[]
    for (panel_idx, (ax, series_group, ylabel_i, ylim_i, yticks_i)) in enumerate(metric_specs)
        ax.xlabel = string(scalar)
        ax.ylabel = ylabel_i
        xlim !== nothing && CairoMakie.xlims!(ax, xlim...)
        ylim_i !== nothing && CairoMakie.ylims!(ax, ylim_i...)
        xticks !== nothing && (ax.xticks = xticks)
        yticks_i !== nothing && (ax.yticks = yticks_i)
        for observable in observable_order
            series = getproperty(series_group, observable)
            xs = Float64[row.scalar for row in series]
            ys = Float64[row.D for row in series]
            plt = CairoMakie.lines!(ax, xs, ys; label = panel_idx == 1 ? getproperty(observable_labels, observable) : nothing)
            if panel_idx == 1
                push!(legend_handles, plt)
                push!(legend_labels, getproperty(observable_labels, observable))
            end
        end
    end

    CairoMakie.colgap!(fig.layout, 16)
    legend = CairoMakie.Legend(fig[1, 4], legend_handles, legend_labels)
    legend.tellheight = false

    CairoMakie.save(fig_path, fig)
    return fig
end
