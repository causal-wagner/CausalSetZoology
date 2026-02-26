"""
    hist_hist_vec_hist_plot_matrix(
        data::Tuple{
            Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
            Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
            Vector{Vector{Tuple{Vector{Float64}, Float64}}},
            Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}}
        };
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
- `data`: Input dataset(s) consumed by this method.

# Keyword Arguments
- `xlim`: Axis limits for plotting.
- `ylim`: Axis limits for plotting.
- `xlabel`: Text label shown in the plot output.
- `ylabel`: Text label shown in the plot output.
- `num_bins`: Bin selection or binning control parameter.
- `colormap`: Keyword option `colormap` controlling this method's behavior.
- `invert_color_scaling`: Boolean flag controlling output formatting or algorithm behavior.
- `colorbar_label`: Text label shown in the plot output.
- `colorbar_ticks`: Keyword option `colorbar_ticks` controlling this method's behavior.
- `colorbar_pos`: Keyword option `colorbar_pos` controlling this method's behavior.
- `colorbar_size`: Keyword option `colorbar_size` controlling this method's behavior.
- `logscale_x`: Toggle for logarithmic axis scaling.
- `logscale_y`: Toggle for logarithmic axis scaling.
- `double_column`: Boolean toggle controlling output or execution behavior.
- `magnification`: Keyword option `magnification` controlling this method's behavior.
- `plot_std`: Boolean toggle controlling output or execution behavior.
- `return_axis`: Boolean toggle controlling output or execution behavior.

# Returns
- `result`: Output of `hist_hist_vec_hist_plot_matrix` as described in the summary above.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function hist_hist_vec_hist_plot_matrix(
    data::Tuple{
        Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
        Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
        Vector{Vector{Tuple{Vector{Float64}, Float64}}},
        Vector{Vector{Tuple{Dict{Int64, Float64}, Real}}},
    },
    fig_name::String;
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
    @assert length(xlim) == 4
    @assert length(ylim) == 4
    @assert length(xlabel) == 4
    @assert length(ylabel) == 4

    h1, h2, v3, h4 = data

    avg1 = [average_histogram_with_std(h1[i]) for i in 1:length(h1)]
    avg2 = [average_histogram_with_std(h2[i]) for i in 1:length(h2)]
    avg3 = [average_vectors_with_std(v3[i]; num_bins = num_bins) for i in 1:length(v3)]
    avg4 = [average_histogram_with_std(h4[i]) for i in 1:length(h4)]

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
            ax.xaxisposition = :top
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
            linkxaxes!(ax, ax_left)
            linkyaxes!(ax, ax_left)
        end
    end

    function plot_on_axis!(
        ax,
        data::AbstractVector{<:Tuple},
        comp_mean_std,
        xlim_i,
        ylim_i,
        xlabel_i,
        ylabel_i,
        xticks_i,
        yticks_i,
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

        # comparison band first
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
            @assert length(mean) == length(std)
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

        # comparison mean line on top
        if comp_x !== nothing
            isnothing(comp_linewidth) ? CairoMakie.lines!(ax, comp_x, comp_mean; color = comp_color) :
                CairoMakie.lines!(ax, comp_x, comp_mean; color = comp_color, linewidth = comp_linewidth)
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

        comp1 = !isempty(comp1_flat) && comp1_flat[1] isa Tuple ?
            average_histogram_with_std(comp1_flat; num_bins = num_bins) :
            average_histogram_with_std(comp1_flat)

        comp2 = !isempty(comp2_flat) && comp2_flat[1] isa Tuple ?
            average_histogram_with_std(comp2_flat; num_bins = num_bins) :
            average_histogram_with_std(comp2_flat)

        comp3 = !isempty(comp3_flat) && comp3_flat[1] isa Tuple ?
            average_vectors_with_std(comp3_flat; num_bins = num_bins) :
            average_vectors_with_std(comp3_flat)

        comp4 = !isempty(comp4_flat) && comp4_flat[1] isa Tuple ?
            average_histogram_with_std(comp4_flat; num_bins = num_bins) :
            average_histogram_with_std(comp4_flat)
    end

    xt = xticks === nothing ? fill(nothing, 4) : xticks
    yt = yticks === nothing ? fill(nothing, 4) : yticks
    @assert length(xt) == 4
    @assert length(yt) == 4

    plot_on_axis!(axs[1], d1, comp1, xlim[1], ylim[1], xlabel[1], ylabel[1], xt[1], yt[1])
    plot_on_axis!(axs[2], d2, comp2, xlim[2], ylim[2], xlabel[2], ylabel[2], xt[2], yt[2])
    plot_on_axis!(axs[3], d3, comp3, xlim[3], ylim[3], xlabel[3], ylabel[3], xt[3], yt[3])
    plot_on_axis!(axs[4], d4, comp4, xlim[4], ylim[4], xlabel[4], ylabel[4], xt[4], yt[4])

    cb_cmap = invert_color_scaling ? CairoMakie.Reverse(colormap) : colormap
    cb = if colorbar_pos === nothing
        if colorbar_side == :right
            Colorbar(fig[1:2, 3], limits = (vmin, vmax), colormap = cb_cmap)
        elseif colorbar_side == :left
            Colorbar(fig[1:2, 0], limits = (vmin, vmax), colormap = cb_cmap)
        elseif colorbar_side == :top
            Colorbar(fig[0, 1:2], limits = (vmin, vmax), colormap = cb_cmap, vertical = false)
        elseif colorbar_side == :bottom
            Colorbar(fig[3, 1:2], limits = (vmin, vmax), colormap = cb_cmap, vertical = false)
        else
            error("colorbar_side must be :left, :right, :top, or :bottom")
        end
    else
        cb = Colorbar(fig[1:2, 3], limits = (vmin, vmax), colormap = cb_cmap)
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
        @assert colorbar_label_pos in (:side, :top) "colorbar_label_pos must be :side or :top"
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

    CairoMakie.save(fig_path(fig_name), fig)
    return return_axis ? (fig, axs) : fig
end

"""
    hist_hist_vec_distinguishability_plot_matrix(
        data_paths,
        comp_paths,
        scalar::Symbol,
        fig_name::String;
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
        num_workers = 1,
        verbose = false,
        rank_tol = 1e-12,
        stabilization_method = :regularization,
        projection_tolerance = 1e-10,
        progress = false,
        progress_step = nothing,
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
- `scalar`: Scalar value(s) or scalar field identifier.
- `fig_name`: Output figure name/path used when saving plots.

# Keyword Arguments
- `xlim`: Axis limits for plotting.
- `ylim`: Axis limits for plotting.
- `xlabel`: Text label shown in the plot output.
- `ylabel`: Text label shown in the plot output.
- `sqrt_scalars`: Scalar value(s) or scalar field identifier.
- `regulator`: Keyword option `regulator` controlling this method's behavior.
- `R`: Numeric control parameter for fitting/sampling resolution.
- `q`: Numeric control parameter for fitting/sampling resolution.
- `alpha`: Numeric control parameter for fitting/sampling resolution.
- `rng`: Random number generator used for stochastic steps.
- `symmetric`: Keyword option `symmetric` controlling this method's behavior.
- `num_workers`: Keyword option `num_workers` controlling this method's behavior.
- `verbose`: Boolean toggle controlling output or execution behavior.
- `rank_tol`: Keyword option `rank_tol` controlling this method's behavior.
- `stabilization_method`: Keyword option `stabilization_method` controlling this method's behavior.
- `projection_tolerance`: Keyword option `projection_tolerance` controlling this method's behavior.
- `progress`: Boolean toggle controlling output or execution behavior.
- `progress_step`: Keyword option `progress_step` controlling this method's behavior.
- `invert_color_scaling`: Boolean flag controlling output formatting or algorithm behavior.
- `colorbar_label`: Text label shown in the plot output.
- `colorbar_ticks`: Keyword option `colorbar_ticks` controlling this method's behavior.
- `colorbar_pos`: Keyword option `colorbar_pos` controlling this method's behavior.
- `colorbar_size`: Keyword option `colorbar_size` controlling this method's behavior.
- `colorbar_side`: Keyword option `colorbar_side` controlling this method's behavior.
- `colorbar_label_pos`: Text label shown in the plot output.
- `comp_color`: Keyword option `comp_color` controlling this method's behavior.
- `comp_linewidth`: Keyword option `comp_linewidth` controlling this method's behavior.
- `xticks`: Keyword option `xticks` controlling this method's behavior.
- `yticks`: Keyword option `yticks` controlling this method's behavior.
- `logscale_x`: Toggle for logarithmic axis scaling.
- `logscale_y`: Toggle for logarithmic axis scaling.
- `double_column`: Boolean toggle controlling output or execution behavior.
- `magnification`: Keyword option `magnification` controlling this method's behavior.
- `plot_std`: Boolean toggle controlling output or execution behavior.
- `right_yaxis`: Boolean toggle controlling output or execution behavior.
- `top_xaxis`: Boolean toggle controlling output or execution behavior.
- `rowgap`: Keyword option `rowgap` controlling this method's behavior.
- `colgap`: Keyword option `colgap` controlling this method's behavior.

# Returns
- `result::CairoMakie.Figure`: Output of `hist_hist_vec_distinguishability_plot_matrix` with type annotation `CairoMakie.Figure`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function hist_hist_vec_distinguishability_plot_matrix(
    data_paths_in,
    comp_paths_in,
    scalar::Symbol,
    fig_name::String;
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
    num_workers::Int = 1,
    verbose::Bool = false,
    rank_tol::Float64 = 1e-12,
    stabilization_method::Symbol = :regularization,
    projection_tolerance::Float64 = 1e-10,
    progress::Bool = false,
    progress_step::Union{Nothing,Int} = nothing,
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
    @assert length(xlim) == 4
    @assert length(ylim) == 4
    @assert length(xlabel) == 4
    @assert length(ylabel) == 4
    @assert !symmetric "hist_hist_vec_distinguishability_plot_matrix enforces symmetric = false"

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
        num_workers = num_workers,
        verbose = verbose,
        rank_tol = rank_tol,
        stabilization_method = stabilization_method,
        projection_tolerance = projection_tolerance,
        progress = progress,
        progress_step = progress_step,
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
        num_workers = num_workers,
        verbose = verbose,
        rank_tol = rank_tol,
        stabilization_method = stabilization_method,
        projection_tolerance = projection_tolerance,
        progress = progress,
        progress_step = progress_step,
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
        num_workers = num_workers,
        verbose = verbose,
        rank_tol = rank_tol,
        stabilization_method = stabilization_method,
        projection_tolerance = projection_tolerance,
        progress = progress,
        progress_step = progress_step,
    )

    # prepare panel data (first 3 panels like hist_hist_vec_hist_plot_matrix)
    avg_hist_or_vec(data_group) = begin
        if isempty(data_group)
            return Vector{Tuple{Real,Vector{Float64},Vector{Float64}}}()
        end
        if data_group[1][1] isa AbstractDict
            return average_histogram_with_std(data_group)
        else
            return average_vectors_with_std(data_group; num_bins = num_bins)
        end
    end
    comp_avg_hist_or_vec(data_group) = begin
        if isempty(data_group)
            return nothing
        end
        if data_group[1] isa Tuple
            if data_group[1][1] isa AbstractDict
                return average_histogram_with_std(data_group; num_bins = num_bins)
            else
                return average_vectors_with_std(data_group; num_bins = num_bins)
            end
        end
        return nothing
    end

    d1 = avg_hist_or_vec(connectivity_data_single[1])
    d2 = avg_hist_or_vec(max_pathlen_data_single[1])
    d3 = avg_hist_or_vec(ev_sym_link_data_single[1])
    comp1 = comp_avg_hist_or_vec(connectivity_ref)
    comp2 = comp_avg_hist_or_vec(max_pathlen_ref)
    comp3 = comp_avg_hist_or_vec(ev_sym_link_ref)

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
            ax.xaxisposition = :top
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
            linkxaxes!(ax, ax_left)
            linkyaxes!(ax, ax_left)
        end
    end

    xt = xticks === nothing ? fill(nothing, 4) : xticks
    yt = yticks === nothing ? fill(nothing, 4) : yticks
    @assert length(xt) == 4
    @assert length(yt) == 4

    function plot_hist_or_vec_panel!(
        ax,
        data::AbstractVector{<:Tuple},
        comp_mean_std,
        xlim_i,
        ylim_i,
        xlabel_i,
        ylabel_i,
        xticks_i,
        yticks_i,
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
    end

    plot_hist_or_vec_panel!(axs[1], d1, comp1, xlim[1], ylim[1], xlabel[1], ylabel[1], xt[1], yt[1])
    plot_hist_or_vec_panel!(axs[2], d2, comp2, xlim[2], ylim[2], xlabel[2], ylabel[2], xt[2], yt[2])
    plot_hist_or_vec_panel!(axs[3], d3, comp3, xlim[3], ylim[3], xlabel[3], ylabel[3], xt[3], yt[3])

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
            Colorbar(fig[1:2, 3], limits = (vmin, vmax), colormap = cb_cmap)
        elseif colorbar_side == :left
            Colorbar(fig[1:2, 0], limits = (vmin, vmax), colormap = cb_cmap)
        elseif colorbar_side == :top
            Colorbar(fig[0, 1:2], limits = (vmin, vmax), colormap = cb_cmap, vertical = false)
        elseif colorbar_side == :bottom
            Colorbar(fig[3, 1:2], limits = (vmin, vmax), colormap = cb_cmap, vertical = false)
        else
            error("colorbar_side must be :left, :right, :top, or :bottom")
        end
    else
        cb = Colorbar(fig[1:2, 3], limits = (vmin, vmax), colormap = cb_cmap)
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
        @assert colorbar_label_pos in (:side, :top) "colorbar_label_pos must be :side or :top"
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

    CairoMakie.save(fig_path(fig_name), fig)
    return fig
end
