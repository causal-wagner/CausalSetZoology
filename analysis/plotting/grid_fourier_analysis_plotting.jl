using CairoMakie
using FFTW
using LaTeXStrings
using Printf
using Random

"""
    plot_grid_points(quad_grid; markersize, magnification, fig_path)

Plot precomputed grid points and optionally save the figure.

# Arguments
- `quad_grid`: Grid point collection to scatter-plot.

# Keyword Arguments
- `markersize`: Scatter marker size (scaled by `magnification`).
- `magnification`: Plot-theme magnification factor.
- `fig_path`: Optional output path to save the figure.

# Returns
- `(fig, ax)`: The created `CairoMakie.Figure` and axis.
"""
function plot_grid_points(
    quad_grid;
    markersize::Real=4,
    magnification::Float64=1.0,
    fig_path::Union{Nothing,String}=nothing,
)
    figsize = apply_paper_theme!(; magnification = magnification)
    fig = CairoMakie.Figure(size = figsize)
    ax = CairoMakie.Axis(fig[1, 1])
    ax.xlabel = "x"
    ax.ylabel = "t"
    CairoMakie.scatter!(ax, quad_grid; markersize = magnification * markersize)
    if !isnothing(fig_path)
        CairoMakie.save(fig_path, fig)
    end
    return fig, ax
end

function plot_grid_points!(
    ax,
    quad_grid;
    markersize::Real = 4,
    magnification::Float64 = 1.0,
    remove_coordinate_ticks::Bool = false,
)
    ax.xlabel = "x"
    ax.ylabel = "t"
    if remove_coordinate_ticks
        ax.xticksvisible = false
        ax.yticksvisible = false
        ax.xticklabelsvisible = false
        ax.yticklabelsvisible = false
    end
    CairoMakie.scatter!(ax, quad_grid; markersize = magnification * markersize)
    return ax
end

"""
    create_grid_and_plot(
        size::Int,
        lattice::String,
        rotation_angle::Float64;
        box::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}} = ((-1.0, -1.0), (1.0, 1.0)),
        segment_ratio::Float64 = 2.0,
        segment_angle::Float64 = 60.0,
        shell_thickness::Union{Nothing,Float64} = nothing,
        markersize::Real = 4,
        magnification::Float64 = 1.0,
        fig_path::Union{Nothing,String} = nothing,
    )

Generate a 2D lattice/grid in a box, sort points by time coordinate, and plot it.

# Arguments
- `size`: Number of points requested for grid generation.
- `lattice`: Lattice identifier understood by `QG.generate_grid_2d_in_box`.
- `rotation_angle`: Rotation angle in degrees applied during grid generation.

# Keyword Arguments
- `box`: Spatial-temporal box bounds `((x_min, t_min), (x_max, t_max))`.
- `segment_ratio`, `segment_angle`, `shell_thickness`: Parameters passed through
  to grid generation.
- `markersize`: Scatter marker size (scaled by `magnification`).
- `magnification`: Plot theme magnification factor.
- `fig_path`: Optional output path to save the figure.

# Returns
- `(fig, ax)`: The created Makie `CairoMakie.Figure` and its `CairoMakie.Axis`.

# Keyword Arguments
- `box`: Keyword option `box` controlling this method's behavior.
- `segment_ratio`: Keyword option `segment_ratio` controlling this method's behavior.
- `segment_angle`: Keyword option `segment_angle` controlling this method's behavior.
- `shell_thickness`: Keyword option `shell_thickness` controlling this method's behavior.
- `markersize`: Keyword option `markersize` controlling this method's behavior.
- `magnification`: Keyword option `magnification` controlling this method's behavior.
- `fig_path`: Path or collection of paths used for loading/saving data.

"""
function create_grid_and_plot(
    size::Int,
    lattice::String,
    rotation_angle::Union{Real,Nothing};
    box::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}=((-1.,-1.),(1.,1.)),
    segment_ratio::Float64=2.,
    segment_angle::Float64=60.,
    shell_thickness::Union{Nothing,Float64} = nothing,
    quasicrystal_rho::Real = 1.0,
    quasicrystal_crystal::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}} = nothing,
    quasicrystal_center::Tuple{Real,Real} = (0.5, 0.0),
    quasicrystal_exact_size::Bool = true,
    quasicrystal_deviation_from_mean_size::Float64 = 0.1,
    quasicrystal_max_iter::Int = 100,
    markersize::Real = 4,
    magnification::Float64 = 1.,
    fig_path::Union{Nothing,String}=nothing,
    )
    quad_grid = generate_sorted_grid(
        size,
        lattice,
        rotation_angle;
        box = box,
        segment_ratio = segment_ratio,
        segment_angle = segment_angle,
        shell_thickness = shell_thickness,
        quasicrystal_rho = quasicrystal_rho,
        quasicrystal_crystal = quasicrystal_crystal,
        quasicrystal_center = quasicrystal_center,
        quasicrystal_exact_size = quasicrystal_exact_size,
        quasicrystal_deviation_from_mean_size = quasicrystal_deviation_from_mean_size,
        quasicrystal_max_iter = quasicrystal_max_iter,
    )
    return plot_grid_points(
        quad_grid;
        markersize = markersize,
        magnification = magnification,
        fig_path = fig_path,
    )
end

"""
    plot_fourier_grid_deviation(spec; fig_path, magnification, linewidth, ylim, xtick_fracs)

Plot Fourier-analysis data produced by `compute_fourier_grid_deviation`.

# Arguments
- `spec`: Named tuple produced by `compute_fourier_grid_deviation`.

# Keyword Arguments
- `fig_path`: Optional output path to save the figure.
- `magnification`: Plot-theme magnification factor.
- `linewidth`: Line width for plotted curves/guide lines.
- `ylim`: Optional y-axis limits.
- `xtick_fracs`: Optional custom x ticks/labels (supports rationals).

# Returns
- `fig`: The created `CairoMakie.Figure`.
"""
function plot_fourier_grid_deviation(
    spec;
    fig_path::Union{Nothing,String}=nothing,
    magnification::Real=1.,
    linewidth::Real=1,
    ylim::Union{Tuple{Float64,Float64},Nothing}=nothing,
    xtick_fracs::Union{Nothing,Vector{<:Any}}=nothing,
)
    figsize = apply_paper_theme!(; magnification = magnification)
    fig = CairoMakie.Figure(size = figsize)
    ax = CairoMakie.Axis(fig[1, 1])

    if xtick_fracs !== nothing
        xticks = collect(xtick_fracs)
        if !isempty(xticks)
            labels = map(xtick_fracs) do x
                if x isa Rational
                    n = numerator(x)
                    d = denominator(x)
                    if d == 1
                        string(n)
                    elseif n < 0
                        LaTeXStrings.LaTeXString("-\\frac{$(abs(n))}{$d}")
                    else
                        LaTeXStrings.LaTeXString("\\frac{$n}{$d}")
                    end
                else
                    Printf.@sprintf("%.2f", Float64(x))
                end
            end
            ax.xticks = (xticks, labels)
            CairoMakie.vlines!(ax, xticks; color = (:black, 1.), linestyle = :dash, linewidth = magnification * linewidth)
        end
    end

    CairoMakie.lines!(ax, spec.freqs[spec.keep], spec.spectrum[spec.keep]; linewidth = magnification * linewidth)
    ax.xlabel = L"\omega~\mathrm{(cycles~per~bin)}"
    ax.ylabel = LaTeXStrings.L"\mathcal{F}_\omega(\mathcal{S}_j^{\mathrm{lat}} / \langle\mathcal{S}^{\mathrm{man}}\rangle -1)"
    CairoMakie.xlims!(ax, (0., 0.51))
    if !isnothing(ylim)
        CairoMakie.ylims!(ax, ylim)
    end
    ax.xminorticksvisible = false
    ax.xminorgridvisible = false

    if !isnothing(fig_path)
        CairoMakie.save(fig_path, fig)
    end
    return fig
end

function plot_fourier_grid_deviation!(
    ax,
    spec;
    linewidth::Real = 1,
    magnification::Real = 1.0,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    xtick_fracs::Union{Nothing,Vector{<:Any}} = nothing,
)
    if xtick_fracs !== nothing
        xticks = collect(xtick_fracs)
        if !isempty(xticks)
            labels = map(xtick_fracs) do x
                if x isa Rational
                    n = numerator(x)
                    d = denominator(x)
                    if d == 1
                        string(n)
                    elseif n < 0
                        LaTeXStrings.LaTeXString("-\\frac{$(abs(n))}{$d}")
                    else
                        LaTeXStrings.LaTeXString("\\frac{$n}{$d}")
                    end
                else
                    Printf.@sprintf("%.2f", Float64(x))
                end
            end
            ax.xticks = (xticks, labels)
            CairoMakie.vlines!(ax, xticks; color = (:black, 1.), linestyle = :dash, linewidth = magnification * linewidth)
        end
    end

    CairoMakie.lines!(ax, spec.freqs[spec.keep], spec.spectrum[spec.keep]; linewidth = magnification * linewidth)
    ax.xlabel = L"\omega~\mathrm{(cycles~per~bin)}"
    ax.ylabel = LaTeXStrings.L"\mathcal{F}_\omega(\mathcal{S}_j^{\mathrm{lat}} / \langle\mathcal{S}^{\mathrm{man}}\rangle -1)"
    CairoMakie.xlims!(ax, (0., 0.51))
    if !isnothing(ylim)
        CairoMakie.ylims!(ax, ylim)
    end
    ax.xminorticksvisible = false
    ax.xminorgridvisible = false
    return ax
end

"""
    fourier_transform_grid_deviation(
        comp_hist::Vector{Float64},
        size::Int64,
        lattice::String;
        P_max::Float64 = 300.0,
        rng::Random.AbstractRNG = Random.GLOBAL_RNG,
        segment_ratio::Float64 = 1.0,
        segment_angle::Float64 = 60.0,
        rotation_angle::Union{Float64,Nothing} = nothing,
        fig_path::Union{Nothing,String} = nothing,
        magnification::Real = 1.0,
        linewidth::Real = 1,
        ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
        xtick_fracs::Union{Nothing,Vector{<:Any}} = nothing,
        max_peak_order::Int = 5,
    )

Compare grid vs reference abundances, compute the Fourier spectrum of their
relative deviation, print dominant periods, and plot the spectrum.

The plotted quantity is the magnitude of
`FFTW.fft(grid_abundance ./ comp_hist .- 1)` over positive frequencies above
`1 / P_max`.

# Arguments
- `comp_hist`: Reference abundance histogram (e.g. manifold baseline).
- `size`, `lattice`: Grid generation parameters.

# Keyword Arguments
- `P_max`: Maximum period considered for peak search (`f_min = 1 / P_max`).
- `rng`: Random number generator used for causal set generation.
- `segment_ratio`, `segment_angle`, `rotation_angle`: Grid-generation parameters.
- `fig_path`: Optional path to save the resulting figure.
- `magnification`, `linewidth`: Plot styling parameters.
- `ylim`: Optional y-axis limits.
- `xtick_fracs`: Optional custom x-tick positions/labels (supports rationals).
- `max_peak_order`: Maximum number of distinct dominant periods to print.

# Returns
- `CairoMakie.Figure`: The generated Fourier spectrum figure.

# Notes
- The function prints intermediate diagnostics (`idx`, `f_peak`, `P_est`) and
  peak summaries to stdout.
"""
function fourier_transform_grid_deviation(
    comp_hist::Vector{Float64}, 
    size::Int64, 
    lattice::String; 
    P_max::Float64=300., 
    rng::Random.AbstractRNG=Random.GLOBAL_RNG, 
    segment_ratio::Float64=1., 
    segment_angle::Float64=60., 
    rotation_angle::Union{Float64,Nothing}=nothing,
    quasicrystal_rho::Real = 1.0,
    quasicrystal_crystal::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}} = nothing,
    quasicrystal_center::Tuple{Real,Real} = (0.5, 0.0),
    quasicrystal_exact_size::Bool = true,
    quasicrystal_deviation_from_mean_size::Float64 = 0.1,
    quasicrystal_max_iter::Int = 100,
    fig_path::Union{Nothing,String}=nothing,
    magnification::Real=1.,
    linewidth::Real=1,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    xtick_fracs::Union{Nothing,Vector{<:Any}}=nothing,
    max_peak_order::Int = 5 
    )
    spec = compute_fourier_grid_deviation(
        comp_hist,
        size,
        lattice;
        P_max = P_max,
        rng = rng,
        segment_ratio = segment_ratio,
        segment_angle = segment_angle,
        rotation_angle = rotation_angle,
        quasicrystal_rho = quasicrystal_rho,
        quasicrystal_crystal = quasicrystal_crystal,
        quasicrystal_center = quasicrystal_center,
        quasicrystal_exact_size = quasicrystal_exact_size,
        quasicrystal_deviation_from_mean_size = quasicrystal_deviation_from_mean_size,
        quasicrystal_max_iter = quasicrystal_max_iter,
        max_peak_order = max_peak_order,
    )

    @show spec.idx
    @show spec.f_peak spec.P_est
    for row in spec.peak_rows
        println("f = ", row.f, "  P ≈ ", row.P, "  A = ", row.A)
    end

    return plot_fourier_grid_deviation(
        spec;
        fig_path = fig_path,
        magnification = magnification,
        linewidth = linewidth,
        ylim = ylim,
        xtick_fracs = xtick_fracs,
    )
end

normalize_four_panel_value(x, name::AbstractString) = fill(x, 4)

function normalize_four_panel_value(x::AbstractVector, name::AbstractString)
    vals = collect(x)
    if length(vals) != 4
        throw(ArgumentError("$(name) must have length 4"))
    end
    return vals
end

function normalize_four_panel_tick_lists(x)
    if x === nothing
        return fill(nothing, 4)
    end
    vals = collect(x)
    if length(vals) == 4 && all(v -> v === nothing || v isa AbstractVector, vals)
        return vals
    end
    return fill(vals, 4)
end

function grid_fourier_plot_matrix(
    fig_path::String;
    comp_hists::AbstractVector{<:AbstractVector{<:Real}},
    sizes::Union{Int,AbstractVector{<:Integer}},
    lattices::Union{AbstractString,AbstractVector{<:AbstractString}},
    rotation_angles::Union{Real,Nothing,AbstractVector},
    boxes::Union{Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}},AbstractVector} = ((-1.0, -1.0), (1.0, 1.0)),
    segment_ratios::Union{Real,AbstractVector{<:Real}} = 2.0,
    segment_angles::Union{Real,AbstractVector{<:Real}} = 60.0,
    shell_thicknesss::Union{Nothing,Real,AbstractVector} = nothing,
    quasicrystal_rhos::Union{Real,AbstractVector{<:Real}} = 1.0,
    quasicrystal_crystal::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}},AbstractVector} = nothing,
    quasicrystal_centers::Union{Tuple{Real,Real},AbstractVector} = (0.5, 0.0),
    quasicrystal_exact_sizes::Union{Bool,AbstractVector{Bool}} = true,
    quasicrystal_deviation_from_mean_sizes::Union{Real,AbstractVector{<:Real}} = 0.1,
    quasicrystal_max_iters::Union{Int,AbstractVector{<:Integer}} = 100,
    markersizes::Union{Real,AbstractVector{<:Real}} = 4,
    P_maxs::Union{Real,AbstractVector{<:Real}} = 300.0,
    rngs::Union{Random.AbstractRNG,AbstractVector} = Random.GLOBAL_RNG,
    fourier_segment_ratios::Union{Real,AbstractVector{<:Real}} = 1.0,
    fourier_segment_angles::Union{Real,AbstractVector{<:Real}} = 60.0,
    linewidths::Union{Real,AbstractVector{<:Real}} = 1,
    ylims_fourier::Union{Nothing,Tuple{Float64,Float64},AbstractVector} = nothing,
    xtick_fracss::Union{Nothing,Vector{<:Any},AbstractVector} = nothing,
    max_peak_orders::Union{Int,AbstractVector{<:Integer}} = 5,
    fourier_yaxis_side::Symbol = :right,
    remove_coordinate_ticks::Bool = false,
    magnification::Real = 1.0,
    double_column::Bool = true,
    rowgap::Real = 8.0,
    colgap::Real = 12.0,
    return_axis::Bool = false,
)
    if fourier_yaxis_side ∉ (:left, :right)
        throw(ArgumentError("fourier_yaxis_side must be :left or :right"))
    end
    comp_hists_n = normalize_four_panel_value(comp_hists, "comp_hists")
    sizes_n = Int.(normalize_four_panel_value(sizes, "sizes"))
    lattices_n = String.(normalize_four_panel_value(lattices, "lattices"))
    rotation_angles_n = normalize_four_panel_value(rotation_angles, "rotation_angles")
    boxes_n = normalize_four_panel_value(boxes, "boxes")
    segment_ratios_n = Float64.(normalize_four_panel_value(segment_ratios, "segment_ratios"))
    segment_angles_n = Float64.(normalize_four_panel_value(segment_angles, "segment_angles"))
    shell_thicknesss_n = normalize_four_panel_value(shell_thicknesss, "shell_thicknesss")
    quasicrystal_rhos_n = Float64.(normalize_four_panel_value(quasicrystal_rhos, "quasicrystal_rhos"))
    quasicrystal_crystals_n = normalize_four_panel_value(quasicrystal_crystal, "quasicrystal_crystal")
    quasicrystal_centers_n = normalize_four_panel_value(quasicrystal_centers, "quasicrystal_centers")
    quasicrystal_exact_sizes_n = Bool.(normalize_four_panel_value(quasicrystal_exact_sizes, "quasicrystal_exact_sizes"))
    quasicrystal_deviation_from_mean_sizes_n = Float64.(normalize_four_panel_value(quasicrystal_deviation_from_mean_sizes, "quasicrystal_deviation_from_mean_sizes"))
    quasicrystal_max_iters_n = Int.(normalize_four_panel_value(quasicrystal_max_iters, "quasicrystal_max_iters"))
    markersizes_n = normalize_four_panel_value(markersizes, "markersizes")
    P_maxs_n = Float64.(normalize_four_panel_value(P_maxs, "P_maxs"))
    rngs_n = normalize_four_panel_value(rngs, "rngs")
    fourier_segment_ratios_n = Float64.(normalize_four_panel_value(fourier_segment_ratios, "fourier_segment_ratios"))
    fourier_segment_angles_n = Float64.(normalize_four_panel_value(fourier_segment_angles, "fourier_segment_angles"))
    linewidths_n = normalize_four_panel_value(linewidths, "linewidths")
    ylims_fourier_n = normalize_four_panel_value(ylims_fourier, "ylims_fourier")
    xtick_fracss_n = normalize_four_panel_tick_lists(xtick_fracss)
    max_peak_orders_n = Int.(normalize_four_panel_value(max_peak_orders, "max_peak_orders"))

    base_size = apply_paper_theme!(double_column = double_column, magnification = magnification)
    fig = CairoMakie.Figure(size = (base_size[1], 4 * base_size[2]))
    CairoMakie.rowgap!(fig.layout, rowgap)
    CairoMakie.colgap!(fig.layout, colgap)

    axs = Matrix{CairoMakie.Axis}(undef, 4, 2)

    for row in 1:4
        left_ax = CairoMakie.Axis(fig[row, 1])
        right_ax = CairoMakie.Axis(fig[row, 2])
        if fourier_yaxis_side == :right
            right_ax.yaxisposition = :right
            right_ax.flip_ylabel = true
        else
            right_ax.yaxisposition = :left
            right_ax.flip_ylabel = false
        end
        axs[row, 1] = left_ax
        axs[row, 2] = right_ax

        quad_grid = CausalSetZoology.generate_sorted_grid(
            sizes_n[row],
            lattices_n[row],
            Float64(rotation_angles_n[row]);
            box = boxes_n[row],
            segment_ratio = segment_ratios_n[row],
            segment_angle = segment_angles_n[row],
            shell_thickness = shell_thicknesss_n[row],
            quasicrystal_rho = quasicrystal_rhos_n[row],
            quasicrystal_crystal = quasicrystal_crystals_n[row],
            quasicrystal_center = quasicrystal_centers_n[row],
            quasicrystal_exact_size = quasicrystal_exact_sizes_n[row],
            quasicrystal_deviation_from_mean_size = quasicrystal_deviation_from_mean_sizes_n[row],
            quasicrystal_max_iter = quasicrystal_max_iters_n[row],
        )
        plot_grid_points!(
            left_ax,
            quad_grid;
            markersize = markersizes_n[row],
            magnification = magnification,
            remove_coordinate_ticks = remove_coordinate_ticks,
        )
        if row < 4
            left_ax.xlabel = ""
        end

        spec = CausalSetZoology.compute_fourier_grid_deviation(
            comp_hists_n[row],
            quad_grid;
            P_max = P_maxs_n[row],
            max_peak_order = max_peak_orders_n[row],
        )
        plot_fourier_grid_deviation!(
            right_ax,
            spec;
            linewidth = linewidths_n[row],
            magnification = magnification,
            ylim = ylims_fourier_n[row],
            xtick_fracs = xtick_fracss_n[row],
        )
        if row < 4
            right_ax.xlabel = ""
        end
    end

    CairoMakie.save(fig_path, fig)
    return return_axis ? (fig, axs) : fig
end
