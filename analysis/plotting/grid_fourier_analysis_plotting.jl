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
    rotation_angle::Float64;
    box::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}=((-1.,-1.),(1.,1.)),
    segment_ratio::Float64=2.,
    segment_angle::Float64=60.,
    shell_thickness::Union{Nothing,Float64} = nothing,
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
    ax.xlabel = "frequency (cycles per bin)"
    ax.ylabel = LaTeXStrings.L"\mathcal{F}(\mathcal{S}_n^{\mathrm{grid}} / \mathcal{S}_n^{\mathrm{man}} -1)"
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
