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

# Keywords
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

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
    mink = CS.MinkowskiManifold{2}()
    quad_grid_unsorted = QG.generate_grid_2d_in_box(
        size,
        lattice,
        box; 
        rotate_deg = rotation_angle, 
        b = segment_ratio, 
        gamma_deg = segment_angle,
        shell_thickness = shell_thickness)

    quad_grid = QG.sort_grid_by_time_from_manifold(mink, quad_grid_unsorted)

    figsize = apply_paper_theme!(; magnification = magnification)
    fig = CairoMakie.Figure(size = figsize)
    ax = CairoMakie.Axis(fig[1,1])
    ax.xlabel="x"
    ax.ylabel="t"
    CairoMakie.scatter!(ax,quad_grid; markersize = magnification * markersize)

    if !isnothing(fig_path)
        CairoMakie.save(fig_path, fig)
    end
    return fig, ax
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

# Keywords
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

# Keyword Arguments
- `P_max`: Keyword option `P_max` controlling this method's behavior.
- `rng`: Random number generator used for stochastic steps.
- `segment_ratio`: Keyword option `segment_ratio` controlling this method's behavior.
- `segment_angle`: Keyword option `segment_angle` controlling this method's behavior.
- `rotation_angle`: Keyword option `rotation_angle` controlling this method's behavior.
- `fig_path`: Path or collection of paths used for loading/saving data.
- `magnification`: Keyword option `magnification` controlling this method's behavior.
- `linewidth`: Keyword option `linewidth` controlling this method's behavior.
- `ylim`: Axis limits for plotting.
- `xtick_fracs`: Keyword option `xtick_fracs` controlling this method's behavior.
- `max_peak_order`: Keyword option `max_peak_order` controlling this method's behavior.

# Throws
- `ArgumentError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
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
    
    grid_cset, _, _, _ = QG.create_grid_causet_in_boundary_2D_polynomial_manifold(size, lattice, CS.BoxBoundary{2}(((-1.,-1.),(1.,1.))), rng, 1, 2.; a = 1., b = segment_ratio, gamma_deg=segment_angle, rotate_deg=rotation_angle)
    grid_cset_abundances = CS.cardinality_abundances(grid_cset)
    idx = findfirst(iszero, comp_hist)-1
    @show idx
    r_comp_grid_man = grid_cset_abundances[2:idx] ./ comp_hist[2:idx] .- 1
    r_dev_fou_comp_grid_man = abs.(FFTW.fft(r_comp_grid_man))
    r_dev_freqs_comp_grid_man = (0:length(r_dev_fou_comp_grid_man)-1) ./ length(r_dev_fou_comp_grid_man)  # cycles per sample    

    f_min = 1 / P_max

    freqs = r_dev_freqs_comp_grid_man
    half = 1:fld(length(freqs), 2) 
    keep = [i for i in half if freqs[i] >= f_min]
    peak_idx = keep[argmax(r_dev_fou_comp_grid_man[keep])]
    f_peak = r_dev_freqs_comp_grid_man[peak_idx]
    P_est = 1 / f_peak           # period in “bins”

    @show f_peak P_est

    min_freq_for_peaks = 1 / 13
    keep_for_peaks = [i for i in keep if r_dev_freqs_comp_grid_man[i] >= min_freq_for_peaks]
    if !isempty(keep_for_peaks)
        idxs = sortperm(r_dev_fou_comp_grid_man[keep_for_peaks]; rev=true)
        printed_periods = Float64[]
        for i in idxs
            f = r_dev_freqs_comp_grid_man[keep_for_peaks[i]]
            P = 1 / f
            # Skip peaks that are effectively the same period as an earlier (stronger) peak.
            if any(abs(P - P0) <= 0.02 for P0 in printed_periods)
                continue
            end
            A = 2 * abs(r_dev_fou_comp_grid_man[keep_for_peaks[i]]) / length(r_comp_grid_man)
            println("f = ", f, "  P ≈ ", P, "  A = ", A)
            push!(printed_periods, P)
            if length(printed_periods) >= max_peak_order
                break
            end
        end
    end

    figsize = apply_paper_theme!(; magnification = magnification)
    fig = CairoMakie.Figure(size = figsize)
    ax = CairoMakie.Axis(fig[1,1])

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
            CairoMakie.vlines!(ax, xticks; color=(:black,1.), linestyle=:dash, linewidth = magnification * linewidth)
        end
    end

    CairoMakie.lines!(ax, r_dev_freqs_comp_grid_man[keep], r_dev_fou_comp_grid_man[keep]; linewidth = magnification * linewidth)
    ax.xlabel = "frequency (cycles per bin)"
    ax.ylabel = LaTeXStrings.L"\mathcal{F}(\mathcal{S}_n^{\mathrm{grid}} / \mathcal{S}_n^{\mathrm{man}} -1)"

    CairoMakie.xlims!(ax, (0.,0.51))
    if !isnothing(ylim)
        CairoMakie.ylims!(ax, ylim)
    end

    ax.xminorticksvisible = false
    ax.xminorgridvisible = false

    if !isnothing(fig_path)
       CairoMakie.save(fig_path, fig)
    end

    fig
end
