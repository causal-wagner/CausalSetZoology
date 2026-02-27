"""
    generate_sorted_grid(size, lattice, rotation_angle; box, segment_ratio, segment_angle, shell_thickness)

Generate a 2D grid in the requested box and return points sorted by time.

# Arguments
- `size`: Number of points requested for grid generation.
- `lattice`: Lattice identifier understood by `QG.generate_grid_2d_in_box`.
- `rotation_angle`: Rotation angle in degrees applied during grid generation.

# Keyword Arguments
- `box`: Spatial-temporal box bounds `((x_min, t_min), (x_max, t_max))`.
- `segment_ratio`: Grid-generation shape parameter `b`.
- `segment_angle`: Grid-generation angle parameter in degrees.
- `shell_thickness`: Optional shell-thickness parameter passed through to grid generation.

# Returns
- `quad_grid`: Time-sorted grid point collection.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised for invalid box bounds.
"""
function generate_sorted_grid(
    size::Int,
    lattice::String,
    rotation_angle::Float64;
    box::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}=((-1.,-1.),(1.,1.)),
    segment_ratio::Float64=2.,
    segment_angle::Float64=60.,
    shell_thickness::Union{Nothing,Float64}=nothing,
)
    if !(size > 0)
        throw(DomainError(size, "size must be positive"))
    end
    if !(segment_ratio > 0)
        throw(DomainError(segment_ratio, "segment_ratio must be positive"))
    end
    if shell_thickness !== nothing && !(shell_thickness >= 0)
        throw(DomainError(shell_thickness, "shell_thickness must be nonnegative"))
    end
    (x_min, t_min), (x_max, t_max) = box
    if !(x_min < x_max && t_min < t_max)
        throw(ArgumentError("box must satisfy x_min < x_max and t_min < t_max"))
    end
    mink = CausalSets.MinkowskiManifold{2}()
    quad_grid_unsorted = QuantumGrav.generate_grid_2d_in_box(
        size,
        lattice,
        box;
        rotate_deg = rotation_angle,
        b = segment_ratio,
        gamma_deg = segment_angle,
        shell_thickness = shell_thickness,
    )
    return QuantumGrav.sort_grid_by_time_from_manifold(mink, quad_grid_unsorted)
end

"""
    compute_fourier_grid_deviation(
        comp_hist,
        size,
        lattice;
        P_max,
        rng,
        segment_ratio,
        segment_angle,
        rotation_angle,
        max_peak_order,
    )

Compute Fourier-analysis data for grid-vs-reference abundance deviation.

The function generates a grid causal set, computes relative deviation
`grid_abundance ./ comp_hist .- 1`, applies `FFTW.fft`, and extracts dominant
peak diagnostics over positive frequencies above `1 / P_max`.

# Arguments
- `comp_hist`: Reference abundance histogram.
- `size`: Grid causal-set size used for generation.
- `lattice`: Grid/lattice identifier for generation.

# Keyword Arguments
- `P_max`: Maximum period considered for peak search (`f_min = 1 / P_max`).
- `rng`: Random number generator used for causal-set generation.
- `segment_ratio`: Grid-generation shape parameter `b`.
- `segment_angle`: Grid-generation angle parameter in degrees.
- `rotation_angle`: Optional rotation angle in degrees.
- `max_peak_order`: Maximum number of distinct dominant peak periods to return.

# Returns
- `spec`: Named tuple with fields:
  `idx`, `spectrum`, `freqs`, `keep`, `f_peak`, `P_est`, and `peak_rows`.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised when required histogram structure assumptions are not met.
- `DimensionMismatch`: Raised when generated abundances are shorter than required by `comp_hist`.
"""
function compute_fourier_grid_deviation(
    comp_hist::Vector{Float64},
    size::Int64,
    lattice::String;
    P_max::Float64=300.,
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    segment_ratio::Float64=1.,
    segment_angle::Float64=60.,
    rotation_angle::Union{Float64,Nothing}=nothing,
    max_peak_order::Int=5,
)
    if isempty(comp_hist)
        throw(ArgumentError("comp_hist must be non-empty"))
    end
    if any(!isfinite, comp_hist)
        throw(ArgumentError("comp_hist must contain only finite values"))
    end
    if !(size > 0)
        throw(DomainError(size, "size must be positive"))
    end
    if !(P_max > 0)
        throw(DomainError(P_max, "P_max must be positive"))
    end
    if !(segment_ratio > 0)
        throw(DomainError(segment_ratio, "segment_ratio must be positive"))
    end
    if !(max_peak_order >= 1)
        throw(DomainError(max_peak_order, "max_peak_order must be >= 1"))
    end

    first_zero = findfirst(iszero, comp_hist)
    if first_zero === nothing
        throw(ArgumentError("comp_hist must contain a zero sentinel entry marking truncation"))
    end
    if first_zero <= 2
        throw(ArgumentError("zero sentinel in comp_hist must occur at index >= 3"))
    end
    idx = first_zero - 1
    denom = comp_hist[2:idx]
    if any(==(0.0), denom)
        throw(ArgumentError("comp_hist[2:idx] must be nonzero to form relative deviation"))
    end

    grid_cset, _, _, _ = QuantumGrav.create_grid_causet_in_boundary_2D_polynomial_manifold(
        size,
        lattice,
        CausalSets.BoxBoundary{2}(((-1., -1.), (1., 1.))),
        rng,
        1,
        2.;
        a = 1.,
        b = segment_ratio,
        gamma_deg = segment_angle,
        rotate_deg = rotation_angle,
    )
    grid_cset_abundances = CausalSets.cardinality_abundances(grid_cset)
    if length(grid_cset_abundances) < idx
        throw(
            DimensionMismatch(
                "grid abundance length $(length(grid_cset_abundances)) is smaller than required index idx=$idx from comp_hist",
            ),
        )
    end
    r_comp_grid_man = grid_cset_abundances[2:idx] ./ comp_hist[2:idx] .- 1
    r_dev_fou_comp_grid_man = abs.(FFTW.fft(r_comp_grid_man))
    r_dev_freqs_comp_grid_man = (0:length(r_dev_fou_comp_grid_man)-1) ./ length(r_dev_fou_comp_grid_man)

    f_min = 1 / P_max
    freqs = r_dev_freqs_comp_grid_man
    half = 1:fld(length(freqs), 2)
    keep = [i for i in half if freqs[i] >= f_min]
    if isempty(keep)
        throw(DomainError(P_max, "no positive-frequency bins remain for peak search; increase P_max"))
    end
    peak_idx = keep[argmax(r_dev_fou_comp_grid_man[keep])]
    f_peak = r_dev_freqs_comp_grid_man[peak_idx]
    P_est = 1 / f_peak

    min_freq_for_peaks = 1 / 13
    keep_for_peaks = [i for i in keep if r_dev_freqs_comp_grid_man[i] >= min_freq_for_peaks]
    peak_rows = NamedTuple{(:f, :P, :A),Tuple{Float64,Float64,Float64}}[]
    if !isempty(keep_for_peaks)
        idxs = sortperm(r_dev_fou_comp_grid_man[keep_for_peaks]; rev = true)
        printed_periods = Float64[]
        for i in idxs
            f = r_dev_freqs_comp_grid_man[keep_for_peaks[i]]
            P = 1 / f
            if any(abs(P - P0) <= 0.02 for P0 in printed_periods)
                continue
            end
            A = 2 * abs(r_dev_fou_comp_grid_man[keep_for_peaks[i]]) / length(r_comp_grid_man)
            push!(peak_rows, (f = f, P = P, A = A))
            push!(printed_periods, P)
            if length(printed_periods) >= max_peak_order
                break
            end
        end
    end

    return (
        idx = idx,
        spectrum = r_dev_fou_comp_grid_man,
        freqs = r_dev_freqs_comp_grid_man,
        keep = keep,
        f_peak = f_peak,
        P_est = P_est,
        peak_rows = peak_rows,
    )
end
