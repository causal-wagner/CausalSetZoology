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
    rotation_angle::Real;
    box::Tuple{Tuple{Real,Real},Tuple{Real,Real}}=((-1.,-1.),(1.,1.)),
    segment_ratio::Real=2.,
    segment_angle::Real=60.,
    shell_thickness::Union{Nothing,Real}=nothing,
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
    validate_fourier_inputs(comp_hist, size; P_max, segment_ratio, max_peak_order)

Validate domain and shape preconditions for Fourier-grid analysis.

# Arguments
- `comp_hist`: Reference abundance histogram.
- `size`: Grid causal-set size used for generation.

# Keyword Arguments
- `P_max`: Maximum period considered for peak search (`f_min = 1 / P_max`).
- `segment_ratio`: Grid-generation shape parameter `b`.
- `max_peak_order`: Maximum number of distinct dominant peak periods to return.

# Returns
- `nothing`: Returns `nothing` when all checks pass.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised when histogram content is empty or non-finite.
"""
function validate_fourier_inputs(
    comp_hist::AbstractVector{<:Real},
    size::Int;
    P_max::Real,
    segment_ratio::Real,
    max_peak_order::Int,
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
    return nothing
end

"""
    prepare_fourier_context(comp_hist, P_max)

Build truncation/frequency context from the reference histogram.

# Arguments
- `comp_hist`: Reference abundance histogram containing a zero sentinel.
- `P_max`: Maximum period considered for peak search.

# Returns
- `context`: Named tuple `(idx, denom, f_min)` where `idx` is the last index before the
  zero sentinel, `denom = comp_hist[2:idx]`, and `f_min = 1 / P_max`.

# Throws
- `ArgumentError`: Raised when sentinel or denominator assumptions are violated.
"""
function prepare_fourier_context(comp_hist::AbstractVector{<:Real}, P_max::Real)
    first_zero = findfirst(iszero, comp_hist)
    if first_zero === nothing
        throw(ArgumentError("comp_hist must contain a zero sentinel entry marking truncation"))
    end
    if first_zero <= 2
        throw(ArgumentError("zero sentinel in comp_hist must occur at index >= 3"))
    end
    idx = first_zero - 1
    denom = comp_hist[2:idx]
    f_min = 1 / P_max
    return (idx = idx, denom = denom, f_min = f_min)
end

"""
    compute_reference_spectrum(comp_hist, idx)

Extract the reference histogram slice used for relative-deviation computation.

# Arguments
- `comp_hist`: Reference abundance histogram.
- `idx`: Last included index (inclusive).

# Returns
- `slice`: `comp_hist[2:idx]`.
"""
function compute_reference_spectrum(comp_hist::AbstractVector{<:Real}, idx::Int)
    # Keep reference slice explicit to isolate truncation logic and stabilize call shape.
    return comp_hist[2:idx]
end

"""
    compute_candidate_spectrum(size, lattice, rng; segment_ratio, segment_angle, rotation_angle)

Generate a grid causal set and return its cardinality abundances.

# Arguments
- `size`: Grid causal-set size used for generation.
- `lattice`: Grid/lattice identifier for generation.
- `rng`: Random number generator used for causal-set generation.

# Keyword Arguments
- `segment_ratio`: Grid-generation shape parameter `b`.
- `segment_angle`: Grid-generation angle parameter in degrees.
- `rotation_angle`: Optional rotation angle in degrees.

# Returns
- `abundances`: Cardinality abundance vector for the generated grid causal set.
"""
function compute_candidate_spectrum(
    size::Int,
    lattice::String,
    rng::Random.AbstractRNG;
    segment_ratio::Real,
    segment_angle::Real,
    rotation_angle::Union{Real,Nothing},
)
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
    return CausalSets.cardinality_abundances(grid_cset)
end

"""
    accumulate_deviation!(candidate_abundances, reference_slice, idx, f_min; min_freq_for_peaks=1/13, max_peak_order=5, P_max=1/f_min)

Compute relative-deviation Fourier spectrum and dominant peak diagnostics.

# Arguments
- `candidate_abundances`: Abundance vector from the generated grid causal set.
- `reference_slice`: Reference abundance slice from `comp_hist`.
- `idx`: Last included index (inclusive).
- `f_min`: Minimum admissible frequency for peak search.

# Keyword Arguments
- `min_freq_for_peaks`: Additional frequency cutoff for reported peak rows.
- `max_peak_order`: Maximum number of distinct dominant peak periods to return.
- `P_max`: Maximum period parameter (used in validation error message context).

# Returns
- `acc`: Named tuple with fields `spectrum`, `freqs`, `keep`, `f_peak`, `P_est`, and `peak_rows`.

# Throws
- `DimensionMismatch`: Raised when `candidate_abundances` is too short for `idx`.
- `DomainError`: Raised when no admissible positive-frequency bins remain.
"""
function accumulate_deviation!(
    candidate_abundances::AbstractVector{<:Real},
    reference_slice::AbstractVector{<:Real},
    idx::Int,
    f_min::Real;
    min_freq_for_peaks::Real = 1 / 13,
    max_peak_order::Int = 5,
    P_max::Real = 1 / f_min,
)
    if length(candidate_abundances) < idx
        throw(
            DimensionMismatch(
                "grid abundance length $(length(candidate_abundances)) is smaller than required index idx=$idx from comp_hist",
            ),
        )
    end

    rel_dev = candidate_abundances[2:idx] ./ reference_slice .- 1
    spectrum = abs.(FFTW.fft(rel_dev))
    freqs = (0:length(spectrum)-1) ./ length(spectrum)
    half = 1:fld(length(freqs), 2)
    keep = [i for i in half if freqs[i] >= f_min]
    if isempty(keep)
        throw(DomainError(P_max, "no positive-frequency bins remain for peak search; increase P_max"))
    end

    peak_idx = keep[argmax(spectrum[keep])]
    f_peak = freqs[peak_idx]
    P_est = 1 / f_peak

    keep_for_peaks = [i for i in keep if freqs[i] >= min_freq_for_peaks]
    peak_rows = NamedTuple{(:f, :P, :A),Tuple{Float64,Float64,Float64}}[]
    if !isempty(keep_for_peaks)
        idxs = sortperm(spectrum[keep_for_peaks]; rev = true)
        printed_periods = Float64[]
        for i in idxs
            f = freqs[keep_for_peaks[i]]
            P = 1 / f
            if any(abs(P - P0) <= 0.02 for P0 in printed_periods)
                continue
            end
            A = 2 * abs(spectrum[keep_for_peaks[i]]) / length(rel_dev)
            push!(peak_rows, (f = f, P = P, A = A))
            push!(printed_periods, P)
            if length(printed_periods) >= max_peak_order
                break
            end
        end
    end

    return (
        spectrum = spectrum,
        freqs = freqs,
        keep = keep,
        f_peak = f_peak,
        P_est = P_est,
        peak_rows = peak_rows,
    )
end

"""
    finalize_fourier_deviation(idx, acc)

Assemble public return shape for `compute_fourier_grid_deviation`.

# Arguments
- `idx`: Last included index (inclusive) before zero-sentinel truncation.
- `acc`: Fourier summary from `accumulate_deviation!`.

# Returns
- `spec`: Named tuple with fields `idx`, `spectrum`, `freqs`, `keep`, `f_peak`, `P_est`, and `peak_rows`.
"""
function finalize_fourier_deviation(idx::Int, acc)
    return (
        idx = idx,
        spectrum = acc.spectrum,
        freqs = acc.freqs,
        keep = acc.keep,
        f_peak = acc.f_peak,
        P_est = acc.P_est,
        peak_rows = acc.peak_rows,
    )
end

"""
    compute_fourier_grid_deviation(comp_hist, size, lattice; P_max, rng, segment_ratio, segment_angle, rotation_angle, max_peak_order)

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
- `spec`: Named tuple with fields `idx`, `spectrum`, `freqs`, `keep`, `f_peak`, `P_est`, and `peak_rows`.

# Throws
- `DomainError`: Raised for out-of-range numeric parameters.
- `ArgumentError`: Raised when required histogram structure assumptions are not met.
- `DimensionMismatch`: Raised when generated abundances are shorter than required by `comp_hist`.
"""
function compute_fourier_grid_deviation(
    comp_hist::AbstractVector{<:Real},
    size::Int,
    lattice::String;
    P_max::Real=300.,
    rng::Random.AbstractRNG=Random.GLOBAL_RNG,
    segment_ratio::Real=1.,
    segment_angle::Real=60.,
    rotation_angle::Union{Real,Nothing}=nothing,
    max_peak_order::Int=5,
)
    validate_fourier_inputs(
        comp_hist,
        size;
        P_max = P_max,
        segment_ratio = segment_ratio,
        max_peak_order = max_peak_order,
    )

    ctx = prepare_fourier_context(comp_hist, P_max)
    ref_hist = compute_reference_spectrum(comp_hist, ctx.idx)
    candidate_abundances = compute_candidate_spectrum(
        size,
        lattice,
        rng;
        segment_ratio = segment_ratio,
        segment_angle = segment_angle,
        rotation_angle = rotation_angle,
    )
    acc = accumulate_deviation!(
        candidate_abundances,
        ref_hist,
        ctx.idx,
        ctx.f_min;
        max_peak_order = max_peak_order,
        P_max = P_max,
    )
    return finalize_fourier_deviation(ctx.idx, acc)
end
