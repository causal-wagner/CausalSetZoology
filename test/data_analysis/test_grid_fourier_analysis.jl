@testsnippet setupGridFourierAnalysis begin
    using Test
    import Random
    import CausalSets as CS
    import QuantumGrav as QG
end

# Validates baseline generation properties and time-sorting behavior.
@testitem "grid_fourier_analysis: generate_sorted_grid basic" setup=[setupGridFourierAnalysis] begin
    # Default generation should return requested number of points.
    g = CausalSetZoology.generate_sorted_grid(16, "square", 0.0)
    @test length(g) == 16

    # Output should already be sorted by time in Minkowski manifold.
    mink = CS.MinkowskiManifold{2}()
    @test g == QG.sort_grid_by_time_from_manifold(mink, g)

    # Non-default keyword path should also work and stay sorted.
    g2 = CausalSetZoology.generate_sorted_grid(
        12,
        "square",
        15.0;
        box = ((-2.0, -1.0), (2.0, 1.0)),
        segment_ratio = 1.5,
        segment_angle = 50.0,
        shell_thickness = 0.1,
    )
    @test length(g2) == 12
    @test g2 == QG.sort_grid_by_time_from_manifold(mink, g2)
end

# Checks that geometric parameters actually affect generated grids.
@testitem "grid_fourier_analysis: generate_sorted_grid parameter influence" setup=[setupGridFourierAnalysis] begin
    # Changing geometric parameters should generally change the generated/sorted grid.
    g_rot0 = CausalSetZoology.generate_sorted_grid(20, "square", 0.0; segment_ratio = 1.5, segment_angle = 55.0)
    g_rot30 = CausalSetZoology.generate_sorted_grid(20, "square", 30.0; segment_ratio = 1.5, segment_angle = 55.0)
    @test g_rot0 != g_rot30

    g_ang55 = CausalSetZoology.generate_sorted_grid(20, "square", 0.0; segment_ratio = 1.5, segment_angle = 55.0)
    g_ang65 = CausalSetZoology.generate_sorted_grid(20, "square", 0.0; segment_ratio = 1.5, segment_angle = 65.0)
    @test g_ang55 != g_ang65
end

# Validates argument/domain checks for grid generation.
@testitem "grid_fourier_analysis: generate_sorted_grid validation" setup=[setupGridFourierAnalysis] begin
    # Non-positive size should fail.
    @test_throws DomainError CausalSetZoology.generate_sorted_grid(0, "square", 0.0)
    # Non-positive segment ratio should fail.
    @test_throws DomainError CausalSetZoology.generate_sorted_grid(8, "square", 0.0; segment_ratio = 0.0)
    # Negative shell thickness should fail.
    @test_throws DomainError CausalSetZoology.generate_sorted_grid(8, "square", 0.0; shell_thickness = -0.1)
    # Invalid box bounds should fail.
    @test_throws ArgumentError CausalSetZoology.generate_sorted_grid(8, "square", 0.0; box = ((1.0, -1.0), (1.0, 1.0)))
end

# Validates explicit helper checks for Fourier-input preconditions.
@testitem "grid_fourier_analysis helpers: validate_fourier_inputs" setup=[setupGridFourierAnalysis] begin
    # Valid inputs should return `nothing`.
    @test CausalSetZoology.validate_fourier_inputs([1.0, 1.0, 0.0, 1.0], 8; P_max = 10.0, segment_ratio = 1.0, max_peak_order = 3) === nothing
end

# Validates throw branches for Fourier-input precondition helper.
@testitem "grid_fourier_analysis helpers: validate_fourier_inputs validation" setup=[setupGridFourierAnalysis] begin
    # Empty/non-finite histogram checks.
    @test_throws ArgumentError CausalSetZoology.validate_fourier_inputs(Float64[], 8; P_max = 10.0, segment_ratio = 1.0, max_peak_order = 3)
    @test_throws ArgumentError CausalSetZoology.validate_fourier_inputs([1.0, Inf, 0.0], 8; P_max = 10.0, segment_ratio = 1.0, max_peak_order = 3)

    # Numeric domain checks.
    @test_throws DomainError CausalSetZoology.validate_fourier_inputs([1.0, 1.0, 0.0], 0; P_max = 10.0, segment_ratio = 1.0, max_peak_order = 3)
    @test_throws DomainError CausalSetZoology.validate_fourier_inputs([1.0, 1.0, 0.0], 8; P_max = 0.0, segment_ratio = 1.0, max_peak_order = 3)
    @test_throws DomainError CausalSetZoology.validate_fourier_inputs([1.0, 1.0, 0.0], 8; P_max = 10.0, segment_ratio = 0.0, max_peak_order = 3)
    @test_throws DomainError CausalSetZoology.validate_fourier_inputs([1.0, 1.0, 0.0], 8; P_max = 10.0, segment_ratio = 1.0, max_peak_order = 0)
end

# Verifies context extraction helper for zero-sentinel truncation logic.
@testitem "grid_fourier_analysis helpers: prepare_fourier_context" setup=[setupGridFourierAnalysis] begin
    comp_hist = [5.0, 2.0, 4.0, 6.0, 0.0, 9.0]
    ctx = CausalSetZoology.prepare_fourier_context(comp_hist, 20.0)

    # first_zero=5 -> idx=4; denominator slice is comp_hist[2:4].
    @test ctx.idx == 4
    @test ctx.denom == [2.0, 4.0, 6.0]
    @test ctx.f_min == 0.05
end

# Validates throw branches for context extraction helper.
@testitem "grid_fourier_analysis helpers: prepare_fourier_context validation" setup=[setupGridFourierAnalysis] begin
    # Missing zero sentinel.
    @test_throws ArgumentError CausalSetZoology.prepare_fourier_context([1.0, 2.0, 3.0], 10.0)

    # Sentinel too early (first_zero <= 2).
    @test_throws ArgumentError CausalSetZoology.prepare_fourier_context([1.0, 0.0, 3.0], 10.0)

end

# Verifies candidate-spectrum helper returns reproducible abundance vectors for fixed seed.
@testitem "grid_fourier_analysis helpers: compute_candidate_spectrum" setup=[setupGridFourierAnalysis] begin
    a = CausalSetZoology.compute_candidate_spectrum(
        20,
        "square",
        Random.Xoshiro(123);
        segment_ratio = 1.2,
        segment_angle = 55.0,
        rotation_angle = 10.0,
    )
    b = CausalSetZoology.compute_candidate_spectrum(
        20,
        "square",
        Random.Xoshiro(123);
        segment_ratio = 1.2,
        segment_angle = 55.0,
        rotation_angle = 10.0,
    )
    @test a == b
    @test !isempty(a)
    @test all(isfinite, Float64.(a))
end

# Verifies Fourier accumulation helper structure and explicit branches.
@testitem "grid_fourier_analysis helpers: accumulate_deviation!" setup=[setupGridFourierAnalysis] begin
    # Candidate equals reference -> zero relative deviation and zero spectrum.
    candidate = [10.0, 2.0, 4.0, 6.0, 8.0]
    reference = [2.0, 4.0, 6.0, 8.0]
    idx = 5
    acc = CausalSetZoology.accumulate_deviation!(
        candidate,
        reference,
        idx,
        0.2;
        min_freq_for_peaks = 2.0,  # force empty keep_for_peaks branch
        max_peak_order = 3,
        P_max = 5.0,
    )
    @test length(acc.spectrum) == idx - 1
    @test all(==(0.0), acc.spectrum)
    @test !isempty(acc.keep)
    @test acc.f_peak in acc.freqs
    @test acc.P_est ≈ 1 / acc.f_peak atol = 1e-12
    @test isempty(acc.peak_rows)
end

# Validates throw branches for Fourier accumulation helper.
@testitem "grid_fourier_analysis helpers: accumulate_deviation! validation" setup=[setupGridFourierAnalysis] begin
    # Candidate abundances too short for requested idx.
    @test_throws DimensionMismatch CausalSetZoology.accumulate_deviation!(
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        5,
        0.2;
        P_max = 5.0,
    )

    # Keep set empty when f_min is too large.
    @test_throws DomainError CausalSetZoology.accumulate_deviation!(
        [10.0, 2.0, 4.0, 6.0, 8.0],
        [2.0, 4.0, 6.0, 8.0],
        5,
        2.0;
        P_max = 0.5,
    )
end

# Verifies helper path where admissible peaks are actually reported.
@testitem "grid_fourier_analysis helpers: accumulate_deviation! nonempty peaks" setup=[setupGridFourierAnalysis] begin
    # Build a synthetic relative-deviation signal with multiple frequencies.
    N = 40
    idx = N + 1
    n = collect(0:N-1)
    rel = @. 0.30 * sin(2π * 4 * n / N) + 0.18 * sin(2π * 7 * n / N)
    reference = ones(Float64, N)
    candidate = vcat([1.0], 1 .+ rel)

    acc = CausalSetZoology.accumulate_deviation!(
        candidate,
        reference,
        idx,
        1 / 80;
        min_freq_for_peaks = 1 / 13,
        max_peak_order = 5,
        P_max = 80.0,
    )

    @test !isempty(acc.peak_rows)
    @test all(row.f > 0 && row.P > 0 && row.A >= 0 for row in acc.peak_rows)
    @test all(isapprox(row.P, 1 / row.f; atol = 1e-12) for row in acc.peak_rows)
end

# Verifies direct max_peak_order truncation behavior inside the helper.
@testitem "grid_fourier_analysis helpers: accumulate_deviation! peak truncation" setup=[setupGridFourierAnalysis] begin
    # Use the same deterministic multi-frequency signal and vary max_peak_order.
    N = 40
    idx = N + 1
    n = collect(0:N-1)
    rel = @. 0.30 * sin(2π * 4 * n / N) + 0.18 * sin(2π * 7 * n / N) + 0.10 * sin(2π * 9 * n / N)
    reference = ones(Float64, N)
    candidate = vcat([1.0], 1 .+ rel)

    acc1 = CausalSetZoology.accumulate_deviation!(
        candidate,
        reference,
        idx,
        1 / 80;
        min_freq_for_peaks = 1 / 13,
        max_peak_order = 1,
        P_max = 80.0,
    )
    acc3 = CausalSetZoology.accumulate_deviation!(
        candidate,
        reference,
        idx,
        1 / 80;
        min_freq_for_peaks = 1 / 13,
        max_peak_order = 3,
        P_max = 80.0,
    )

    @test length(acc1.peak_rows) == 1
    @test 1 <= length(acc3.peak_rows) <= 3
    @test length(acc3.peak_rows) >= length(acc1.peak_rows)
end

# Verifies explicit period de-duplication rule inside the helper output.
@testitem "grid_fourier_analysis helpers: accumulate_deviation! peak distinctness" setup=[setupGridFourierAnalysis] begin
    N = 40
    idx = N + 1
    n = collect(0:N-1)
    rel = @. 0.30 * sin(2π * 4 * n / N) + 0.18 * sin(2π * 7 * n / N) + 0.10 * sin(2π * 9 * n / N)
    reference = ones(Float64, N)
    candidate = vcat([1.0], 1 .+ rel)

    acc = CausalSetZoology.accumulate_deviation!(
        candidate,
        reference,
        idx,
        1 / 80;
        min_freq_for_peaks = 1 / 13,
        max_peak_order = 6,
        P_max = 80.0,
    )

    Ps = [row.P for row in acc.peak_rows]
    @test all(abs(Ps[i] - Ps[j]) > 0.02 for i in eachindex(Ps) for j in eachindex(Ps) if i < j)
end

# Verifies output-shaping helper for final public return tuple.
@testitem "grid_fourier_analysis helpers: finalize_fourier_deviation" setup=[setupGridFourierAnalysis] begin
    acc = (
        spectrum = [1.0, 2.0],
        freqs = [0.0, 0.5],
        keep = [2],
        f_peak = 0.5,
        P_est = 2.0,
        peak_rows = [(f = 0.5, P = 2.0, A = 1.0)],
    )
    out = CausalSetZoology.finalize_fourier_deviation(3, acc)
    @test keys(out) == (:idx, :spectrum, :freqs, :keep, :f_peak, :P_est, :peak_rows)
    @test out.idx == 3
    @test out.spectrum == acc.spectrum
    @test out.freqs == acc.freqs
    @test out.keep == acc.keep
    @test out.f_peak == acc.f_peak
    @test out.P_est == acc.P_est
    @test out.peak_rows == acc.peak_rows
end

# Verifies structure and invariants of Fourier deviation outputs.
@testitem "grid_fourier_analysis: compute_fourier_grid_deviation structure" setup=[setupGridFourierAnalysis] begin
    rng = Random.Xoshiro(7)
    comp_hist = ones(24)
    comp_hist[12] = 0.0

    spec = CausalSetZoology.compute_fourier_grid_deviation(
        comp_hist,
        24,
        "square";
        P_max = 40.0,
        rng = rng,
        segment_ratio = 1.2,
        segment_angle = 55.0,
        rotation_angle = 10.0,
        max_peak_order = 3,
    )

    @test keys(spec) == (:idx, :spectrum, :freqs, :keep, :f_peak, :P_est, :peak_rows)
    @test spec.idx == 11
    @test length(spec.spectrum) == spec.idx - 1
    @test length(spec.freqs) == length(spec.spectrum)
    @test !isempty(spec.keep)
    @test all(1 <= i <= fld(length(spec.freqs), 2) for i in spec.keep)
    @test all(spec.freqs[i] >= 1 / 40.0 for i in spec.keep)
    @test spec.f_peak in spec.freqs
    @test spec.P_est ≈ 1 / spec.f_peak atol = 1e-12
    @test all(isfinite, spec.spectrum)

    @test length(spec.peak_rows) <= 3
    @test all(row.f > 0 && row.P > 0 && row.A >= 0 for row in spec.peak_rows)
    @test all(isapprox(row.P, 1 / row.f; atol = 1e-12) for row in spec.peak_rows)
end

# Checks reproducibility for identical RNG seeds and inputs.
@testitem "grid_fourier_analysis: compute_fourier_grid_deviation reproducibility" setup=[setupGridFourierAnalysis] begin
    comp_hist = ones(24)
    comp_hist[12] = 0.0

    a = CausalSetZoology.compute_fourier_grid_deviation(comp_hist, 24, "square"; rng = Random.Xoshiro(99), P_max = 40.0)
    b = CausalSetZoology.compute_fourier_grid_deviation(comp_hist, 24, "square"; rng = Random.Xoshiro(99), P_max = 40.0)

    @test a.idx == b.idx
    @test a.keep == b.keep
    @test a.f_peak ≈ b.f_peak atol = 1e-12
    @test a.P_est ≈ b.P_est atol = 1e-12
    @test a.spectrum ≈ b.spectrum atol = 1e-12
    @test a.freqs ≈ b.freqs atol = 1e-12
    @test a.peak_rows == b.peak_rows
end

# Verifies peak list truncation by `max_peak_order`.
@testitem "grid_fourier_analysis: compute_fourier_grid_deviation peak truncation" setup=[setupGridFourierAnalysis] begin
    # max_peak_order should truncate the number of reported distinct peaks.
    comp_hist = ones(40)
    comp_hist[24] = 0.0
    rng_seed = 77

    spec1 = CausalSetZoology.compute_fourier_grid_deviation(
        comp_hist,
        40,
        "square";
        rng = Random.Xoshiro(rng_seed),
        P_max = 80.0,
        max_peak_order = 1,
    )
    spec4 = CausalSetZoology.compute_fourier_grid_deviation(
        comp_hist,
        40,
        "square";
        rng = Random.Xoshiro(rng_seed),
        P_max = 80.0,
        max_peak_order = 4,
    )

    @test length(spec1.peak_rows) == 1
    @test length(spec4.peak_rows) == 4
end