@testsnippet setupGridFourierAnalysis begin
    using Test
    import Random
    import CausalSets as CS
    import QuantumGrav as QG
end

@testitem "grid_fourier_analysis: generate_sorted_grid basic" setup=[setupGridFourierAnalysis] begin
    g = CausalSetZoology.generate_sorted_grid(16, "square", 0.0)
    @test length(g) == 16

    mink = CS.MinkowskiManifold{2}()
    @test g == QG.sort_grid_by_time_from_manifold(mink, g)

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

@testitem "grid_fourier_analysis: generate_sorted_grid validation" setup=[setupGridFourierAnalysis] begin
    @test_throws DomainError CausalSetZoology.generate_sorted_grid(0, "square", 0.0)
    @test_throws DomainError CausalSetZoology.generate_sorted_grid(8, "square", 0.0; segment_ratio = 0.0)
    @test_throws DomainError CausalSetZoology.generate_sorted_grid(8, "square", 0.0; shell_thickness = -0.1)
    @test_throws ArgumentError CausalSetZoology.generate_sorted_grid(8, "square", 0.0; box = ((1.0, -1.0), (1.0, 1.0)))
end

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

@testitem "grid_fourier_analysis: compute_fourier_grid_deviation domain validation" setup=[setupGridFourierAnalysis] begin
    comp_hist = [1.0, 1.0, 0.0, 1.0]

    @test_throws DomainError CausalSetZoology.compute_fourier_grid_deviation(comp_hist, 0, "square")
    @test_throws DomainError CausalSetZoology.compute_fourier_grid_deviation(comp_hist, 8, "square"; P_max = 0.0)
    @test_throws DomainError CausalSetZoology.compute_fourier_grid_deviation(comp_hist, 8, "square"; segment_ratio = 0.0)
    @test_throws DomainError CausalSetZoology.compute_fourier_grid_deviation(comp_hist, 8, "square"; max_peak_order = 0)
end

@testitem "grid_fourier_analysis: compute_fourier_grid_deviation argument validation" setup=[setupGridFourierAnalysis] begin
    @test_throws ArgumentError CausalSetZoology.compute_fourier_grid_deviation(Float64[], 8, "square")
    @test_throws ArgumentError CausalSetZoology.compute_fourier_grid_deviation([1.0, Inf, 0.0], 8, "square")
    @test_throws ArgumentError CausalSetZoology.compute_fourier_grid_deviation([1.0, 1.0, 1.0], 8, "square")
    @test_throws ArgumentError CausalSetZoology.compute_fourier_grid_deviation([1.0, 0.0, 1.0], 8, "square")
    @test_throws ArgumentError CausalSetZoology.compute_fourier_grid_deviation([1.0, 0.0, 0.0, 1.0], 8, "square")
end

@testitem "grid_fourier_analysis: compute_fourier_grid_deviation spectral-edge validation" setup=[setupGridFourierAnalysis] begin
    # idx = 4 -> FFT length 3 -> positive-half has only frequency 0; with f_min > 0 no bins remain.
    comp_hist_short = [1.0, 1.0, 1.0, 1.0, 0.0]
    @test_throws DomainError CausalSetZoology.compute_fourier_grid_deviation(comp_hist_short, 12, "square"; P_max = 0.5)
end

@testitem "grid_fourier_analysis: compute_fourier_grid_deviation dimension mismatch" setup=[setupGridFourierAnalysis] begin
    # idx intentionally large for tiny generated causal set.
    comp_hist_long = ones(80)
    comp_hist_long[70] = 0.0
    @test_throws DimensionMismatch CausalSetZoology.compute_fourier_grid_deviation(comp_hist_long, 8, "square"; rng = Random.Xoshiro(3))
end
