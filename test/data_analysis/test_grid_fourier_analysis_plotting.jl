@testsnippet setupGridFourier begin
    using Test
    using Random
    using FFTW
    using CairoMakie
    using LaTeXStrings
    using Printf
    using Colors
    using CausalSets
    using QuantumGrav

    const CS = CausalSets
    const QG = QuantumGrav

end

@testitem "grid_fourier_analysis_and_plots: grid and FFT plot" setup=[setupGridFourier] begin
    if isdefined(QG, :generate_grid_2d_in_box)
        fig, ax = CausalSetZoology.create_grid_and_plot(16, "square", 0.0; magnification = 0.8)
        @test fig isa Figure
        @test ax isa Axis
    else
        @test_throws UndefVarError CausalSetZoology.create_grid_and_plot(16, "square", 0.0; magnification = 0.8)
    end

    comp_hist = ones(60)
    comp_hist[30] = 0.0
    if isdefined(QG, :create_grid_causet_in_boundary_2D_polynomial_manifold)
        f = CausalSetZoology.fourier_transform_grid_deviation(comp_hist, 30, "square"; P_max = 20.0, rng = Random.Xoshiro(7), max_peak_order = 2)
        @test f isa Figure
    else
        @test_throws UndefVarError CausalSetZoology.fourier_transform_grid_deviation(comp_hist, 30, "square"; P_max = 20.0, rng = Random.Xoshiro(7), max_peak_order = 2)
    end
end
