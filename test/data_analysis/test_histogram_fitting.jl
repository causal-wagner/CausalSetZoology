@testsnippet setupHistogramFitting begin
    using Test
    using Statistics
    using Random
    using LinearAlgebra
    using Optim

    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "utils.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "histogram_fitting.jl"))
end

@testitem "histogram_fitting: fit_curve and bin slicing" setup=[setupHistogramFitting] begin
    rng = Random.Xoshiro(123)
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))

    p = fit_curve(ys, f, (:a, :b); x_values = xs, init = (a = 1.0, b = 0.0), multistart = 3, rng = rng)
    @test p.a ≈ 2.0 atol = 1e-2
    @test p.b ≈ 1.0 atol = 1e-2

    stds = fill(0.1, length(ys))
    gof = fit_curve(ys, f, (:a, :b); x_values = xs, stds = stds, minimize_χ² = true, goodness_of_fit = true, init = (a = 1.5, b = 0.5))
    @test haskey(gof, :χ²)
    @test haskey(gof, :params)

    cov = fit_curve(ys, f, (:a, :b); x_values = xs, return_cov = true, init = (a = 1.0, b = 0.0))
    @test haskey(cov, :cov)
    @test haskey(cov, :stderr)

    bfit = fit_histogram_bins(ys, f, (:a, :b), 3, 8; x_values = xs, init = (a = 1.0, b = 0.0))
    @test bfit.a ≈ 2.0 atol = 1e-2

    @test_throws AssertionError fit_histogram_bins(ys, f, (:a, :b), 0, 3)
    @test_throws ErrorException fit_curve(ys, f, (:a, :b); std_fn = (y, yhat, s, p) -> s)
    @test_throws AssertionError fit_curve(ys, f, (:a, :b); minimize_χ² = true)
    @test_throws AssertionError fit_curve(ys, f, (:a, :b); stds = [0.1], minimize_χ² = true)
    @test_throws TypeError fit_curve(ys, f, (:a, :b); bounds = (1.0, 2.0))
    @test_throws ErrorException fit_curve(ys, f, (:a, :b); bootstrap_errorbars = true, return_cov = true)
end
