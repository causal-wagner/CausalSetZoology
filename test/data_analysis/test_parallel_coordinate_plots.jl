@testsnippet setupParallelPlots begin
    using Test
    using Statistics
    using Random
    using CairoMakie
    using DataFrames
    using CategoricalArrays
    using AlgebraOfGraphics
    using StatsBase
    using Observables
    using Colors
    using Printf

    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "plot_theme.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "parallel_coordinate_plots.jl"))
end

@testitem "parallel_coordinate_plots: transform, dataframe, draw" setup=[setupParallelPlots] begin
    @test transform_to_scale!([1.0, 2.0, 3.0]) ≈ [0.0, 0.5, 1.0]
    @test transform_to_scale!([2.0, 2.0]) ≈ [1.0, 1.0]

    data = [
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        [[1.5, 2.5, 3.5], [2.5, 3.5, 4.5]],
    ]
    obs = ["a", "b"]
    df = parallel_plot_df(data, obs; kinds = ["x", "y"], thinning = 1.0)
    @test nrow(df) == 6
    @test all(n in names(df) for n in ["a", "b", "kind", "id"])

    fig = create_parallel_plot(data, obs, ["x", "y"]; thinning = 1.0, sample_n = 4, legend = true)
    @test fig !== nothing

    @test_throws AssertionError parallel_plot_df(data, obs; thinning = 0.0)
    @test_throws AssertionError parallel_plot_df([[[1.0], [1.0, 2.0]]], obs)
    @test_throws AssertionError create_parallel_plot(data, obs, ["x", "y"]; order_vec = [1, 3])
end
