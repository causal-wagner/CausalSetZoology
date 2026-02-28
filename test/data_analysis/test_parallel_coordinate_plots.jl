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

end

# Covers all normalization branches in `transform_to_scale!`.
@testitem "parallel_coordinate_plots: transform_to_scale branches" setup=[setupParallelPlots] begin
    # Generic min-max normalization branch.
    @test CausalSetZoology.transform_to_scale!([1.0, 2.0, 3.0]) ≈ [0.0, 0.5, 1.0]

    # Nearly-constant nonzero branch: divide by common value.
    @test CausalSetZoology.transform_to_scale!([2.0, 2.0]) ≈ [1.0, 1.0]

    # Nearly-constant near-zero branch: return input unchanged.
    z = [1e-4, 5e-4, 2e-4]
    @test CausalSetZoology.transform_to_scale!(copy(z)) == z

    @test_throws ArgumentError CausalSetZoology.transform_to_scale!(Float64[])
    @test_throws DomainError CausalSetZoology.transform_to_scale!([1.0, Inf])
end

# Verifies dataframe shape/content for explicit `kinds` and no thinning.
@testitem "parallel_coordinate_plots: parallel_plot_df basic with explicit kinds" setup=[setupParallelPlots] begin
    data = [
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        [[1.5, 2.5, 3.5], [2.5, 3.5, 4.5]],
    ]
    obs = ["a", "b"]

    df = CausalSetZoology.parallel_plot_df(data, obs; kinds = ["x", "y"], thinning = 1.0)

    # Two paths x three samples each.
    @test nrow(df) == 6
    @test names(df) == ["a", "b", "kind", "id"]
    @test df.kind == ["x", "x", "x", "y", "y", "y"]
    @test df.id == [1, 2, 3, 1, 2, 3]
    @test df.a ≈ [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]
    @test df.b ≈ [2.0, 3.0, 4.0, 2.5, 3.5, 4.5]
end

# Verifies default kinds and thinning step logic.
@testitem "parallel_coordinate_plots: parallel_plot_df defaults and thinning" setup=[setupParallelPlots] begin
    data = [
        [[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]],
        [[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
    ]
    obs = [:x, :y]

    # thinning=0.5 -> step = round(Int, 1/0.5) = 2 -> keep indices 1,3.
    df = CausalSetZoology.parallel_plot_df(data, obs; thinning = 0.5)

    @test nrow(df) == 4
    @test unique(df.kind) == ["set1", "set2"]
    @test df.id == [1, 2, 1, 2]
    @test df.x ≈ [10.0, 30.0, 5.0, 7.0]
    @test df.y ≈ [1.0, 3.0, 9.0, 11.0]
end

# Throws for invalid thinning and structural mismatch.
@testitem "parallel_coordinate_plots: parallel_plot_df throws" setup=[setupParallelPlots] begin
    data = [[[1.0, 2.0], [3.0, 4.0]]]
    obs = ["a", "b"]

    # thinning must satisfy 0 < thinning <= 1.
    @test_throws DomainError CausalSetZoology.parallel_plot_df(data, obs; thinning = 0.0)
    @test_throws DomainError CausalSetZoology.parallel_plot_df(data, obs; thinning = 1.1)

    # Observable/value-vector count must match.
    @test_throws ArgumentError CausalSetZoology.parallel_plot_df([[[1.0, 2.0]]], obs)

    # Each observable vector must have same number of samples.
    @test_throws ArgumentError CausalSetZoology.parallel_plot_df([[[1.0], [1.0, 2.0]]], obs)

    @test_throws ArgumentError CausalSetZoology.parallel_plot_df([], obs)
    @test_throws ArgumentError CausalSetZoology.parallel_plot_df(data, String[])
    @test_throws ArgumentError CausalSetZoology.parallel_plot_df(data, obs; kinds = String[])
end

# Basic draw smoke test with visible axis contract checks.
@testitem "parallel_coordinate_plots: create_parallel_plot basic draw and axis" setup=[setupParallelPlots] begin
    data = [
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        [[1.5, 2.5, 3.5], [2.5, 3.5, 4.5]],
    ]
    obs = ["a", "b"]
    figgrid = CausalSetZoology.create_parallel_plot(data, obs, ["x", "y"]; thinning = 1.0, legend = true)

    @test figgrid isa AlgebraOfGraphics.FigureGrid
    @test figgrid.figure !== nothing

    # Axis is configured to normalized y-range and observable x labels.
    axes = filter(x -> x isa CairoMakie.Axis, figgrid.figure.content)
    @test !isempty(axes)
    ax = first(axes)
    xticks = ax.xticks[]
    @test xticks[2] == obs
    @test string(ax.ylabel[]) == ""
end

# Covers choose/order/sample/color/save branches and legend disabled branch.
@testitem "parallel_coordinate_plots: create_parallel_plot options branches" setup=[setupParallelPlots] begin
    data = [
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
        [[2.0, 4.0, 6.0, 8.0], [1.0, 1.5, 2.0, 2.5]],
        [[3.0, 5.0, 7.0, 9.0], [0.5, 0.6, 0.7, 0.8]],
    ]
    obs = ["u", "v"]
    kinds = ["k1", "k2", "k3"]
    out = joinpath(mktempdir(), "parallel_plot.png")

    # Exercise integer color_vec mapping, order/filtering, sampling and save path.
    figgrid = CausalSetZoology.create_parallel_plot(
        data,
        obs,
        kinds;
        thinning = 1.0,
        legend = false,
        sample_n = 100,                 # larger than nrows -> internal min branch
        color_vec = [3, 1, 2],          # integer indexing branch
        order_vec = [3, 1, 2],          # reordering branch
        choose_kinds = [1, 3],          # filtering branch
        fig_path = out,                 # save branch
    )

    @test figgrid isa AlgebraOfGraphics.FigureGrid
    @test isfile(out)
end

# Throws for invalid `order_vec` indices.
@testitem "parallel_coordinate_plots: create_parallel_plot throws" setup=[setupParallelPlots] begin
    data = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 3.0], [4.0, 5.0]],
    ]
    obs = ["a", "b"]
    kinds = ["x", "y"]

    # Index 3 is invalid for two kinds.
    @test_throws ArgumentError CausalSetZoology.create_parallel_plot(
        data,
        obs,
        kinds;
        order_vec = [1, 3],
    )
    @test_throws ArgumentError CausalSetZoology.create_parallel_plot(data, obs, kinds; order_vec = [1, 1])
    @test_throws ArgumentError CausalSetZoology.create_parallel_plot(data, obs, kinds; choose_kinds = [3])
    @test_throws DomainError CausalSetZoology.create_parallel_plot(data, obs, kinds; sample_n = 0)
    @test_throws DomainError CausalSetZoology.create_parallel_plot(data, obs, kinds; color_transparency = 1.1)
    @test_throws ArgumentError CausalSetZoology.create_parallel_plot(data, obs, kinds; color_vec = [1])
end
