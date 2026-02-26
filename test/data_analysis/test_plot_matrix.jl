@testsnippet setupPlotMatrix begin
    using Test
    using Random
    using Statistics
    using CairoMakie
    using LaTeXStrings
    using PlotUtils
    using Observables
    using Colors
    using Printf


    fig_path(name::String) = joinpath(mktempdir(), string(name, ".png"))

end

@testitem "plot_matrix: matrix plot and input validation" setup=[setupPlotMatrix] begin
    h1 = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
        [(Dict{Int64,Float64}(1 => 1.0, 2 => 2.0), 1.0), (Dict{Int64,Float64}(1 => 2.0, 2 => 1.0), 2.0)],
    ])
    h2 = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
        [(Dict{Int64,Float64}(1 => 1.5, 2 => 2.5), 1.0), (Dict{Int64,Float64}(1 => 2.5, 2 => 1.5), 2.0)],
    ])
    v3 = [[([1.0, 2.0], 1.0), ([2.0, 3.0], 2.0)]]
    h4 = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
        [(Dict{Int64,Float64}(1 => 3.0, 2 => 1.0), 1.0), (Dict{Int64,Float64}(1 => 2.0, 2 => 2.0), 2.0)],
    ])

    fig = CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        (h1, h2, v3, h4),
        "matrix_test";
        xlim = [nothing, nothing, nothing, nothing],
        ylim = [nothing, nothing, nothing, nothing],
        xlabel = ["x1", "x2", "x3", "x4"],
        ylabel = ["y1", "y2", "y3", "y4"],
        num_bins = 2,
        logscale_x = false,
        logscale_y = false,
    )
    @test fig isa Figure

    @test_throws AssertionError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, "bad";
        xlim = [nothing],
        ylim = [nothing, nothing, nothing, nothing],
        xlabel = ["x", "x", "x", "x"],
        ylabel = ["y", "y", "y", "y"],
    )

    @test_throws ErrorException CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        (h1, h2, v3, h4),
        "matrix_bad_side";
        xlim = [nothing, nothing, nothing, nothing],
        ylim = [nothing, nothing, nothing, nothing],
        xlabel = ["x1", "x2", "x3", "x4"],
        ylabel = ["y1", "y2", "y3", "y4"],
        colorbar_side = :invalid,
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws AssertionError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["missing_path"], ["missing_path"], :scalar, "bad2";
        xlim = [nothing, nothing, nothing, nothing],
        ylim = [nothing, nothing, nothing, nothing],
        xlabel = ["x", "x", "x", "x"],
        ylabel = ["y", "y", "y", "y"],
        symmetric = true,
    )
end
