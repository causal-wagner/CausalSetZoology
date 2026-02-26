@testsnippet setupHistogramPlotting begin
    using Test
    using Statistics
    using Random
    using CairoMakie
    using LaTeXStrings
    using PlotUtils
    using Observables
    using Colors
    using Printf


    fig_path(name::String) = joinpath(mktempdir(), string(name, ".png"))

end

@testitem "histogram_plotting: all overloads" setup=[setupHistogramPlotting] begin
    d_plain = [([1.0, 2.0, 3.0], [0.1, 0.2, 0.1]), ([1.5, 2.2, 2.8], [0.1, 0.1, 0.2])]
    fig = CausalSetZoology.plot_mean_histograms_with_std(d_plain)
    @test fig isa Figure
    fig_ra, ax_ra = CausalSetZoology.plot_mean_histograms_with_std(d_plain; return_axis = true, plot_types = [:line, :scatter])
    @test fig_ra isa Figure
    @test ax_ra isa Axis

    d_scalar = [(1.0, [1.0, 2.0], [0.1, 0.2]), (2.0, [2.0, 3.0], [0.2, 0.3])]
    fig2 = CausalSetZoology.plot_mean_histograms_with_std(d_scalar; colorbar_label = "s")
    @test fig2 isa Figure
    @test_throws ErrorException CausalSetZoology.plot_mean_histograms_with_std(d_plain; plot_types = [:bad, :line])

    hplain = [[Dict(1 => 1.0, 2 => 2.0), Dict(1 => 2.0, 2 => 1.0)]]
    p1 = CausalSetZoology.plot_and_save_hists(hplain, "h_plain")
    @test p1 isa Figure

    hscalar = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
        [(Dict{Int64,Float64}(1 => 1.0, 2 => 2.0), 1.0), (Dict{Int64,Float64}(1 => 2.0, 2 => 1.0), 2.0)],
    ])
    p2 = CausalSetZoology.plot_and_save_hists(hscalar, "h_scalar"; num_bins = 2)
    @test p2 isa Figure

    vplain = [[[1.0, 2.0], [2.0, 3.0]]]
    @test_throws ErrorException CausalSetZoology.plot_and_save_vectors(vplain, "v_plain")

    vscalar = [[([1.0, 2.0], 1.0), ([2.0, 3.0], 2.0), ([1.5, 2.5], 1.0)]]
    p4 = CausalSetZoology.plot_and_save_vectors(vscalar, "v_scalar"; num_bins = 2)
    @test p4 isa Figure

    @test_throws ErrorException CausalSetZoology.plot_and_save_vectors(Any[1, 2, 3], "bad")
end
