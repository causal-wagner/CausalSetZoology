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

    obsval(x) = x isa Observables.Observable ? Observables.to_value(x) : x
    figure_axes(fig::Figure) = [b for b in fig.content if b isa Axis]
    figure_colorbars(fig::Figure) = [b for b in fig.content if b isa Colorbar]
    figure_legends(fig::Figure) = [b for b in fig.content if b isa Legend]
    nplots(ax::Axis) = length(ax.scene.plots)
end

@testitem "histogram_plotting: plain overload structure and metadata" setup=[setupHistogramPlotting] begin
    d_plain = [
        ([1.0, 2.0, 3.0], [0.1, 0.2, 0.1]),
        ([1.5, 2.2, 2.8], [0.1, 0.1, 0.2]),
    ]

    fig, ax = CausalSetZoology.plot_mean_histograms_with_std(
        d_plain;
        return_axis = true,
        plot_types = [:line, :scatter],
        hist_labels = ["A", "B"],
        xlabel = "bin",
        ylabel = "count",
        logscale_x = true,
        logscale_y = true,
    )

    @test fig isa Figure
    @test ax isa Axis
    @test string(obsval(ax.xlabel)) == "bin"
    @test string(obsval(ax.ylabel)) == "count"
    @test obsval(ax.xscale) === log10
    @test obsval(ax.yscale) === log10
    @test nplots(ax) >= 4
    @test length(figure_legends(fig)) >= 1
end

@testitem "histogram_plotting: plain overload validation errors" setup=[setupHistogramPlotting] begin
    d_plain = [
        ([1.0, 2.0, 3.0], [0.1, 0.2, 0.1]),
        ([1.5, 2.2, 2.8], [0.1, 0.1, 0.2]),
    ]

    @test_throws ErrorException CausalSetZoology.plot_mean_histograms_with_std(d_plain; plot_types = [:bad, :line])
    @test_throws ArgumentError CausalSetZoology.plot_mean_histograms_with_std(d_plain; hist_labels = ["only_one"])
    @test_throws ArgumentError CausalSetZoology.plot_mean_histograms_with_std(d_plain; plot_types = [:line])
end

@testitem "histogram_plotting: scalar overload includes colorbar and comp" setup=[setupHistogramPlotting] begin
    d_scalar = [
        (1.0, [1.0, 2.0], [0.1, 0.2]),
        (2.0, [2.0, 3.0], [0.2, 0.3]),
    ]

    fig, ax = CausalSetZoology.plot_mean_histograms_with_std(
        d_scalar;
        return_axis = true,
        colorbar_label = "s",
        colorbar_ticks = [(1.0, "low"), (2.0, "high")],
        comp = ([1.4, 2.6], [0.1, 0.1]),
    )

    @test fig isa Figure
    @test ax isa Axis
    @test length(figure_colorbars(fig)) >= 1
    @test nplots(ax) >= 5

    cb = first(figure_colorbars(fig))
    @test string(obsval(cb.label)) == "s"
end

@testitem "histogram_plotting: scalar overload option behavior" setup=[setupHistogramPlotting] begin
    d_scalar = [
        (1.0, [1.0, 2.0], [0.1, 0.2]),
        (2.0, [2.0, 3.0], [0.2, 0.3]),
    ]

    fig_std, ax_std = CausalSetZoology.plot_mean_histograms_with_std(d_scalar; return_axis = true, plot_std = true)
    fig_no_std, ax_no_std = CausalSetZoology.plot_mean_histograms_with_std(d_scalar; return_axis = true, plot_std = false)

    @test fig_std isa Figure
    @test fig_no_std isa Figure
    @test nplots(ax_std) > nplots(ax_no_std)
end

@testitem "histogram_plotting: plot_and_save_hists plain wrapper" setup=[setupHistogramPlotting] begin
    hplain = [[Dict(1 => 1.0, 2 => 2.0), Dict(1 => 2.0, 2 => 1.0)]]
    fig, ax = CausalSetZoology.plot_and_save_hists(hplain, "h_plain"; return_axis = true)

    @test fig isa Figure
    @test ax isa Axis
    @test obsval(ax.xscale) === log10
    @test obsval(ax.yscale) === log10
end

@testitem "histogram_plotting: plot_and_save_hists scalar wrapper" setup=[setupHistogramPlotting] begin
    hscalar = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
        [(Dict{Int64,Float64}(1 => 1.0, 2 => 2.0), 1.0), (Dict{Int64,Float64}(1 => 2.0, 2 => 1.0), 2.0)],
    ])

    fig, ax = CausalSetZoology.plot_and_save_hists(
        hscalar,
        "h_scalar";
        num_bins = 2,
        return_axis = true,
        colorbar_label = "scalar",
    )

    @test fig isa Figure
    @test ax isa Axis
    @test length(figure_colorbars(fig)) >= 1
end

@testitem "histogram_plotting: plot_and_save_vectors plain wrapper" setup=[setupHistogramPlotting] begin
    vplain = [[[1.0, 2.0], [2.0, 3.0]]]
    fig, ax = CausalSetZoology.plot_and_save_vectors(vplain, "v_plain"; return_axis = true)

    @test fig isa Figure
    @test ax isa Axis
    @test obsval(ax.xscale) === log10
    @test obsval(ax.yscale) === log10
end

@testitem "histogram_plotting: plot_and_save_vectors scalar wrapper" setup=[setupHistogramPlotting] begin
    vscalar = [[([1.0, 2.0], 1.0), ([2.0, 3.0], 2.0), ([1.5, 2.5], 1.0)]]
    fig, ax = CausalSetZoology.plot_and_save_vectors(
        vscalar,
        "v_scalar";
        num_bins = 2,
        return_axis = true,
        colorbar_label = "scalar",
    )

    @test fig isa Figure
    @test ax isa Axis
    @test length(figure_colorbars(fig)) >= 1
end

@testitem "histogram_plotting: plot_and_save_vectors invalid format throws" setup=[setupHistogramPlotting] begin
    @test_throws ErrorException CausalSetZoology.plot_and_save_vectors(Any[1, 2, 3], "bad")
end
