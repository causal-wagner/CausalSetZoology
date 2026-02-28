@testsnippet setupGridFourierPlotting begin
    using Test
    using Random
    using CairoMakie
    using Observables
    using LaTeXStrings
    using Printf

    obsval(x) = x isa Observables.Observable ? Observables.to_value(x) : x
    figure_axes(fig::Figure) = [b for b in fig.content if b isa Axis]
    nplots(ax::Axis) = length(ax.scene.plots)
end

# Verifies direct plotting of precomputed grid points.
@testitem "grid_fourier_analysis_plotting: plot_grid_points basic" setup=[setupGridFourierPlotting] begin
    # Simple synthetic grid should render as one scatter plot.
    quad_grid = [(0.0, 0.0), (0.5, 1.0), (-0.5, -1.0)]
    fig, ax = CausalSetZoology.plot_grid_points(quad_grid; markersize = 5, magnification = 1.2)

    @test fig isa Figure
    @test ax isa Axis
    @test string(obsval(ax.xlabel)) == "x"
    @test string(obsval(ax.ylabel)) == "t"
    @test nplots(ax) >= 1
end

# Verifies save-path behavior of grid-point plotting helper.
@testitem "grid_fourier_analysis_plotting: plot_grid_points save" setup=[setupGridFourierPlotting] begin
    quad_grid = [(0.0, 0.0), (1.0, 1.0)]
    out = joinpath(mktempdir(), "grid_points.png")

    fig, ax = CausalSetZoology.plot_grid_points(quad_grid; fig_path = out)
    @test fig isa Figure
    @test ax isa Axis
    @test isfile(out)
end

# Verifies grid generation + plotting wrapper on valid inputs.
@testitem "grid_fourier_analysis_plotting: create_grid_and_plot basic" setup=[setupGridFourierPlotting] begin
    fig, ax = CausalSetZoology.create_grid_and_plot(
        16,
        "square",
        10.0;
        segment_ratio = 1.5,
        segment_angle = 55.0,
        shell_thickness = 0.1,
        magnification = 0.9,
    )

    @test fig isa Figure
    @test ax isa Axis
    @test string(obsval(ax.xlabel)) == "x"
    @test string(obsval(ax.ylabel)) == "t"
    @test nplots(ax) >= 1
end

# Validates that generation/plotting wrapper propagates input validation errors.
@testitem "grid_fourier_analysis_plotting: create_grid_and_plot validation" setup=[setupGridFourierPlotting] begin
    # Non-positive size should fail in underlying generation helper.
    @test_throws DomainError CausalSetZoology.create_grid_and_plot(0, "square", 0.0)

    # Invalid box ordering should fail.
    @test_throws ArgumentError CausalSetZoology.create_grid_and_plot(8, "square", 0.0; box = ((1.0, -1.0), (1.0, 1.0)))
end

# Verifies Fourier-spectrum plotting helper on precomputed spectrum data.
@testitem "grid_fourier_analysis_plotting: plot_fourier_grid_deviation basic" setup=[setupGridFourierPlotting] begin
    spec = (
        idx = 8,
        spectrum = [0.0, 1.0, 0.3, 0.2],
        freqs = [0.0, 0.25, 0.5, 0.75],
        keep = [2, 3],
        f_peak = 0.25,
        P_est = 4.0,
        peak_rows = [(f = 0.25, P = 4.0, A = 0.5)],
    )

    fig = CausalSetZoology.plot_fourier_grid_deviation(spec; magnification = 1.1, linewidth = 2, ylim = (0.0, 2.0))
    axes = figure_axes(fig)
    @test fig isa Figure
    @test length(axes) >= 1

    ax = first(axes)
    @test string(obsval(ax.xlabel)) == "frequency (cycles per bin)"
    @test nplots(ax) >= 1
end

# Verifies xtick customization (including rationals) and save-path behavior.
@testitem "grid_fourier_analysis_plotting: plot_fourier_grid_deviation xticks and save" setup=[setupGridFourierPlotting] begin
    spec = (
        idx = 8,
        spectrum = [0.0, 1.0, 0.3, 0.2],
        freqs = [0.0, 0.25, 0.5, 0.75],
        keep = [2, 3],
        f_peak = 0.25,
        P_est = 4.0,
        peak_rows = [(f = 0.25, P = 4.0, A = 0.5)],
    )
    out = joinpath(mktempdir(), "fourier_plot.png")

    fig = CausalSetZoology.plot_fourier_grid_deviation(
        spec;
        fig_path = out,
        xtick_fracs = [1 // 4, -1 // 3, 0.2],
        linewidth = 1.5,
    )

    axes = figure_axes(fig)
    @test fig isa Figure
    @test length(axes) >= 1
    ax = first(axes)

    # Custom ticks should be set and extra vline guides should be added.
    ticks = obsval(ax.xticks)
    @test length(ticks[1]) == 3
    @test nplots(ax) >= 2
    @test isfile(out)
end

# Verifies end-to-end wrapper: compute Fourier deviation then plot.
@testitem "grid_fourier_analysis_plotting: fourier_transform_grid_deviation basic" setup=[setupGridFourierPlotting] begin
    comp_hist = ones(60)
    comp_hist[30] = 0.0

    fig = CausalSetZoology.fourier_transform_grid_deviation(
        comp_hist,
        30,
        "square";
        P_max = 20.0,
        rng = Random.Xoshiro(7),
        segment_ratio = 1.1,
        segment_angle = 50.0,
        max_peak_order = 3,
        xtick_fracs = [1 // 4, 1 // 3],
        linewidth = 1.2,
    )

    @test fig isa Figure
    axes = figure_axes(fig)
    @test length(axes) >= 1
    @test nplots(first(axes)) >= 1
end

# Verifies save-path behavior for end-to-end Fourier plotting wrapper.
@testitem "grid_fourier_analysis_plotting: fourier_transform_grid_deviation save" setup=[setupGridFourierPlotting] begin
    comp_hist = ones(40)
    comp_hist[20] = 0.0
    out = joinpath(mktempdir(), "fourier_wrapper.png")

    fig = CausalSetZoology.fourier_transform_grid_deviation(
        comp_hist,
        20,
        "square";
        P_max = 15.0,
        rng = Random.Xoshiro(11),
        fig_path = out,
    )

    @test fig isa Figure
    @test isfile(out)
end

# Validates that end-to-end wrapper propagates compute-layer input errors.
@testitem "grid_fourier_analysis_plotting: fourier_transform_grid_deviation validation" setup=[setupGridFourierPlotting] begin
    # Missing sentinel in comp_hist should fail in compute helper.
    @test_throws ArgumentError CausalSetZoology.fourier_transform_grid_deviation([1.0, 1.0, 1.0], 12, "square")

    # Invalid size should fail.
    @test_throws DomainError CausalSetZoology.fourier_transform_grid_deviation([1.0, 1.0, 0.0], 0, "square")
end
