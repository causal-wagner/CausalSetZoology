@testsnippet setupConvergencePlotting begin
    using Test
    using Statistics
    using Random
    using Optim
    using CairoMakie
    using ColorSchemes
    using Colors
    using Observables
    using LaTeXStrings
    using Printf

    obsval(x) = x isa Observables.Observable ? Observables.to_value(x) : x
    figure_axes(fig::Figure) = [b for b in fig.content if b isa Axis]
    figure_colorbars(fig::Figure) = [b for b in fig.content if b isa Colorbar]
    figure_legends(fig::Figure) = [b for b in fig.content if b isa Legend]
    nplots(ax::Axis) = length(ax.scene.plots)
end

@testitem "convergence_plotting: std-change structure and metadata" setup=[setupConvergencePlotting] begin
    hists = [
        Dict(1 => 1.0, 2 => 2.0),
        Dict(1 => 2.0, 2 => 1.0),
        Dict(1 => 1.5, 2 => 1.5),
        Dict(1 => 2.5, 2 => 1.0),
    ]

    fig = CausalSetZoology.convergence_plots_std_change(hists; batchsize = 1, bin_average = 1)
    @test fig isa Figure

    axes = figure_axes(fig)
    @test length(axes) == 2
    @test length(figure_colorbars(fig)) == 1

    ax_top = only(filter(ax -> string(obsval(ax.xlabel)) == "histogram bins", axes))
    ax_bottom = only(filter(ax -> string(obsval(ax.xlabel)) == "sample size", axes))

    @test string(obsval(ax_top.ylabel)) == "Δσₖ"
    @test obsval(ax_top.xscale) === identity
    @test obsval(ax_top.yscale) === log10
    @test nplots(ax_top) >= 3

    @test string(obsval(ax_bottom.ylabel)) == "⟨Δσ⟩"
    @test obsval(ax_bottom.xscale) === log10
    @test obsval(ax_bottom.yscale) === log10
    @test nplots(ax_bottom) >= 1
end

@testitem "convergence_plotting: std-change bin-averaged labeling" setup=[setupConvergencePlotting] begin
    hists = [
        Dict(1 => 1.0, 2 => 2.0, 3 => 1.0),
        Dict(1 => 1.5, 2 => 2.5, 3 => 1.2),
        Dict(1 => 2.0, 2 => 2.2, 3 => 1.1),
    ]

    fig = CausalSetZoology.convergence_plots_std_change(hists; batchsize = 1, bin_average = 2, xlabel = "k")
    ax_top = only(filter(ax -> string(obsval(ax.xlabel)) == "k", figure_axes(fig)))
    @test string(obsval(ax_top.ylabel)) == "Δσ (bin-averaged)"
end

@testitem "convergence_plotting: alpha bins structure" setup=[setupConvergencePlotting] begin
    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]

    fig = CausalSetZoology.plot_alpha_bins(X; batchsize = 1, legend = true, plot_mean = true)
    @test fig isa Figure

    axes = figure_axes(fig)
    @test length(axes) == 1
    @test length(figure_legends(fig)) >= 1

    ax = only(axes)
    @test string(obsval(ax.xlabel)) == "histogram bins"
    @test string(obsval(ax.ylabel)) == "α"
    @test nplots(ax) >= 3
end

@testitem "convergence_plotting: alpha plot_mean changes plotted content" setup=[setupConvergencePlotting] begin
    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]

    fig_no_mean = CausalSetZoology.plot_alpha_bins(X; batchsize = 1, plot_mean = false)
    fig_mean = CausalSetZoology.plot_alpha_bins(X; batchsize = 1, plot_mean = true)

    ax_no_mean = only(figure_axes(fig_no_mean))
    ax_mean = only(figure_axes(fig_mean))
    @test nplots(ax_mean) > nplots(ax_no_mean)
end

@testitem "convergence_plotting: alpha bin-specific overlay adds second axis" setup=[setupConvergencePlotting] begin
    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]

    fig = CausalSetZoology.plot_alpha_bins(X; batchsize = 1, N0 = 2.0, bin_plot = 1)
    axes = figure_axes(fig)
    @test length(axes) == 2

    overlay_ax = only(filter(ax -> string(obsval(ax.ylabel)) != "α", axes))
    @test string(obsval(overlay_ax.xlabel)) == "sample size N"
    @test string(obsval(overlay_ax.ylabel)) == "σ(N)"
    @test nplots(overlay_ax) >= 2
end

@testitem "convergence_plotting: beta bins structure" setup=[setupConvergencePlotting] begin
    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]

    fig = CausalSetZoology.plot_beta_bins(X; batchsize = 1, legend = true, plot_mean = true)
    @test fig isa Figure

    axes = figure_axes(fig)
    @test length(axes) == 1
    @test length(figure_legends(fig)) >= 1

    ax = only(axes)
    @test string(obsval(ax.xlabel)) == "histogram bins"
    @test string(obsval(ax.ylabel)) == "β"
    @test nplots(ax) >= 3
end

@testitem "convergence_plotting: beta plot_mean changes plotted content" setup=[setupConvergencePlotting] begin
    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]

    fig_no_mean = CausalSetZoology.plot_beta_bins(X; batchsize = 1, plot_mean = false)
    fig_mean = CausalSetZoology.plot_beta_bins(X; batchsize = 1, plot_mean = true)

    ax_no_mean = only(figure_axes(fig_no_mean))
    ax_mean = only(figure_axes(fig_mean))
    @test nplots(ax_mean) > nplots(ax_no_mean)
end

@testitem "convergence_plotting: beta bin-specific overlay adds second axis" setup=[setupConvergencePlotting] begin
    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]

    fig = CausalSetZoology.plot_beta_bins(X; batchsize = 1, N0 = 2.0, bin_plot = 1)
    axes = figure_axes(fig)
    @test length(axes) == 2

    overlay_ax = only(filter(ax -> string(obsval(ax.ylabel)) != "β", axes))
    @test occursin("sample size", string(obsval(overlay_ax.xlabel)))
    @test occursin("μ", string(obsval(overlay_ax.ylabel))) || occursin("\\mu", string(obsval(overlay_ax.ylabel)))
    @test nplots(overlay_ax) >= 2
end

@testitem "convergence_plotting: invalid bin_plot throws" setup=[setupConvergencePlotting] begin
    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]
    @test_throws BoundsError CausalSetZoology.plot_alpha_bins(X; batchsize = 1, bin_plot = 99)
    @test_throws BoundsError CausalSetZoology.plot_beta_bins(X; batchsize = 1, bin_plot = 99)
end

@testitem "convergence_plotting: domain errors for missing fits" setup=[setupConvergencePlotting] begin
    # With only two samples, no bin has enough points for convergence fits.
    X_short = [1.0 2.0; 2.0 3.0]
    @test_throws DomainError CausalSetZoology.plot_alpha_bins(X_short; batchsize = 1)
    @test_throws DomainError CausalSetZoology.plot_beta_bins(X_short; batchsize = 1)

    # Second bin becomes unfittable (NaNs from step 3 on), while first bin remains valid.
    # This isolates the "bin_plot has no valid fit" DomainError branch.
    X_partial = [1.0 10.0; 2.0 11.0; 3.0 NaN; 4.0 NaN; 5.0 NaN]
    @test_throws DomainError CausalSetZoology.plot_alpha_bins(X_partial; batchsize = 1, bin_plot = 2)
    @test_throws DomainError CausalSetZoology.plot_beta_bins(X_partial; batchsize = 1, bin_plot = 2)
end
