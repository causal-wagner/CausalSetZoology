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

end

@testitem "convergence_plotting: std-change and alpha/beta plots" setup=[setupConvergencePlotting] begin
    hists = [Dict(1 => 1.0, 2 => 2.0), Dict(1 => 2.0, 2 => 1.0), Dict(1 => 1.5, 2 => 1.5), Dict(1 => 2.5, 2 => 1.0)]
    fig1 = CausalSetZoology.convergence_plots_std_change(hists; batchsize = 1, bin_average = 1)
    @test fig1 isa Figure

    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]
    fig2 = CausalSetZoology.plot_alpha_bins(X; batchsize = 1, legend = true, plot_mean = true)
    @test fig2 isa Figure

    fig3 = CausalSetZoology.plot_beta_bins(X; batchsize = 1, legend = true, plot_mean = true)
    @test fig3 isa Figure

    fig4 = CausalSetZoology.plot_alpha_bins(X; batchsize = 1, N0 = 2.0, bin_plot = 1)
    @test fig4 isa Figure
    fig5 = CausalSetZoology.plot_beta_bins(X; batchsize = 1, N0 = 2.0, bin_plot = 1)
    @test fig5 isa Figure
    @test_throws AssertionError CausalSetZoology.plot_alpha_bins(X; batchsize = 1, bin_plot = 99)
end
