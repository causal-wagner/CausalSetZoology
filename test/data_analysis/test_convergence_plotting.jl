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

    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "plot_theme.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "dataloading.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "convergence_fitting.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "convergence_plotting.jl"))
end

@testitem "convergence_plotting: std-change and alpha/beta plots" setup=[setupConvergencePlotting] begin
    hists = [Dict(1 => 1.0, 2 => 2.0), Dict(1 => 2.0, 2 => 1.0), Dict(1 => 1.5, 2 => 1.5), Dict(1 => 2.5, 2 => 1.0)]
    fig1 = convergence_plots_std_change(hists; batchsize = 1, bin_average = 1)
    @test fig1 isa Figure

    X = [1.0 2.0; 2.0 2.5; 2.5 3.0; 3.0 3.5; 3.2 3.8]
    fig2 = plot_alpha_bins(X; batchsize = 1, legend = true, plot_mean = true)
    @test fig2 isa Figure

    fig3 = plot_beta_bins(X; batchsize = 1, legend = true, plot_mean = true)
    @test fig3 isa Figure

    fig4 = plot_alpha_bins(X; batchsize = 1, N0 = 2.0, bin_plot = 1)
    @test fig4 isa Figure
    fig5 = plot_beta_bins(X; batchsize = 1, N0 = 2.0, bin_plot = 1)
    @test fig5 isa Figure
    @test_throws AssertionError plot_alpha_bins(X; batchsize = 1, bin_plot = 99)
end
