@testsnippet setupPlotTheme begin
    using Test
    using CairoMakie
    using LaTeXStrings
    using Printf
    using Colors

    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "plot_theme.jl"))
end

@testitem "plot_theme: ticks and theme" setup=[setupPlotTheme] begin
    ticks, labels, kind = _logticks_internal(0.1, 100.0)
    @test !isempty(ticks)
    @test length(ticks) == length(labels)
    @test kind isa Symbol

    major, mlabels = logticks(0.1, 100.0)
    @test length(major) == length(mlabels)

    minors = logminorticks(0.1, 100.0)
    @test all(x -> x > 0, minors)

    mt = Makie.get_minor_tickvalues(logminorticks, log10, nothing, 0.1, 10.0)
    @test mt isa Vector{Float64}

    sz = apply_paper_theme!(; double_column = true, magnification = 1.2, logscale_x = true, logscale_y = true)
    @test length(sz) == 2
    @test sz[1] > 0
    @test sz[2] > 0

    @test_throws ArgumentError _logticks_internal(0.0, 10.0)
    @test_throws AssertionError apply_paper_theme!(; color_transparency = 2.0)
end
