@testsnippet setupPlotTheme begin
    using Test
    using CairoMakie
    using LaTeXStrings
    using Printf
    using Colors

end

@testitem "plot_theme: ticks and theme" setup=[setupPlotTheme] begin
    ticks, labels, kind = CausalSetZoology._logticks_internal(0.1, 100.0)
    @test !isempty(ticks)
    @test length(ticks) == length(labels)
    @test kind isa Symbol

    major, mlabels = CausalSetZoology.logticks(0.1, 100.0)
    @test length(major) == length(mlabels)

    minors = CausalSetZoology.logminorticks(0.1, 100.0)
    @test all(x -> x > 0, minors)

    mt = Makie.get_minor_tickvalues(logminorticks, log10, nothing, 0.1, 10.0)
    @test mt isa Vector{Float64}

    sz = CausalSetZoology.apply_paper_theme!(; double_column = true, magnification = 1.2, logscale_x = true, logscale_y = true)
    @test length(sz) == 2
    @test sz[1] > 0
    @test sz[2] > 0

    @test_throws ArgumentError CausalSetZoology._logticks_internal(0.0, 10.0)
    @test_throws AssertionError CausalSetZoology.apply_paper_theme!(; color_transparency = 2.0)
end
