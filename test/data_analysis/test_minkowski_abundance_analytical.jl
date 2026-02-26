@testsnippet setupMinkowski begin
    using Test
    using SpecialFunctions

    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "minkowski_abundance_analytical.jl"))
end

@testitem "minkowski_abundance_analytical: formula branches" setup=[setupMinkowski] begin
    @test minkowski_interval_abundance_2d_inclusive_asymptotic(1, 10) == 10.0
    v = minkowski_interval_abundance_2d_inclusive_asymptotic(3, 100)
    @test isfinite(v)
    @test_throws AssertionError minkowski_interval_abundance_2d_inclusive_asymptotic(0, 10)
    @test_throws AssertionError minkowski_interval_abundance_2d_inclusive_asymptotic(2, 0)
end
