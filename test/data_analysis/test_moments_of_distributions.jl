@testsnippet setupMomentsOfDistributions begin
    using Test
    import CausalSetZoology
end

@testitem "moments_of_distributions: weighted histogram moments basic" setup=[setupMomentsOfDistributions] begin
    hist = Dict(1 => 1.0, 3 => 3.0)

    @test CausalSetZoology.weighted_hist_mean(hist) ≈ 2.5 atol = 1e-12
    @test CausalSetZoology.weighted_hist_mean(hist; bins = [10.0, 20.0]) ≈ 17.5 atol = 1e-12
    @test CausalSetZoology.weighted_hist_skew(hist) ≈ -1.1547005383792515 atol = 1e-12
    @test CausalSetZoology.weighted_hist_exkurt(hist) ≈ -0.6666666666666665 atol = 1e-12
end

@testitem "moments_of_distributions: weighted histogram moments undefined cases" setup=[setupMomentsOfDistributions] begin
    zero_hist = Dict(1 => 0.0, 2 => 0.0)
    point_mass = Dict(2 => 5.0)

    @test isnan(CausalSetZoology.weighted_hist_mean(zero_hist))
    @test isnan(CausalSetZoology.weighted_hist_skew(zero_hist))
    @test isnan(CausalSetZoology.weighted_hist_exkurt(zero_hist))
    @test isnan(CausalSetZoology.weighted_hist_skew(point_mass))
    @test isnan(CausalSetZoology.weighted_hist_exkurt(point_mass))
end

@testitem "moments_of_distributions: weighted histogram moments validation" setup=[setupMomentsOfDistributions] begin
    @test_throws DimensionMismatch CausalSetZoology.weighted_hist_mean(Dict(1 => 1.0, 2 => 2.0); bins = [1.0])
    @test_throws DomainError CausalSetZoology.weighted_hist_mean(Dict(1 => -1.0, 2 => 2.0))
    @test_throws DomainError CausalSetZoology.weighted_hist_skew(Dict(1 => Inf))
    @test_throws DomainError CausalSetZoology.weighted_hist_exkurt(Dict(1 => 1.0); bins = [NaN])
end

@testitem "moments_of_distributions: aggregate_hist_moment" setup=[setupMomentsOfDistributions] begin
    hists = [Dict(1 => 1.0, 2 => 1.0), Dict(1 => 1.0, 2 => 3.0)]
    out = CausalSetZoology.aggregate_hist_moment(hists, CausalSetZoology.weighted_hist_mean)

    @test out.mean ≈ 1.625 atol = 1e-12
    @test out.std ≈ 0.125 atol = 1e-12
    @test out.values ≈ [1.5, 1.75] atol = 1e-12

    @test_throws ArgumentError CausalSetZoology.aggregate_hist_moment(Any[], CausalSetZoology.weighted_hist_mean)
    @test_throws ArgumentError CausalSetZoology.aggregate_hist_moment([Dict(1 => 0.0)], CausalSetZoology.weighted_hist_mean)
end
