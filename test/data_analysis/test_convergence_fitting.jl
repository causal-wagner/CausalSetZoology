@testsnippet setupConvergenceFitting begin
    using Test
    using Statistics
    using Random
    using Optim

end

@testitem "convergence_fitting: sigma/mu evolution" setup=[setupConvergenceFitting] begin
    X = [1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 5.0]

    Ns, σ = CausalSetZoology.compute_sigma_evolution(X; batchsize = 1, bin_average = 1)
    @test Ns == [1, 2, 3, 4]
    @test size(σ) == (4, 2)

    Nsm, μ = CausalSetZoology.compute_mu_evolution(X; batchsize = 2, bin_average = 1)
    @test Nsm == [2, 4]
    @test size(μ) == (2, 2)
    @test μ[end, :] ≈ [2.5, 3.5]
end

@testitem "convergence_fitting: sigma infty/alpha fit" setup=[setupConvergenceFitting] begin
    ns = Float64.(1:10)
    σn = 1.0 .+ 2.0 .* ns .^ (-0.5)

    fs = CausalSetZoology.fit_sigma_infty_alpha(σn, ns; σinf_init = 1.0, α_init = 0.5, fix_sigma_inf = true)
    @test isfinite(fs.α)
    @test fs.σinf ≈ 1.0 atol = 1e-6

    fs_free = CausalSetZoology.fit_sigma_infty_alpha(σn, ns; α_init = 0.3, fix_sigma_inf = false)
    @test isfinite(fs_free.σinf)
    @test isfinite(fs_free.α)
end

@testitem "convergence_fitting: mu infty/beta fit" setup=[setupConvergenceFitting] begin
    ns = Float64.(1:10)
    μn = 3.0 .+ 1.5 .* ns .^ (-0.7)

    fm = CausalSetZoology.fit_mu_infty_beta(μn, ns; μinf_init = 3.0, β_init = 0.7, fix_sigma_inf = true)
    @test isfinite(fm.β)
    @test fm.μinf ≈ 3.0 atol = 1e-6

    fm_free = CausalSetZoology.fit_mu_infty_beta(μn, ns; β_init = 0.4, fix_sigma_inf = false)
    @test isfinite(fm_free.μinf)
    @test isfinite(fm_free.β)
end

@testitem "convergence_fitting: convergence wrappers" setup=[setupConvergenceFitting] begin
    X = [1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 5.0]

    sc = CausalSetZoology.fit_sigma_convergence(X; batchsize = 1)
    @test length(sc.Ns) == size(sc.σ, 1)
    @test length(sc.bin_fits) == size(sc.σ, 2)

    mc = CausalSetZoology.fit_mu_convergence(X; batchsize = 1)
    @test length(mc.Ns) == size(mc.μ, 1)
    @test length(mc.bin_fits) == size(mc.μ, 2)
end

@testitem "convergence_fitting: validation errors" setup=[setupConvergenceFitting] begin
    X = [1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 5.0]
    @test_throws AssertionError CausalSetZoology.compute_sigma_evolution(X; batchsize = 0)
    @test_throws AssertionError CausalSetZoology.compute_mu_evolution(X; bin_average = 0)
end
