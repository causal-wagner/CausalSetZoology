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
    expected_σ = [
        0.0                0.0;
        0.5                0.5;
        sqrt(2.0 / 3.0)    sqrt(2.0 / 3.0);
        sqrt(1.25)         sqrt(1.25)
    ]
    @test σ ≈ expected_σ atol = 1e-12

    Nsm, μ = CausalSetZoology.compute_mu_evolution(X; batchsize = 1, bin_average = 1)
    @test Nsm == [1, 2, 3, 4]
    @test size(μ) == (4, 2)
    expected_μ = [
        1.0 2.0;
        1.5 2.5;
        2.0 3.0;
        2.5 3.5
    ]
    @test μ ≈ expected_μ atol = 1e-12
    @test μ[end, :] ≈ [2.5, 3.5]
end

@testitem "convergence_fitting: sigma infty/alpha fit" setup=[setupConvergenceFitting] begin
    ns = Float64.(1:10)
    σinf_true = 1.0
    α_true = 0.5
    A_true = 2.0
    σn = σinf_true .+ A_true .* ns .^ (-α_true)

    fs = CausalSetZoology.fit_sigma_infty_alpha(
        σn,
        ns;
        σinf_init = σinf_true,
        α_init = 0.9,
        fix_sigma_inf = true
    )
    @test fs.σinf ≈ σinf_true atol = 1e-10
    @test fs.α ≈ α_true atol = 1e-3
    @test fs.A ≈ A_true atol = 1e-3
    @test fs.objective ≤ 1e-10
    @test fs.σinf .+ fs.A .* ns .^ (-fs.α) ≈ σn atol = 1e-5

    fs_free = CausalSetZoology.fit_sigma_infty_alpha(
        σn,
        ns;
        σinf_init = 1.4,
        α_init = 0.8,
        bounds_σinf = (0.0, 3.0),
        fix_sigma_inf = false
    )
    @test fs_free.σinf ≈ σinf_true atol = 5e-3
    @test fs_free.α ≈ α_true atol = 5e-3
    @test fs_free.A ≈ A_true atol = 5e-3
    @test fs_free.objective ≤ 1e-7
    @test all(isapprox.(
        fs_free.σinf .+ fs_free.A .* ns .^ (-fs_free.α),
        σn;
        atol = 1e-4
    ))


end

@testitem "convergence_fitting: mu infty/beta fit" setup=[setupConvergenceFitting] begin
    ns = Float64.(1:10)
    μinf_true = 3.0
    β_true = 0.7
    B_true = 1.5
    μn = μinf_true .+ B_true .* ns .^ (-β_true)

    fm = CausalSetZoology.fit_mu_infty_beta(
        μn,
        ns;
        μinf_init = μinf_true,
        β_init = 0.3,
        fix_sigma_inf = true
    )
    @test fm.μinf ≈ μinf_true atol = 1e-10
    @test fm.β ≈ β_true atol = 1e-3
    @test fm.B ≈ B_true atol = 1e-3
    @test fm.objective ≤ 1e-7
    @test all(isapprox.(
        fm.μinf .+ fm.B .* ns .^ (-fm.β),
        μn;
        atol = 1e-4
    ))

    fm_free = CausalSetZoology.fit_mu_infty_beta(
        μn,
        ns;
        μinf_init = 2.2,
        β_init = 0.4,
        bounds_μinf = (1.0, 5.0),
        fix_sigma_inf = false
    )
    @test fm_free.μinf ≈ μinf_true atol = 5e-3
    @test fm_free.β ≈ β_true atol = 5e-3
    @test fm_free.B ≈ B_true atol = 5e-3
    @test fm_free.objective ≤ 1e-7
    @test fm_free.μinf .+ fm_free.B .* ns .^ (-fm_free.β) ≈ μn atol = 1e-4
end

@testitem "convergence_fitting: convergence wrappers" setup=[setupConvergenceFitting] begin
    X = [1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 5.0]

    sc = CausalSetZoology.fit_sigma_convergence(X; batchsize = 1)
    @test length(sc.Ns) == 4
    @test size(sc.σ) == (4, 2)
    @test length(sc.bin_fits) == 2
    @test length(sc.Ns) == size(sc.σ, 1)
    @test length(sc.bin_fits) == size(sc.σ, 2)

    mc = CausalSetZoology.fit_mu_convergence(X; batchsize = 1)
    @test length(mc.Ns) == 4
    @test size(mc.μ) == (4, 2)
    @test length(mc.bin_fits) == 2
    @test length(mc.Ns) == size(mc.μ, 1)
    @test length(mc.bin_fits) == size(mc.μ, 2)
end

@testitem "convergence_fitting: validation errors" setup=[setupConvergenceFitting] begin
    X = [1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 5.0]
    @test_throws ArgumentError CausalSetZoology.compute_sigma_evolution(X; batchsize = 0)
    @test_throws ArgumentError CausalSetZoology.compute_mu_evolution(X; bin_average = 0)
end
