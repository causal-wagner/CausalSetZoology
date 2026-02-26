@testsnippet setupDistinguishability begin
    using Test
    using Random
    using Statistics
    using LinearAlgebra
    using Distributions
    using Distributed
    using ProgressMeter

    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "distinguishability.jl"))
end

@testitem "distinguishability: all wrappers and core routines" setup=[setupDistinguishability] begin
    @testset "basic helpers" begin
        @test relative_change(2.0, 4.0) ≈ 1 / 3
        @test hellinger_distance([1.0, 0.0], [1.0, 0.0]) ≈ 0.0 atol = 1e-12
        @test hellinger_distance([1.0, 0.0], [0.0, 1.0]) > 0.9

        pairs = Tuple{Vector{Float64},Real}[([1.0], 1.0), ([2.0], 2.0), ([3.0], 2.0)]
        b_exact = bin_scalar_pairs(pairs, nothing, nothing)
        @test length(b_exact) == 2
        @test b_exact[1][1] == 1.0
        @test length(b_exact[2][2]) == 2

        b_binned = bin_scalar_pairs(pairs, 2, [1.0, 1.5, 2.0])
        @test length(b_binned) == 2
        @test isempty(bin_scalar_pairs(Tuple{Vector{Float64},Real}[]))

        dense = densify_hists([Dict(0 => 1.0, 2 => 2.0), Dict(1 => 3.0)])
        @test size(dense) == (2, 3)
        @test dense[1, 1] == 1.0
        @test dense[1, 3] == 2.0
        @test dense[2, 2] == 3.0
    end

    @testset "normalize_hists" begin
        h = [[Dict(1 => 2, 2 => 2), Dict(1 => 1, 2 => 3)]]
        nprob = normalize_hists(h; normalization = :probability)
        @test sum(values(nprob[1][1])) ≈ 1.0
        @test sum(values(nprob[1][2])) ≈ 1.0

        nmax = normalize_hists(h; normalization = :max)
        @test maximum(values(nmax[1][1])) ≈ 1.0
        @test maximum(values(nmax[1][2])) ≈ 1.0
        nconst = normalize_hists(h; normalization = 2)
        @test nconst[1][1][1] ≈ 1.0

        hs = [[(Dict(1 => 2, 2 => 2), 1.0), (Dict(1 => 1, 2 => 3), 2.0)]]
        nscalar = normalize_hists(hs; normalization = :probability, num_bins = 2)
        @test length(nscalar) == 1
        @test length(nscalar[1]) == 2
        @test all(sum(values(d)) ≈ 1.0 for (d, _) in nscalar[1])

        @test_throws AssertionError normalize_hists([[Dict{Int,Int}()]]; normalization = :probability)
    end

    @testset "histogram_distinguishability" begin
        rng = Random.Xoshiro(123)

        vecs_a = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]
        vecs_b = [[0.0, 1.0], [0.1, 0.9], [0.05, 0.95]]
        vecs_c = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]

        d_same = histogram_distinguishability(vecs_a, vecs_c).D
        d_diff = histogram_distinguishability(vecs_a, vecs_b).D
        @test d_same ≈ 0.0 atol = 1e-12
        @test d_diff > 0.5

        d_mc = histogram_distinguishability(vecs_a, vecs_b, 150; rng = rng)
        @test d_mc.D > 0.3
        @test d_mc.std >= 0.0

        hists_a = [Dict(1 => 10, 2 => 0), Dict(1 => 9, 2 => 1)]
        hists_b = [Dict(1 => 0, 2 => 10), Dict(1 => 1, 2 => 9)]
        d_hist = histogram_distinguishability(hists_a, hists_b).D
        @test d_hist > 0.5
    end

    @testset "permutation distinguishability" begin
        rng = Random.Xoshiro(7)
        vecs_a = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05], [0.85, 0.15]]
        vecs_b = [[0.0, 1.0], [0.1, 0.9], [0.05, 0.95], [0.15, 0.85]]

        p1 = histogram_distinguishability_permutation(vecs_a, vecs_b; n_perm = 80, rng = rng)
        @test 0.0 <= p1.p_value <= 1.0
        @test isfinite(p1.z_emp)
        @test isfinite(p1.z_coll)
        @test p1.std_Ts >= 0.0

        p2 = histogram_distinguishability_permutation(vecs_a, vecs_b, 80; n_perm = 80, rng = rng)
        @test 0.0 <= p2.p_value <= 1.0
    end

    @testset "mahalanobis gap core" begin
        rng = Random.Xoshiro(99)
        b = [[0.0], [0.1], [-0.1], [0.2], [-0.2], [0.05], [-0.05], [0.15]]
        a = [[3.0], [2.9], [3.1], [3.2], [2.8], [3.05], [2.95], [3.15]]

        res = mahalanobis_gap_distinguishability(a, b; R = 80, rng = rng, q = 0.0, alpha = 0.05, regulator = 1e-8)
        @test isfinite(res.M_obs)
        @test isfinite(res.threshold)
        @test res.M_obs > res.threshold
        @test res.distinguishable

        res_sym = mahalanobis_gap_distinguishability(a, b; R = 80, rng = rng, symmetric = true, alpha = 0.05, regulator = 1e-8)
        @test res_sym.M_obs_sym !== nothing
        @test res_sym.threshold_sym !== nothing

        h_a = [Dict(1 => 3, 2 => 1), Dict(1 => 4, 2 => 2), Dict(1 => 2, 2 => 2), Dict(1 => 5, 2 => 1), Dict(1 => 4, 2 => 1), Dict(1 => 3, 2 => 2)]
        h_b = [Dict(1 => 0, 2 => 4), Dict(1 => 1, 2 => 3), Dict(1 => 0, 2 => 5), Dict(1 => 2, 2 => 2), Dict(1 => 1, 2 => 4), Dict(1 => 2, 2 => 3)]
        res_h = mahalanobis_gap_distinguishability(h_a, h_b; R = 30, rng = rng, regulator = 1e-8)
        @test isfinite(res_h.M_obs)
    end

    @testset "wrappers and validation" begin
        rng = Random.Xoshiro(5353)
        data = [Tuple{Vector{Float64},Real}[
            ([0.0], 1.0), ([0.1], 1.0), ([0.2], 1.0), ([0.3], 1.0),
            ([1.0], 2.0), ([1.1], 2.0), ([1.2], 2.0), ([1.3], 2.0),
        ]]
        ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]

        @test length(scalar_bin_distinguishability(data; num_bins = 2)) == 1
        @test length(scalar_bin_distinguishability(data, ref; num_bins = 2)) == 2
        @test length(scalar_bin_distinguishability_permutation(data; num_bins = 2, n_perm = 20, rng = rng)) == 1
        @test length(scalar_bin_distinguishability_permutation(data, ref; num_bins = 2, n_perm = 20, rng = rng)) == 2

        mr = scalar_bin_mahalanobis_gap_distinguishability(data, ref; num_bins = 2, R = 20, rng = rng, regulator = 1e-8)
        @test length(mr) == 2
        @test all(isfinite(x.M_obs) for x in mr)

        @test_throws AssertionError relative_change(0.0, 1.0)
        @test_throws AssertionError hellinger_distance([1.0], [1.0, 0.0])
        @test _summary_stat([1.0, 2.0, 3.0], 0.5) == 2.0

        b1, b2 = _random_split_equal([[0.0], [0.1], [0.2], [0.3]], Random.Xoshiro(22))
        @test length(b1) == 2
        @test length(b2) == 2
    end

    @testset "distributed path" begin
        file = abspath(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "distinguishability.jl"))
        n_before = Distributed.nworkers()
        added = Int[]
        try
            needed = max(2, n_before) - n_before
            if needed > 0
                added = Distributed.addprocs(needed)
            end
            @everywhere begin
                using Random, Statistics, LinearAlgebra, Distributions, Distributed, ProgressMeter
                import Random: randperm
            end
            @everywhere include($file)

            a = [[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7]]
            b = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]]
            rd = mahalanobis_gap_distinguishability(a, b; R = 25, rng = Random.Xoshiro(6060), num_workers = max(2, Distributed.nworkers()), regulator = 1e-8)
            @test isfinite(rd.M_obs)
        finally
            if !isempty(added)
                Distributed.rmprocs(added...)
            end
        end
    end
end
