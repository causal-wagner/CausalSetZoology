using Test
using Random
using Statistics
using LinearAlgebra
using Distributions
using Distributed

include(joinpath(@__DIR__, "..", "distinguishability.jl"))

@testset "distinguishability.jl" begin
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

        d_mc = histogram_distinguishability(vecs_a, vecs_b, 200; rng = rng)
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

        p1 = histogram_distinguishability_permutation(vecs_a, vecs_b; n_perm = 100, rng = rng)
        @test 0.0 <= p1.p_value <= 1.0
        @test isfinite(p1.z_emp)
        @test isfinite(p1.z_coll)
        @test p1.std_Ts >= 0.0

        p2 = histogram_distinguishability_permutation(vecs_a, vecs_b, 100; n_perm = 100, rng = rng)
        @test 0.0 <= p2.p_value <= 1.0
        @test isfinite(p2.z_emp)
        @test isfinite(p2.z_coll)
    end

    @testset "mahalanobis gap core" begin
        rng = Random.Xoshiro(99)
        b = [[0.0], [0.1], [-0.1], [0.2], [-0.2], [0.05], [-0.05], [0.15]]
        a = [[3.0], [2.9], [3.1], [3.2], [2.8], [3.05], [2.95], [3.15]]

        res = mahalanobis_gap_distinguishability(a, b; R = 120, rng = rng, q = 0.0, alpha = 0.05, regulator = 1e-8)
        @test haskey(res, :M_obs)
        @test haskey(res, :threshold)
        @test isfinite(res.M_obs)
        @test isfinite(res.threshold)
        @test res.M_obs > res.threshold
        @test res.distinguishable
        @test res.M_obs_sym === nothing
        @test res.threshold_sym === nothing

        res_sym = mahalanobis_gap_distinguishability(a, b; R = 120, rng = rng, symmetric = true, alpha = 0.05, regulator = 1e-8)
        @test res_sym.M_obs_sym !== nothing
        @test res_sym.threshold_sym !== nothing
        @test res_sym.M_obs_min !== nothing
        @test res_sym.threshold_max !== nothing

        h_a = [Dict(1 => 3, 2 => 1), Dict(1 => 4, 2 => 2), Dict(1 => 2, 2 => 2), Dict(1 => 5, 2 => 1), Dict(1 => 4, 2 => 1), Dict(1 => 3, 2 => 2)]
        h_b = [Dict(1 => 0, 2 => 4), Dict(1 => 1, 2 => 3), Dict(1 => 0, 2 => 5), Dict(1 => 2, 2 => 2), Dict(1 => 1, 2 => 4), Dict(1 => 2, 2 => 3)]
        res_h = mahalanobis_gap_distinguishability(h_a, h_b; R = 50, rng = rng, regulator = 1e-8)
        @test isfinite(res_h.M_obs)
    end

    @testset "mahalanobis stabilization edge cases" begin
        rng = Random.Xoshiro(321)
        a = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
        b_const = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        b_line = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]

        @test_throws ErrorException mahalanobis_gap_distinguishability(
            a, b_const; R = 20, rng = rng, regulator = 0.0, stabilization_method = :regularization
        )

        res_proj = mahalanobis_gap_distinguishability(
            a, b_line; R = 20, rng = rng, stabilization_method = :projection, projection_tolerance = 1e-14
        )
        @test isfinite(res_proj.M_obs)
        @test isfinite(res_proj.threshold)

        @test_throws AssertionError mahalanobis_gap_distinguishability(a, b_const; num_workers = 0)
    end

    @testset "scalar-bin mahalanobis wrappers" begin
        rng = Random.Xoshiro(2025)

        # (value, scalar) format
        data_vec = [Tuple{Vector{Float64},Real}[
            ([1.00], 1.0),
            ([1.10], 1.0),
            ([1.20], 1.0),
            ([1.30], 1.0),
            ([2.00], 2.0),
            ([2.10], 2.0),
            ([2.20], 2.0),
            ([2.30], 2.0),
        ]]
        ref_vec = [
            [0.95],
            [1.05],
            [1.15],
            [1.25],
            [1.35],
            [1.45],
        ]

        res_ref = scalar_bin_mahalanobis_gap_distinguishability(
            data_vec, ref_vec; num_bins = 2, R = 40, rng = rng, symmetric = false, num_workers = 1, regulator = 1e-8
        )
        @test length(res_ref) == 2
        @test all(haskey(r, :scalar) && haskey(r, :M_obs) && haskey(r, :threshold) for r in res_ref)
        @test all(isfinite(r.M_obs) for r in res_ref)

        res_pairs = scalar_bin_mahalanobis_gap_distinguishability(
            data_vec; num_bins = 2, R = 40, rng = rng, symmetric = false, num_workers = 1, regulator = 1e-8
        )
        @test length(res_pairs) == 1 # 2 bins -> 1 pair
        @test haskey(res_pairs[1], :s1) && haskey(res_pairs[1], :s2)

        # (scalar, value) format should be accepted too
        data_vec_flip = [Tuple{Real,Vector{Float64}}[
            (1.0, [1.00]),
            (1.0, [1.10]),
            (1.0, [1.20]),
            (1.0, [1.30]),
            (2.0, [2.00]),
            (2.0, [2.10]),
            (2.0, [2.20]),
            (2.0, [2.30]),
        ]]
        res_flip = scalar_bin_mahalanobis_gap_distinguishability(
            data_vec_flip, ref_vec; num_bins = 2, R = 30, rng = rng, symmetric = false, num_workers = 1, regulator = 1e-8
        )
        @test length(res_flip) == 2
        @test all(isfinite(r.M_obs) for r in res_flip)
    end

    @testset "scalar-bin D wrappers (all overloads)" begin
        rng = Random.Xoshiro(4242)
        data = [Tuple{Vector{Float64},Real}[
            ([0.0], 1.0), ([0.1], 1.0), ([0.2], 1.0), ([0.3], 1.0),
            ([1.0], 2.0), ([1.1], 2.0), ([1.2], 2.0), ([1.3], 2.0),
        ]]
        ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]

        r1 = scalar_bin_distinguishability(data; num_bins = 2)
        @test length(r1) == 1
        @test haskey(r1[1], :D)

        r2 = scalar_bin_distinguishability(data, 30; num_bins = 2, rng = rng)
        @test length(r2) == 1
        @test haskey(r2[1], :std)
        @test r2[1].std >= 0.0

        r3 = scalar_bin_distinguishability(data, ref; num_bins = 2)
        @test length(r3) == 2
        @test all(haskey(x, :scalar) && haskey(x, :D) for x in r3)

        r4 = scalar_bin_distinguishability(data, ref, 30; num_bins = 2, rng = rng)
        @test length(r4) == 2
        @test all(haskey(x, :std) && x.std >= 0.0 for x in r4)
    end

    @testset "scalar-bin permutation wrappers (all overloads)" begin
        rng = Random.Xoshiro(5353)
        data = [Tuple{Vector{Float64},Real}[
            ([0.0], 1.0), ([0.1], 1.0), ([0.2], 1.0), ([0.3], 1.0),
            ([1.0], 2.0), ([1.1], 2.0), ([1.2], 2.0), ([1.3], 2.0),
        ]]
        ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]

        p1 = scalar_bin_distinguishability_permutation(data; num_bins = 2, n_perm = 25, rng = rng)
        @test length(p1) == 1
        @test 0.0 <= p1[1].p_value <= 1.0

        p2 = scalar_bin_distinguishability_permutation(data, 30; num_bins = 2, n_perm = 25, rng = rng)
        @test length(p2) == 1
        @test 0.0 <= p2[1].p_value <= 1.0

        p3 = scalar_bin_distinguishability_permutation(data, ref; num_bins = 2, n_perm = 25, rng = rng)
        @test length(p3) == 2
        @test all(0.0 <= x.p_value <= 1.0 for x in p3)

        p4 = scalar_bin_distinguishability_permutation(data, ref, 30; num_bins = 2, n_perm = 25, rng = rng)
        @test length(p4) == 2
        @test all(isfinite(x.z_emp) && isfinite(x.z_coll) for x in p4)
    end

    @testset "validation and edge branches" begin
        rng = Random.Xoshiro(111)

        @test_throws AssertionError relative_change(0.0, 1.0)
        @test_throws AssertionError hellinger_distance([1.0], [1.0, 0.0])
        @test_throws AssertionError histogram_distinguishability(Vector{Vector{Float64}}(), [[1.0]])
        @test_throws AssertionError histogram_distinguishability([[1.0]], Vector{Vector{Float64}}())
        @test_throws AssertionError histogram_distinguishability([[1.0]], [[1.0]], 0; rng = rng)
        @test_throws AssertionError bin_scalar_pairs(Tuple{Vector{Float64},Real}[([1.0], 1.0)], 0)

        a = [[1.0], [2.0], [3.0], [4.0]]
        b = [[0.0], [0.1], [0.2], [0.3]]
        @test_throws AssertionError mahalanobis_gap_distinguishability(a, b; q = -0.1)
        @test_throws AssertionError mahalanobis_gap_distinguishability(a, b; q = 1.1)
        @test_throws AssertionError mahalanobis_gap_distinguishability(a, b; alpha = -0.1)
        @test_throws AssertionError mahalanobis_gap_distinguishability(a, b; alpha = 1.0)
        @test_throws AssertionError mahalanobis_gap_distinguishability(a, b; regulator = -1.0)
        @test_throws ErrorException mahalanobis_gap_distinguishability(a, b; stabilization_method = :unknown)
        @test_throws ErrorException mahalanobis_gap_distinguishability(Any[[1.0], Dict(1 => 1)], Any[[1.0], [2.0]])

        # q > 0 and alpha == 0 branch
        rq = mahalanobis_gap_distinguishability(a, b; q = 0.5, alpha = 0.0, R = 30, rng = rng, regulator = 1e-8)
        @test isfinite(rq.M_obs)
        @test isfinite(rq.threshold)
    end

    @testset "internal helpers" begin
        A, B = _prepare_vectors_for_mahalanobis([[1.0, 2.0], [1.0]], [[0.0, 0.0, 1.0], [0.0, 0.0]])
        @test length(A) == 2 && length(B) == 2
        @test all(length(v) == 3 for v in vcat(A, B))

        @test _summary_stat([1.0, 2.0, 3.0], 0.0) == 1.0
        @test _summary_stat([1.0, 2.0, 3.0], 0.5) == 2.0

        rng = Random.Xoshiro(22)
        b = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
        b1, b2 = _random_split_equal(b, rng)
        @test length(b1) == 3
        @test length(b2) == 3

        # projection failure path: all-eigenvalues-below-tolerance
        @test_throws ErrorException _fit_reference(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0;
            stabilization_method = :projection,
            projection_tolerance = 1e-10,
        )
    end

    @testset "deterministic RNG behavior" begin
        seed = 8888
        a = [[1.0], [1.1], [1.2], [1.3], [1.4], [1.5]]
        b = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]

        r1 = mahalanobis_gap_distinguishability(a, b; R = 80, rng = Random.Xoshiro(seed), regulator = 1e-8)
        r2 = mahalanobis_gap_distinguishability(a, b; R = 80, rng = Random.Xoshiro(seed), regulator = 1e-8)
        @test r1.M_obs == r2.M_obs
        @test r1.threshold == r2.threshold
        @test r1.z_emp == r2.z_emp

        data = [Tuple{Vector{Float64},Real}[
            ([1.0], 1.0), ([1.1], 1.0), ([1.2], 1.0), ([1.3], 1.0),
            ([2.0], 2.0), ([2.1], 2.0), ([2.2], 2.0), ([2.3], 2.0),
        ]]
        ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
        s1 = scalar_bin_mahalanobis_gap_distinguishability(data, ref; num_bins = 2, R = 30, rng = Random.Xoshiro(seed), regulator = 1e-8)
        s2 = scalar_bin_mahalanobis_gap_distinguishability(data, ref; num_bins = 2, R = 30, rng = Random.Xoshiro(seed), regulator = 1e-8)
        @test s1 == s2
    end

    @testset "distributed and progress paths" begin
        file = abspath(joinpath(@__DIR__, "..", "distinguishability.jl"))
        n_before = Distributed.nworkers()
        added = Int[]
        try
            needed = max(2, n_before) - n_before
            if needed > 0
                added = Distributed.addprocs(needed)
            end

            # preload workers so function stays on distributed path
            @everywhere begin
                using Random, Statistics, LinearAlgebra, Distributions, Distributed, ProgressMeter
                import Random: randperm
            end
            @everywhere include($file)

            a = [[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7]]
            b = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]]
            rng = Random.Xoshiro(6060)

            n_pre = Distributed.nworkers()
            rd = mahalanobis_gap_distinguishability(
                a, b; R = 40, rng = rng, num_workers = max(2, Distributed.nworkers()), regulator = 1e-8
            )
            @test isfinite(rd.M_obs)
            @test isfinite(rd.threshold)
            @test Distributed.nworkers() == n_pre

            data = [Tuple{Vector{Float64},Real}[
                ([1.0], 1.0), ([1.1], 1.0), ([1.2], 1.0), ([1.3], 1.0),
                ([2.0], 2.0), ([2.1], 2.0), ([2.2], 2.0), ([2.3], 2.0),
            ]]
            ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]]

            rp = scalar_bin_mahalanobis_gap_distinguishability(
                data;
                num_bins = 2,
                R = 20,
                rng = Random.Xoshiro(7070),
                num_workers = max(2, Distributed.nworkers()),
                regulator = 1e-8,
                progress = true,
                progress_step = 1,
            )
            @test length(rp) == 1
            @test isfinite(rp[1].M_obs)

            rr = scalar_bin_mahalanobis_gap_distinguishability(
                data,
                ref;
                num_bins = 2,
                R = 20,
                rng = Random.Xoshiro(8080),
                num_workers = max(2, Distributed.nworkers()),
                regulator = 1e-8,
                progress = true,
                progress_step = 1,
            )
            @test length(rr) == 2
            @test all(isfinite(x.M_obs) for x in rr)
            @test Distributed.nworkers() == n_pre

            # progress branches in serial mode (stable in tests)
            rs_prog = scalar_bin_mahalanobis_gap_distinguishability(
                data;
                num_bins = 2,
                R = 10,
                rng = Random.Xoshiro(9090),
                num_workers = 1,
                regulator = 1e-8,
                progress = true,
                progress_step = 1,
            )
            @test length(rs_prog) == 1

            rr_prog = scalar_bin_mahalanobis_gap_distinguishability(
                data,
                ref;
                num_bins = 2,
                R = 10,
                rng = Random.Xoshiro(9191),
                num_workers = 1,
                regulator = 1e-8,
                progress = true,
                progress_step = 1,
            )
            @test length(rr_prog) == 2

            # malformed scalar-pair should fail
            bad_pairs = [[([1.0], [2.0])]]
            @test_throws AssertionError scalar_bin_mahalanobis_gap_distinguishability(bad_pairs; R = 5)
        finally
            if !isempty(added)
                Distributed.rmprocs(added...)
            end
        end
    end
end
