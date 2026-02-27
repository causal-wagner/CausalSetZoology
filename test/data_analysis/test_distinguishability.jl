@testsnippet setupDistinguishability begin
    using Test
    using Random
    using Statistics
    using LinearAlgebra
    using Distributions
    using Distributed
    using ProgressMeter
end


@testitem "distinguishability: relative_change" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: relative_change behavior and output contract.
    @test CausalSetZoology.relative_change(2.0, 4.0) ≈ 1 / 3
    @test CausalSetZoology.relative_change(3.0, 3.0) == 0.0
end


@testitem "distinguishability: bin_scalar_pairs" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: bin_scalar_pairs behavior and output contract.
    pairs = Tuple{Vector{Float64},Real}[([1.0], 1.0), ([2.0], 2.0), ([3.0], 2.0)]

    exact_bins = CausalSetZoology.bin_scalar_pairs(pairs, nothing, nothing)
    @test exact_bins == [(1.0, [[1.0]]), (2.0, [[2.0], [3.0]])]

    binned = CausalSetZoology.bin_scalar_pairs(pairs, 2, [1.0, 1.5, 2.0])
    @test binned == [(1.0, [[1.0]]), (2.0, [[2.0], [3.0]])]

    # midpoint mode when requested bins differ from number of distinct labels
    midpoint_pairs = Tuple{Vector{Float64},Real}[([1.0], 1.0), ([2.0], 2.0), ([3.0], 3.0)]
    midpoint_binned = CausalSetZoology.bin_scalar_pairs(midpoint_pairs, 2, [1.0, 2.0, 3.0])
    @test midpoint_binned == [(1.5, [[1.0]]), (2.5, [[2.0], [3.0]])]

    @test isempty(CausalSetZoology.bin_scalar_pairs(Tuple{Vector{Float64},Real}[]))
end


@testitem "distinguishability helpers: typed scalar pairs" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: typed scalar pairs behavior and output contract.
    pairs_vs = Any[(Float64[1.0], 1.0), (Float64[2.0], 2.0)]
    out_vs = CausalSetZoology._typed_scalar_pairs(pairs_vs)
    @test out_vs == Tuple{Vector{Float64},Float64}[(Float64[1.0], 1.0), (Float64[2.0], 2.0)]

    pairs_sv = Any[(1.0, Float64[1.0]), (2.0, Float64[2.0])]
    @test_throws TypeError CausalSetZoology._typed_scalar_pairs(pairs_sv)
end


@testitem "distinguishability helpers: typed scalar pairs validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: typed scalar pairs validation behavior and output contract.
    bad_slot = Any[("x", "y"), ("u", "v")]
    bad_scalar = Any[(Float64[0.0], 1.0), (Float64[1.0], "bad")]
    @test_throws TypeError CausalSetZoology._typed_scalar_pairs(bad_slot)
    @test_throws TypeError CausalSetZoology._typed_scalar_pairs(bad_scalar)
end


@testitem "distinguishability helpers: prepare scalar bin context" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: prepare scalar bin context behavior and output contract.
    data = [Any[
        (Float64[0.0], 1.0), (Float64[0.1], 1.0),
        (Float64[1.0], 2.0), (Float64[1.1], 2.0),
    ]]
    ctx = CausalSetZoology._prepare_scalar_bin_context(data, "test"; num_bins = 2)
    @test length(ctx.pairs) == 4
    @test length(ctx.bins) == 2
    @test [b[1] for b in ctx.bins] ≈ [1.0, 2.0]
    @test length.(last.(ctx.bins)) == [2, 2]

    # midpoint mode when num_bins != n_unique_scalars
    data_mid = [Any[
        (Float64[0.0], 1.0), (Float64[1.0], 2.0), (Float64[2.0], 3.0),
    ]]
    ctx_mid = CausalSetZoology._prepare_scalar_bin_context(data_mid, "test"; num_bins = 2)
    @test [b[1] for b in ctx_mid.bins] ≈ [1.5, 2.5]
end


@testitem "distinguishability helpers: prepare scalar bin context validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: prepare scalar bin context validation behavior and output contract.
    data = [Any[(Float64[0.0], 1.0)]]
    @test_throws DimensionMismatch CausalSetZoology._prepare_scalar_bin_context(vcat(data, data), "ctx")
    @test_throws ArgumentError CausalSetZoology._prepare_scalar_bin_context([Any[]], "ctx")
    @test_throws ArgumentError CausalSetZoology._prepare_scalar_bin_context(data, "ctx"; ref = Any[])
    @test_throws DomainError CausalSetZoology._prepare_scalar_bin_context(data, "ctx"; num_bins = 0)
end


@testitem "distinguishability helpers: map scalar-bin pairs" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: map scalar-bin pairs behavior and output contract.
    bins = [(1.0, [Float64[0.0], Float64[0.1]]), (2.0, [Float64[1.0]]), (3.0, [Float64[2.0]])]
    compute = (s1, v1, s2, v2) -> (s1 = s1, s2 = s2, n1 = length(v1), n2 = length(v2))

    out_a = CausalSetZoology._map_scalar_bin_pairs(compute, bins)
    @test length(out_a) == 3
    @test out_a[1] == (s1 = 1.0, s2 = 2.0, n1 = 2, n2 = 1)
    @test out_a[2] == (s1 = 1.0, s2 = 3.0, n1 = 2, n2 = 1)
    @test out_a[3] == (s1 = 2.0, s2 = 3.0, n1 = 1, n2 = 1)
end


@testitem "distinguishability helpers: map scalar-bin reference" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: map scalar-bin reference behavior and output contract.
    bins = [(1.0, [Float64[0.0], Float64[0.1]]), (2.0, [Float64[1.0]])]
    compute = (s, v) -> (scalar = s, n = length(v))

    out_a = CausalSetZoology._map_scalar_bin_reference(compute, bins)
    @test out_a == [(scalar = 1.0, n = 2), (scalar = 2.0, n = 1)]
end


@testitem "distinguishability: scalar_bin_distinguishability wrappers" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: scalar_bin_distinguishability wrappers behavior and output contract.
    rng = Random.Xoshiro(5353)
    data = [Tuple{Vector{Float64},Real}[
        ([0.0], 1.0), ([0.1], 1.0), ([0.2], 1.0), ([0.3], 1.0),
        ([1.0], 2.0), ([1.1], 2.0), ([1.2], 2.0), ([1.3], 2.0),
    ]]
    ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]

    r_pair = CausalSetZoology.scalar_bin_distinguishability(data; num_bins = 2)
    @test length(r_pair) == 1
    @test keys(r_pair[1]) == (:s1, :s2, :rel_change, :D)
    @test 0.0 <= r_pair[1].D <= 1.0

    r_ref = CausalSetZoology.scalar_bin_distinguishability(data, ref; num_bins = 2)
    @test length(r_ref) == 2
    @test all(keys(x) == (:scalar, :D) for x in r_ref)
    @test [x.scalar for x in r_ref] ≈ [1.0, 2.0]
    @test all(0.0 <= x.D <= 1.0 for x in r_ref)

    r_pair_mc = CausalSetZoology.scalar_bin_distinguishability(data, 40; num_bins = 2, rng = rng)
    @test length(r_pair_mc) == 1
    @test keys(r_pair_mc[1]) == (:s1, :s2, :rel_change, :D, :std)
    @test 0.0 <= r_pair_mc[1].D <= 1.0
    @test r_pair_mc[1].std >= 0.0

    r_ref_mc = CausalSetZoology.scalar_bin_distinguishability(data, ref, 40; num_bins = 2, rng = rng)
    @test length(r_ref_mc) == 2
    @test all(keys(x) == (:scalar, :D, :std) for x in r_ref_mc)
    @test [x.scalar for x in r_ref_mc] ≈ [1.0, 2.0]
    @test all(x.std >= 0.0 for x in r_ref_mc)
end


@testitem "distinguishability: scalar_bin_distinguishability_permutation wrappers" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: scalar_bin_distinguishability_permutation wrappers behavior and output contract.
    rng = Random.Xoshiro(1234)
    data = [Tuple{Vector{Float64},Real}[
        ([0.0], 1.0), ([0.1], 1.0), ([0.2], 1.0), ([0.3], 1.0),
        ([1.0], 2.0), ([1.1], 2.0), ([1.2], 2.0), ([1.3], 2.0),
    ]]
    ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]

    p_pair = CausalSetZoology.scalar_bin_distinguishability_permutation(data; num_bins = 2, n_perm = 20, rng = rng)
    @test length(p_pair) == 1
    @test keys(p_pair[1]) == (:s1, :s2, :rel_change, :D_obs, :p_value, :z_emp, :z_coll, :std_Ts)
    @test 0.0 <= p_pair[1].D_obs <= 1.0
    @test 0.0 <= p_pair[1].p_value <= 1.0

    p_pair_mc = CausalSetZoology.scalar_bin_distinguishability_permutation(data, 30; num_bins = 2, n_perm = 20, rng = rng)
    @test length(p_pair_mc) == 1
    @test keys(p_pair_mc[1]) == (:s1, :s2, :rel_change, :D_obs, :p_value, :z_emp, :z_coll, :std_Ts)
    @test 0.0 <= p_pair_mc[1].D_obs <= 1.0
    @test 0.0 <= p_pair_mc[1].p_value <= 1.0

    p_ref = CausalSetZoology.scalar_bin_distinguishability_permutation(data, ref; num_bins = 2, n_perm = 20, rng = rng)
    @test length(p_ref) == 2
    @test all(keys(x) == (:scalar, :D_obs, :p_value, :z_emp, :z_coll, :std_Ts) for x in p_ref)
    @test [x.scalar for x in p_ref] ≈ [1.0, 2.0]
    @test all(0.0 <= x.p_value <= 1.0 for x in p_ref)

    p_ref_mc = CausalSetZoology.scalar_bin_distinguishability_permutation(data, ref, 30; num_bins = 2, n_perm = 20, rng = rng)
    @test length(p_ref_mc) == 2
    @test all(keys(x) == (:scalar, :D_obs, :p_value, :z_emp, :z_coll, :std_Ts) for x in p_ref_mc)
    @test [x.scalar for x in p_ref_mc] ≈ [1.0, 2.0]
    @test all(0.0 <= x.p_value <= 1.0 for x in p_ref_mc)
end


@testitem "distinguishability: scalar_bin_distinguishability validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: scalar_bin_distinguishability validation behavior and output contract.
    rng = Random.Xoshiro(1234)
    data = [Tuple{Vector{Float64},Real}[([0.0], 1.0), ([1.0], 2.0)]]
    ref_empty = Vector{Vector{Float64}}()
    @test_throws DimensionMismatch CausalSetZoology.scalar_bin_distinguishability(vcat(data, data); num_bins = 2)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distinguishability([Tuple{Vector{Float64},Real}[]]; num_bins = 2)
    @test_throws DomainError CausalSetZoology.scalar_bin_distinguishability(data; num_bins = 0)
end

@testitem "distinguishability: scalar_bin_distinguishability reference validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: scalar_bin_distinguishability reference validation behavior and output contract.
    rng = Random.Xoshiro(1234)
    data = [Tuple{Vector{Float64},Real}[([0.0], 1.0), ([1.0], 2.0)]]
    ref_empty = Vector{Vector{Float64}}()
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distinguishability(data, Vector{Vector{Float64}}(); num_bins = 2)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distinguishability(data, ref_empty; num_bins = 2)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distinguishability(data, ref_empty, 10; num_bins = 2, rng = rng)
end

@testitem "distinguishability: scalar_bin_distinguishability_permutation validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: scalar_bin_distinguishability_permutation validation behavior and output contract.
    rng = Random.Xoshiro(1234)
    data = [Tuple{Vector{Float64},Real}[([0.0], 1.0), ([1.0], 2.0)]]
    ref_empty = Vector{Vector{Float64}}()
    @test_throws DimensionMismatch CausalSetZoology.scalar_bin_distinguishability_permutation(vcat(data, data); num_bins = 2, n_perm = 10, rng = rng)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distinguishability_permutation([Tuple{Vector{Float64},Real}[]]; num_bins = 2, n_perm = 10, rng = rng)
    @test_throws DomainError CausalSetZoology.scalar_bin_distinguishability_permutation(data; num_bins = 0, n_perm = 10, rng = rng)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distinguishability_permutation(data, ref_empty; num_bins = 2, n_perm = 10, rng = rng)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distinguishability_permutation(data, ref_empty, 10; num_bins = 2, n_perm = 10, rng = rng)
end


@testitem "distinguishability: hellinger_distance" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: hellinger_distance behavior and output contract.
    @test CausalSetZoology.hellinger_distance([1.0, 0.0], [1.0, 0.0]) ≈ 0.0 atol = 1e-12
    @test CausalSetZoology.hellinger_distance([1.0, 0.0], [0.0, 1.0]) ≈ 1.0 atol = 1e-12
end


@testitem "distinguishability helpers: prepare vectors for distance" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: prepare vectors for distance behavior and output contract.
    a = [[1.0, 0.0], [0.5]]
    b = [[0.2], [0.0, 0.0, 0.1]]
    A, B = CausalSetZoology._prepare_vectors_for_distance(a, b)
    @test A == [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    @test B == [[0.2, 0.0, 0.0], [0.0, 0.0, 0.1]]

    A2, B2 = CausalSetZoology._prepare_vectors_for_distance([[0.0], [0.0, 0.0]], [[0.0]])
    @test all(length(v) == 2 for v in vcat(A2, B2))
end


@testitem "distinguishability helpers: distance matrix exact" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: distance matrix exact behavior and output contract.
    vecs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    D = CausalSetZoology._distance_matrix_exact(vecs)
    @test size(D) == (3, 3)
    @test D ≈ transpose(D) atol = 1e-12
    @test all(iszero, diag(D))
    @test D[1, 2] ≈ 1.0 atol = 1e-12
    @test D[1, 3] ≈ CausalSetZoology.hellinger_distance(vecs[1], vecs[3]) atol = 1e-12
end


@testitem "distinguishability: basic helper validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: basic helper validation behavior and output contract.
    pairs = Tuple{Vector{Float64},Real}[([1.0], 1.0), ([2.0], 2.0)]
    @test_throws DomainError CausalSetZoology.relative_change(0.0, 1.0)
    @test_throws DomainError CausalSetZoology.bin_scalar_pairs(pairs, 0, nothing)
    @test_throws DimensionMismatch CausalSetZoology.hellinger_distance([1.0], [1.0, 0.0])
end


@testitem "distinguishability: histogram_distinguishability vectors" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: histogram_distinguishability vectors behavior and output contract.
    rng = Random.Xoshiro(123)
    a = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]
    b = [[0.0, 1.0], [0.1, 0.9], [0.05, 0.95]]
    c = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]

    d_same = CausalSetZoology.histogram_distinguishability(a, c)
    d_diff = CausalSetZoology.histogram_distinguishability(a, b)
    @test d_same.D ≈ 0.0 atol = 1e-12
    @test d_diff.D > 0.9

    d_mc = CausalSetZoology.histogram_distinguishability(a, b, 150; rng = rng)
    @test d_mc.D > .9
    @test d_mc.std >= 0.0
end


@testitem "distinguishability: histogram_distinguishability histograms" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: histogram_distinguishability histograms behavior and output contract.
    h_a = [Dict(1 => 10, 2 => 0), Dict(1 => 9, 2 => 1)]
    h_b = [Dict(1 => 0, 2 => 10), Dict(1 => 1, 2 => 9)]

    d = CausalSetZoology.histogram_distinguishability(h_a, h_b)
    @test d.D > 0.9

    d_mc = CausalSetZoology.histogram_distinguishability(h_a, h_b, 100; rng = Random.Xoshiro(42))
    @test d_mc.D > 0.9
    @test d_mc.std >= 0.0
end


@testitem "distinguishability: histogram_distinguishability validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: histogram_distinguishability validation behavior and output contract.
    a = [[1.0, 0.0], [0.9, 0.1]]
    b = [[0.0, 1.0], [0.1, 0.9]]
    h_b = [Dict(1 => 0, 2 => 10), Dict(1 => 1, 2 => 9)]
    @test_throws DomainError CausalSetZoology.histogram_distinguishability(a, b, 0)
    @test_throws ArgumentError CausalSetZoology.histogram_distinguishability(Vector{Vector{Float64}}(), b)
    @test_throws ArgumentError CausalSetZoology.histogram_distinguishability(Dict{Int,Float64}[], h_b)
end


@testitem "distinguishability: histogram permutation vectors" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: histogram permutation vectors behavior and output contract.
    rng = Random.Xoshiro(7)
    a = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05], [0.85, 0.15]]
    b = [[0.0, 1.0], [0.1, 0.9], [0.05, 0.95], [0.15, 0.85]]

    p1 = CausalSetZoology.histogram_distinguishability_permutation(a, b; n_perm = 80, rng = rng)
    @test 0.0 <= p1.D_obs <= 1.0
    @test p1.D_obs > 0.5
    @test 0.0 <= p1.p_value <= 1.0
    @test isfinite(p1.D_obs)
    @test isfinite(p1.z_emp)
    @test isfinite(p1.z_coll)
    @test p1.std_Ts >= 0.0

    p2 = CausalSetZoology.histogram_distinguishability_permutation(a, b, 80; n_perm = 80, rng = rng)
    @test 0.0 <= p2.D_obs <= 1.0
    @test p2.D_obs > 0.5
    @test 0.0 <= p2.p_value <= 1.0
    @test isfinite(p2.D_obs)
end


@testitem "distinguishability: histogram permutation histograms" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: histogram permutation histograms behavior and output contract.
    h_a = [Dict(1 => 10, 2 => 0), Dict(1 => 9, 2 => 1), Dict(1 => 8, 2 => 2)]
    h_b = [Dict(1 => 0, 2 => 10), Dict(1 => 1, 2 => 9), Dict(1 => 2, 2 => 8)]

    p = CausalSetZoology.histogram_distinguishability_permutation(h_a, h_b; n_perm = 40, rng = Random.Xoshiro(11))
    @test 0.0 <= p.D_obs <= 1.0
    @test p.D_obs > 0.5
    @test 0.0 <= p.p_value <= 1.0
    @test isfinite(p.D_obs)
end


@testitem "distinguishability: histogram permutation validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: histogram permutation validation behavior and output contract.
    rng = Random.Xoshiro(7)
    a = [[1.0, 0.0], [0.9, 0.1]]
    b = [[0.0, 1.0], [0.1, 0.9]]
    @test_throws DomainError CausalSetZoology.histogram_distinguishability_permutation(a, b, 0; n_perm = 20, rng = rng)
    @test_throws ArgumentError CausalSetZoology.histogram_distinguishability_permutation(Vector{Vector{Float64}}(), b; n_perm = 20, rng = rng)
end


@testitem "distinguishability: mahalanobis core histogram overload" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis core histogram overload behavior and output contract.
    h_a = [Dict(1 => 3, 2 => 1), Dict(1 => 4, 2 => 2), Dict(1 => 2, 2 => 2), Dict(1 => 5, 2 => 1), Dict(1 => 4, 2 => 1), Dict(1 => 3, 2 => 2)]
    h_b = [Dict(1 => 0, 2 => 4), Dict(1 => 1, 2 => 3), Dict(1 => 0, 2 => 5), Dict(1 => 2, 2 => 2), Dict(1 => 1, 2 => 4), Dict(1 => 2, 2 => 3)]
    res_h = CausalSetZoology.mahalanobis_gap_distinguishability(h_a, h_b; R = 30, rng = Random.Xoshiro(99), regulator = 1e-8)
    @test isfinite(res_h.M_obs)
end


@testitem "distinguishability: _prepare_vectors_for_mahalanobis" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: _prepare_vectors_for_mahalanobis behavior and output contract.
    A, B = CausalSetZoology._prepare_vectors_for_mahalanobis([[1.0, 0.0], [0.5]], [[0.2, 0.0], [0.1, 0.0, 0.0]])
    @test all(length(v) == 1 for v in vcat(A, B))
    @test A == [[1.0], [0.5]]
    @test B == [[0.2], [0.1]]

    A2, B2 = CausalSetZoology._prepare_vectors_for_mahalanobis([[0.0, 0.0], [0.0]], [[0.0], [0.0, 0.0]])
    @test all(length(v) == 2 for v in vcat(A2, B2))
    @test all(all(==(0.0), v) for v in vcat(A2, B2))
end


@testitem "distinguishability: _fit_reference" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: _fit_reference behavior and output contract.
    Bref = [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8]]
    mu, inv_mul = CausalSetZoology._fit_reference(Bref, 1e-6)
    @test length(mu) == 2
    @test mu ≈ [0.2, 0.4] atol = 1e-12
    d = [1.0, -1.0]
    y = inv_mul(d)
    @test length(y) == 2
    @test all(isfinite, y)
    X = reduce(vcat, permutedims.(Bref))
    Σ = Statistics.cov(X; dims = 1)
    y_expected = (Σ + 1e-6 * LinearAlgebra.I) \ d
    @test y ≈ y_expected atol = 1e-6

    mu_p, inv_mul_p = CausalSetZoology._fit_reference(
        Bref,
        0.0;
        stabilization_method = :projection,
        projection_tolerance = 1e-12,
    )
    @test mu_p ≈ [0.2, 0.4] atol = 1e-12
    d_p = [1.0, -1.0]
    y_p = inv_mul_p(d_p)
    @test length(y_p) == 2
    @test all(isfinite, y_p)
    eig = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Σ))
    keep = eig.values .> 1e-12
    V = eig.vectors[:, keep]
    Λinv = LinearAlgebra.Diagonal(1.0 ./ eig.values[keep])
    Σ_pinv = V * Λinv * V'
    y_p_expected = Σ_pinv * d_p
    @test y_p ≈ y_p_expected atol = 1e-15

end


@testitem "distinguishability: _mahal_sigmas" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: _mahal_sigmas behavior and output contract.
    X = [[1.0, 2.0], [3.0, 4.0]]
    mu = [1.0, 1.0]
    inv_mul_id = d -> d
    sig = CausalSetZoology._mahal_sigmas(X, mu, inv_mul_id)
    @test sig ≈ [1.0, sqrt(13.0)] atol = 1e-12

    calls = Ref(0)
    inv_mul_diag = d -> begin
        calls[] += 1
        [4.0 * d[1], 9.0 * d[2]]
    end
    Xb = [[2.0, 1.0], [0.0, 3.0], [1.0, -1.0]]
    mub = [1.0, 1.0]
    sig_b = CausalSetZoology._mahal_sigmas(Xb, mub, inv_mul_diag)
    expected_b = [2.0, sqrt(40.0), 6.0]
    @test sig_b ≈ expected_b atol = 1e-12
    @test length(sig_b) == length(Xb)
    @test calls[] == length(Xb)

    Bref = [[0.0], [0.1], [0.2], [0.3]]
    mu_fit, inv_mul_fit = CausalSetZoology._fit_reference(Bref, 1e-8)
    sig_fit = CausalSetZoology._mahal_sigmas([[0.0], [0.2]], mu_fit, inv_mul_fit)
    @test length(sig_fit) == 2
    @test all(isfinite, sig_fit)
end


@testitem "distinguishability: _summary_stat" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: _summary_stat behavior and output contract.
    x = [1.0, 2.0, 3.0, 4.0]
    @test CausalSetZoology._summary_stat(x, 0.0) == 1.0
    @test CausalSetZoology._summary_stat(x, 0.25) == Statistics.quantile(x, 0.25)
    @test CausalSetZoology._summary_stat(x, 0.5) == Statistics.quantile(x, 0.5)
    @test CausalSetZoology._summary_stat(x, 1.0) == 4.0
end


@testitem "distinguishability helpers: mahalanobis resampling serial and chunk" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: mahalanobis resampling serial and chunk behavior and output contract.
    X = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
    seeds = UInt64[11, 22, 33, 44]
    args = (seeds, X, 1e-8, 0.0, :regularization, 1e-10, false, 1e-12)

    out_many = CausalSetZoology._mahal_resample_many(args...)
    out_chunk = CausalSetZoology._mahal_resample_chunk(args)
    out_once = [
        CausalSetZoology._mahal_resample_once(s, X, 1e-8, 0.0, :regularization, 1e-10, false, 1e-12)
        for s in seeds
    ]
    @test out_many == out_chunk
    @test out_many ≈ out_once atol = 1e-12
    @test length(out_many) == length(seeds)
    @test all(isfinite, out_many)
end


@testitem "distinguishability helpers: mahalanobis resampling distributed" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: mahalanobis resampling distributed behavior and output contract.
    n_before = Distributed.nworkers()
    let added = Int[]
        try
            needed = max(2, n_before) - n_before
            if needed > 0
                added = Distributed.addprocs(needed)
            end
            @everywhere begin
                using Random, Statistics, LinearAlgebra, Distributions, Distributed, ProgressMeter
                import Random: randperm
                import CausalSetZoology
            end

            X = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
            seeds = UInt64[11, 22, 33, 44, 55, 66]
            serial = CausalSetZoology._mahal_resample_many(seeds, X, 1e-8, 0.0, :regularization, 1e-10, false, 1e-12)
            dist = CausalSetZoology._mahal_resample_many_distributed(
                seeds,
                X,
                1e-8,
                0.0,
                :regularization,
                1e-10,
                false,
                1e-12,
                Distributed.workers()[1:min(2, Distributed.nworkers())],
            )
            @test dist ≈ serial atol = 1e-12
        finally
            if !isempty(added)
                Distributed.rmprocs(added...)
            end
        end
    end
end


@testitem "distinguishability helpers: mahalanobis resampling validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: mahalanobis resampling validation behavior and output contract.
    X = [[0.0], [0.1], [0.2], [0.3]]
    @test_throws ArgumentError CausalSetZoology._mahal_resample_once(UInt64(1), X, 0.0, 0.0, :bad_method, 1e-10, false, 1e-12)
end


@testitem "distinguishability: _random_split_equal" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: _random_split_equal behavior and output contract.
    B = [[0.0], [0.1], [0.2], [0.3]]
    b1, b2 = CausalSetZoology._random_split_equal(B, Random.Xoshiro(22))
    @test length(b1) == 2
    @test length(b2) == 2
    @test isempty(intersect(b1, b2))
    @test all(v in B for v in vcat(b1, b2))

    Bodd = [[0.0], [0.1], [0.2], [0.3], [0.4]]
    b1o, b2o = CausalSetZoology._random_split_equal(Bodd, Random.Xoshiro(7))
    @test length(b1o) == 2
    @test length(b2o) == 2
    @test length(vcat(b1o, b2o)) == 4

    s1 = CausalSetZoology._random_split_equal(B, Random.Xoshiro(123))
    s2 = CausalSetZoology._random_split_equal(B, Random.Xoshiro(123))
    @test s1 == s2
end


@testitem "distinguishability: mahalanobis helper validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis helper validation behavior and output contract.
    @test_throws DomainError CausalSetZoology._random_split_equal([[0.0]], Random.Xoshiro(1))

    Bref = [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8]]
    @test_throws ArgumentError CausalSetZoology._fit_reference(Bref, 0.0; stabilization_method = :not_a_method)

    Bref_sing = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    @test_throws DomainError CausalSetZoology._fit_reference(Bref_sing, 0.0; stabilization_method = :regularization)
    @test_throws DomainError CausalSetZoology._fit_reference(Bref_sing, 0.0; stabilization_method = :projection, projection_tolerance = 1e6)
end


@testitem "distinguishability: mahalanobis core contract and analytic M_obs" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis core contract and analytic M_obs behavior and output contract.
    b = [[0.0], [1.0], [2.0], [3.0]]
    a_far = [[10.0], [11.0], [12.0], [13.0]]
    reg = 1e-8
    sigma_b = sqrt((5 / 3) + reg)
    expected_min_far = 8.5 / sigma_b

    res = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_far,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        q = 0.0,
        alpha = 0.05,
        regulator = reg,
    )
    @test keys(res) == (:M_obs, :distinguishable, :threshold, :z_emp, :M_obs_sym, :M_obs_min, :threshold_sym, :threshold_max)
    @test res.M_obs ≈ expected_min_far atol = 1e-12
    @test res.M_obs_sym === nothing
    @test res.M_obs_min === nothing
    @test res.threshold_sym === nothing
    @test res.threshold_max === nothing
    @test isfinite(res.threshold)
    @test isfinite(res.z_emp)
    @test res.M_obs > res.threshold
    @test res.distinguishable
end


@testitem "distinguishability: mahalanobis core reproducibility" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis core reproducibility behavior and output contract.
    b = [[0.0], [1.0], [2.0], [3.0]]
    a_far = [[10.0], [11.0], [12.0], [13.0]]
    reg = 1e-8

    res_1 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_far,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        q = 0.0,
        alpha = 0.05,
        regulator = reg,
    )
    res_2 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_far,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        q = 0.0,
        alpha = 0.05,
        regulator = reg,
    )
    @test res_2 == res_1
end


@testitem "distinguishability: mahalanobis core q and alpha behavior" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis core q and alpha behavior behavior and output contract.
    b = [[0.0], [1.0], [2.0], [3.0]]
    a_far = [[10.0], [11.0], [12.0], [13.0]]
    reg = 1e-8
    sigma_b = sqrt((5 / 3) + reg)
    expected_min_far = 8.5 / sigma_b
    expected_med_far = 10.0 / sigma_b

    res_q0 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_far,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        q = 0.0,
        alpha = 0.05,
        regulator = reg,
    )
    res_q50 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_far,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        q = 0.5,
        alpha = 0.05,
        regulator = reg,
    )
    @test res_q0.M_obs ≈ expected_min_far atol = 1e-12
    @test res_q50.M_obs ≈ expected_med_far atol = 1e-12
    @test res_q50.M_obs > res_q0.M_obs

    res_alpha0 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_far,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        q = 0.0,
        alpha = 0.0,
        regulator = reg,
    )
    @test res_alpha0.threshold >= res_q0.threshold
end


@testitem "distinguishability: mahalanobis core symmetric mode" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis core symmetric mode behavior and output contract.
    b = [[0.0], [1.0], [2.0], [3.0]]
    a_far = [[10.0], [11.0], [12.0], [13.0]]
    reg = 1e-8
    sigma_b = sqrt((5 / 3) + reg)
    expected_min_far = 8.5 / sigma_b

    res_sym = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_far,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        symmetric = true,
        q = 0.0,
        alpha = 0.05,
        regulator = reg,
    )
    @test res_sym.M_obs ≈ expected_min_far atol = 1e-12
    @test res_sym.M_obs_sym ≈ expected_min_far atol = 1e-12
    @test res_sym.M_obs_min == min(res_sym.M_obs, res_sym.M_obs_sym)
    @test res_sym.threshold_max == max(res_sym.threshold, res_sym.threshold_sym)
    @test res_sym.distinguishable == ((res_sym.M_obs > res_sym.threshold) && (res_sym.M_obs_sym > res_sym.threshold_sym))
end


@testitem "distinguishability: mahalanobis core near vs far" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis core near vs far behavior and output contract.
    b = [[0.0], [1.0], [2.0], [3.0]]
    a_far = [[10.0], [11.0], [12.0], [13.0]]
    reg = 1e-8
    sigma_b = sqrt((5 / 3) + reg)
    expected_min_near = 0.5 / sigma_b

    res_far = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_far,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        q = 0.0,
        alpha = 0.05,
        regulator = reg,
    )
    res_near = CausalSetZoology.mahalanobis_gap_distinguishability(
        b,
        b;
        R = 100,
        rng = Random.Xoshiro(99),
        q = 0.0,
        alpha = 0.05,
        regulator = reg,
    )
    @test res_near.M_obs ≈ expected_min_near atol = 1e-12
    @test res_near.M_obs < res_far.M_obs
end


@testitem "distinguishability: distributed mahalanobis path" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: distributed mahalanobis path behavior and output contract.
    n_before = Distributed.nworkers()
    let added = Int[]
        try
            needed = max(2, n_before) - n_before
            if needed > 0
                added = Distributed.addprocs(needed)
            end
            @everywhere begin
                using Random, Statistics, LinearAlgebra, Distributions, Distributed, ProgressMeter
                import Random: randperm
                import CausalSetZoology
            end

            a = [[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7]]
            b = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]]
            seeds = rand(Random.Xoshiro(4242), UInt64, 64)
            @test length(unique(seeds)) == length(seeds)

            rs = CausalSetZoology.mahalanobis_gap_distinguishability(
                a,
                b;
                R = 25,
                rng = Random.Xoshiro(6060),
                num_workers = 1,
                regulator = 1e-8,
            )
            rd = CausalSetZoology.mahalanobis_gap_distinguishability(
                a,
                b;
                R = 25,
                rng = Random.Xoshiro(6060),
                num_workers = max(2, Distributed.nworkers()),
                regulator = 1e-8,
            )
            @test keys(rd) == keys(rs)
            @test rd.M_obs ≈ rs.M_obs atol = 1e-12
            @test rd.threshold ≈ rs.threshold atol = 1e-12
            @test rd.z_emp ≈ rs.z_emp atol = 1e-12
            @test rd.distinguishable == rs.distinguishable
            @test isfinite(rd.M_obs)
        finally
            if !isempty(added)
                Distributed.rmprocs(added...)
            end
        end
    end
end

@testitem "distinguishability: scalar_bin_mahalanobis wrappers" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: scalar_bin_mahalanobis wrappers behavior and output contract.
    rng = Random.Xoshiro(5353)
    data = [Tuple{Vector{Float64},Real}[
        ([0.0], 1.0), ([0.1], 1.0), ([0.2], 1.0), ([0.3], 1.0),
        ([1.0], 2.0), ([1.1], 2.0), ([1.2], 2.0), ([1.3], 2.0),
    ]]
    ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]

    out_pairs = CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(data; num_bins = 2, R = 20, rng = rng, regulator = 1e-8)
    @test length(out_pairs) == 1
    @test keys(out_pairs[1]) == (:s1, :s2, :rel_change, :M_obs, :distinguishable, :threshold, :z_emp, :M_obs_sym, :M_obs_min, :threshold_sym, :threshold_max)
    @test isfinite(out_pairs[1].M_obs)

    out_ref = CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(data, ref; num_bins = 2, R = 20, rng = rng, regulator = 1e-8)
    @test length(out_ref) == 2
    @test all(keys(x) == (:scalar, :M_obs, :distinguishable, :threshold, :z_emp, :M_obs_sym, :M_obs_min, :threshold_sym, :threshold_max) for x in out_ref)
    @test [x.scalar for x in out_ref] ≈ [1.0, 2.0]
    @test all(isfinite(x.M_obs) for x in out_ref)
end


@testitem "distinguishability: mahalanobis wrapper validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis wrapper validation behavior and output contract.
    a = [[3.0], [2.9], [3.1], [3.2]]
    b = [[0.0], [0.1], [-0.1], [0.2]]
    rng = Random.Xoshiro(99)
    @test_throws DomainError CausalSetZoology.mahalanobis_gap_distinguishability(a, b; R = 0)
    @test_throws DomainError CausalSetZoology.mahalanobis_gap_distinguishability(a, b; q = -0.1)
    @test_throws DomainError CausalSetZoology.mahalanobis_gap_distinguishability(a, b; alpha = 1.0)
    @test_throws DomainError CausalSetZoology.mahalanobis_gap_distinguishability(a, b; regulator = -1.0)
    @test_throws ArgumentError CausalSetZoology.mahalanobis_gap_distinguishability(Vector{Vector{Float64}}(), b)
    @test_throws ArgumentError CausalSetZoology.mahalanobis_gap_distinguishability(Any[1], Any[2])
    @test_throws TypeError CausalSetZoology.mahalanobis_gap_distinguishability(Any[[1.0, "bad"]], Any[[0.0, 1.0]])

    data = [Tuple{Vector{Float64},Real}[([0.0], 1.0), ([1.0], 2.0)]]
    bad_pair_slot = [Any[("x", "y"), ("u", "v")]]
    bad_scalar_val = [Any[(Float64[0.0], 1.0), (Float64[1.0], "bad")]]
    @test_throws DimensionMismatch CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(vcat(data, data); num_bins = 2, R = 10, rng = rng)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability([Tuple{Vector{Float64},Real}[]]; num_bins = 2, R = 10, rng = rng)
    @test_throws TypeError CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(bad_pair_slot; num_bins = 2, R = 10, rng = rng)
    @test_throws TypeError CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(bad_scalar_val; num_bins = 2, R = 10, rng = rng)
    @test_throws DomainError CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(data; num_bins = 0, R = 10, rng = rng)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(data, Vector{Vector{Float64}}(); num_bins = 2, R = 10, rng = rng)
    @test_throws TypeError CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(bad_pair_slot, [[0.0]]; num_bins = 2, R = 10, rng = rng)
    @test_throws TypeError CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(bad_scalar_val, [[0.0]]; num_bins = 2, R = 10, rng = rng)
    @test_throws DomainError CausalSetZoology.scalar_bin_mahalanobis_gap_distinguishability(data, [[0.0]]; num_bins = 0, R = 10, rng = rng)
end
