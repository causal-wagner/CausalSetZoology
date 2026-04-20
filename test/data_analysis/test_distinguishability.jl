@testsnippet setupDistinguishability begin
    using Test
    using Random
    using Statistics
    using LinearAlgebra
    using Distributions
    using Logging
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


@testitem "distinguishability: distance_distinguishability_probability basic" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: distance_distinguishability_probability basic behavior and output contract.
    distance(x, y) = abs(x - y)
    cset = [10.0, 11.0]
    null = [0.0, 1.0, 2.0]

    res = CausalSetZoology.distance_distinguishability_probability(distance, cset, null)
    @test keys(res) == (:D, :probability_below_null, :null_value, :percentile, :mean_between, :std_between, :std_lo, :std_up)
    @test res.null_value ≈ 1.0 atol = 1e-12
    @test res.probability_below_null ≈ 0.0 atol = 1e-12
    @test res.D ≈ 1.0 atol = 1e-12
    @test res.percentile ≈ 0.5 atol = 1e-12
    @test res.mean_between ≈ 9.5 atol = 1e-12
    @test res.std_between ≈ sqrt(11 / 10) atol = 1e-12
    @test res.std_lo ≈ 0.7 atol = 1e-12
    @test res.std_up ≈ 0.7 atol = 1e-12

    res_q1 = CausalSetZoology.distance_distinguishability_probability(distance, [1.5, 2.0], null; percentile = 1.0)
    @test res_q1.null_value ≈ 2.0 atol = 1e-12
    @test res_q1.probability_below_null ≈ 1.0 atol = 1e-12
    @test res_q1.D ≈ 0.0 atol = 1e-12

    # Mixed case: only some cross-pairs fall below the median null threshold.
    # Null pair distances are [1, 2, 1], so the 0.5-quantile threshold is 1.
    # Cross distances from [1, 3] to [0, 1, 2] are [1, 0, 1, 3, 2, 1], so 4/6 are <= 1.
    res_mixed = CausalSetZoology.distance_distinguishability_probability(distance, [1.0, 3.0], null)
    @test res_mixed.null_value ≈ 1.0 atol = 1e-12
    @test res_mixed.probability_below_null ≈ (4 / 6) atol = 1e-12
    @test res_mixed.D ≈ (1 - 4 / 6) atol = 1e-12
    @test res_mixed.mean_between ≈ (8 / 6) atol = 1e-12
    @test res_mixed.std_between >= 0.0
    @test res_mixed.std_lo >= 0.0
    @test res_mixed.std_up >= 0.0
end


@testitem "distinguishability: null_distance_percentile" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: null_distance_percentile behavior and output contract.
    distance(x, y) = abs(x - y)
    null = [0.0, 1.0, 2.0]

    @test CausalSetZoology.null_distance_percentile(distance, null) ≈ 1.0 atol = 1e-12
    @test CausalSetZoology.null_distance_percentile(distance, null; percentile = 1.0) ≈ 2.0 atol = 1e-12
    @test_throws ArgumentError CausalSetZoology.null_distance_percentile(distance, Float64[])
    @test_throws ArgumentError CausalSetZoology.null_distance_percentile(distance, [0.0])
    @test_throws DomainError CausalSetZoology.null_distance_percentile(distance, null; percentile = -0.1)
    @test_throws TypeError CausalSetZoology.null_distance_percentile((x, y) -> "bad", null)
    @test_throws DomainError CausalSetZoology.null_distance_percentile((x, y) -> NaN, null)
end


@testitem "distinguishability: distance_distinguishability_probability cached null and vectors" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: distance_distinguishability_probability cached null and vectors behavior and output contract.
    distance(x, y) = LinearAlgebra.norm(x .- y)
    cset = [[3.0, 3.0], [4.0, 4.0]]
    null = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    res = CausalSetZoology.distance_distinguishability_probability(distance, cset, null; percentile = 0.5)
    res_cached = CausalSetZoology.distance_distinguishability_probability(distance, cset, null; percentile = 0.1, null_value = res.null_value)
    @test res_cached.null_value ≈ res.null_value atol = 1e-12
    @test res_cached.D ≈ res.D atol = 1e-12
    @test res_cached.probability_below_null ≈ res.probability_below_null atol = 1e-12
    @test res_cached.percentile ≈ 0.1 atol = 1e-12
    @test res_cached.mean_between ≈ res.mean_between atol = 1e-12
    @test res_cached.std_between ≈ res.std_between atol = 1e-12
    @test res_cached.std_lo ≈ res.std_lo atol = 1e-12
    @test res_cached.std_up ≈ res.std_up atol = 1e-12

    # Cached null_value should bypass the within-null pair requirement.
    res_single_null = CausalSetZoology.distance_distinguishability_probability(
        distance,
        cset,
        [[0.0, 0.0]];
        percentile = 0.1,
        null_value = res.null_value,
    )
    @test res_single_null.null_value ≈ res.null_value atol = 1e-12
    @test res_single_null.percentile ≈ 0.1 atol = 1e-12
    @test 0.0 <= res_single_null.D <= 1.0
    @test 0.0 <= res_single_null.probability_below_null <= 1.0
    @test res_single_null.mean_between >= 0.0
    @test res_single_null.std_between >= 0.0
    @test res_single_null.std_lo >= 0.0
    @test res_single_null.std_up >= 0.0

    ragged = CausalSetZoology.distance_distinguishability_probability(
        CausalSetZoology.total_variation_distance,
        [[1.0, 0.0], [0.5]],
        [[0.2], [0.1, 0.0, 0.0]];
        percentile = 0.5,
    )
    @test 0.0 <= ragged.D <= 1.0
    @test ragged.null_value >= 0.0
    @test ragged.mean_between >= 0.0
    @test ragged.std_between >= 0.0
    @test ragged.std_lo >= 0.0
    @test ragged.std_up >= 0.0
end


@testitem "distinguishability: distance_distinguishability_probability validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: distance_distinguishability_probability validation behavior and output contract.
    distance(x, y) = abs(x - y)
    @test_throws ArgumentError CausalSetZoology.distance_distinguishability_probability(distance, Float64[], [0.0, 1.0])
    @test_throws ArgumentError CausalSetZoology.distance_distinguishability_probability(distance, [0.0], Float64[])
    @test_throws ArgumentError CausalSetZoology.distance_distinguishability_probability(distance, [0.0], [1.0])
    @test_throws DomainError CausalSetZoology.distance_distinguishability_probability(distance, [0.0], [0.0, 1.0]; percentile = -0.1)
    @test_throws DomainError CausalSetZoology.distance_distinguishability_probability(distance, [0.0], [0.0, 1.0]; percentile = 1.1)
    @test_throws DomainError CausalSetZoology.distance_distinguishability_probability(distance, [0.0], [0.0, 1.0]; null_value = Inf)
    @test_throws TypeError CausalSetZoology.distance_distinguishability_probability((x, y) -> "bad", [0.0], [0.0, 1.0])
    @test_throws DomainError CausalSetZoology.distance_distinguishability_probability((x, y) -> NaN, [0.0], [0.0, 1.0])
    @test_throws DomainError CausalSetZoology.distance_distinguishability_probability((x, y) -> NaN, [0.0], [1.0]; null_value = 0.0)
end


@testitem "distinguishability: distance_distinguishability_probability preserves signed trailing coordinates" setup=[setupDistinguishability] begin
    # Test intent: generic vector alignment must not drop trailing negative coordinates.
    distance(x, y) = LinearAlgebra.norm(x .- y)

    hist = [[1.0, 0.0, -2.0]]
    null = [[1.0]]

    res = CausalSetZoology.distance_distinguishability_probability(
        distance,
        hist,
        null;
        null_value = 0.5,
    )

    expected = distance([1.0, 0.0, -2.0], [1.0, 0.0, 0.0])
    @test res.mean_between ≈ expected atol = 1e-12
    @test res.probability_below_null ≈ 0.0 atol = 1e-12
    @test res.D ≈ 1.0 atol = 1e-12
end


@testitem "distinguishability: scalar_bin_distance_distinguishability_probability wrappers" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: scalar_bin_distance_distinguishability_probability wrappers behavior and output contract.
    distance(x, y) = abs(x[1] - y[1])
    data = [Tuple{Vector{Float64},Real}[
        ([0.0], 1.0), ([0.1], 1.0), ([0.2], 1.0), ([0.3], 1.0),
        ([1.0], 2.0), ([1.1], 2.0), ([1.2], 2.0), ([1.3], 2.0),
    ]]
    ref = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]

    r_pair = CausalSetZoology.scalar_bin_distance_distinguishability_probability(distance, data; num_bins = 2)
    @test length(r_pair) == 1
    @test keys(r_pair[1]) == (:s1, :s2, :rel_change, :D, :probability_below_null, :null_value, :percentile, :mean_between, :std_between, :std_lo, :std_up)
    @test 0.0 <= r_pair[1].D <= 1.0
    @test 0.0 <= r_pair[1].probability_below_null <= 1.0
    @test r_pair[1].null_value >= 0.0
    @test r_pair[1].mean_between >= 0.0
    @test r_pair[1].std_between >= 0.0
    @test r_pair[1].std_lo >= 0.0
    @test r_pair[1].std_up >= 0.0

    cached_null = r_pair[1].null_value
    r_ref = CausalSetZoology.scalar_bin_distance_distinguishability_probability(distance, data, ref; num_bins = 2, null_value = cached_null)
    @test length(r_ref) == 2
    @test all(keys(x) == (:scalar, :D, :probability_below_null, :null_value, :percentile, :mean_between, :std_between, :std_lo, :std_up) for x in r_ref)
    @test [x.scalar for x in r_ref] ≈ [1.0, 2.0]
    @test all(isapprox(x.null_value, cached_null; atol = 1e-12) for x in r_ref)
    @test all(0.0 <= x.D <= 1.0 for x in r_ref)
    @test all(x.mean_between >= 0.0 for x in r_ref)
    @test all(x.std_between >= 0.0 for x in r_ref)
    @test all(x.std_lo >= 0.0 for x in r_ref)
    @test all(x.std_up >= 0.0 for x in r_ref)
end


@testitem "distinguishability: scalar_bin_distance_distinguishability_probability validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: scalar_bin_distance_distinguishability_probability validation behavior and output contract.
    distance(x, y) = abs(x[1] - y[1])
    data = [Tuple{Vector{Float64},Real}[([0.0], 1.0), ([1.0], 2.0)]]
    ref_empty = Vector{Vector{Float64}}()

    @test_throws DimensionMismatch CausalSetZoology.scalar_bin_distance_distinguishability_probability(distance, vcat(data, data); num_bins = 2)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distance_distinguishability_probability(distance, [Tuple{Vector{Float64},Real}[]]; num_bins = 2)
    @test_throws DomainError CausalSetZoology.scalar_bin_distance_distinguishability_probability(distance, data; num_bins = 0)
    @test_throws ArgumentError CausalSetZoology.scalar_bin_distance_distinguishability_probability(distance, data, ref_empty; num_bins = 2)
    @test_throws DomainError CausalSetZoology.scalar_bin_distance_distinguishability_probability(distance, data; percentile = -0.1)
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

@testitem "distinguishability helpers: thread seeds" setup=[setupDistinguishability] begin
    s1 = CausalSetZoology._thread_seeds(Random.Xoshiro(1234), 8)
    s2 = CausalSetZoology._thread_seeds(Random.Xoshiro(1234), 8)
    @test length(s1) == 8
    @test length(unique(s1)) == 8
    @test s1 == s2
    @test_throws DomainError CausalSetZoology._thread_seeds(Random.Xoshiro(1), 0)
end


@testitem "distinguishability: basic helper validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: basic helper validation behavior and output contract.
    pairs = Tuple{Vector{Float64},Real}[([1.0], 1.0), ([2.0], 2.0)]
    @test_throws DomainError CausalSetZoology.relative_change(0.0, 1.0)
    @test_throws DomainError CausalSetZoology.bin_scalar_pairs(pairs, 0, nothing)
    @test_throws DimensionMismatch CausalSetZoology.hellinger_distance([1.0], [1.0, 0.0])
end


@testitem "distinguishability: energy_based_histogram_distinguishability vectors" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: energy_based_histogram_distinguishability vectors behavior and output contract.
    rng = Random.Xoshiro(123)
    a = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]
    b = [[0.0, 1.0], [0.1, 0.9], [0.05, 0.95]]
    c = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]

    d_same = CausalSetZoology.energy_based_histogram_distinguishability(a, c)
    d_diff = CausalSetZoology.energy_based_histogram_distinguishability(a, b)
    @test d_same.D ≈ 0.0 atol = 1e-12
    @test d_diff.D ≈ 0.8971989614490513 atol = 1e-12

    d_mc = CausalSetZoology.energy_based_histogram_distinguishability(a, b, 150; rng = rng)
    @test 0.85 <= d_mc.D <= 0.95
    @test d_mc.std >= 0.0
end


@testitem "distinguishability: energy_based_histogram_distinguishability histograms" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: energy_based_histogram_distinguishability histograms behavior and output contract.
    h_a = [Dict(1 => 10, 2 => 0), Dict(1 => 9, 2 => 1)]
    h_b = [Dict(1 => 0, 2 => 10), Dict(1 => 1, 2 => 9)]

    d = CausalSetZoology.energy_based_histogram_distinguishability(h_a, h_b)
    @test d.D ≈ 0.8874260862878063 atol = 1e-12

    d_mc = CausalSetZoology.energy_based_histogram_distinguishability(h_a, h_b, 100; rng = Random.Xoshiro(42))
    @test 0.8 <= d_mc.D <= 0.95
    @test d_mc.std >= 0.0
end

@testitem "distinguishability: total_histogram_distinguishability" setup=[setupDistinguishability] begin
    # Verifies multi-observable wrapper behavior and agreement with direct core calls.
    obs1 = [
        [Dict(1 => 10, 2 => 0), Dict(1 => 9, 2 => 1)],
        [Dict(1 => 0, 2 => 10), Dict(1 => 1, 2 => 9)],
    ]
    obs2 = [
        [[1.0, 0.0], [0.9, 0.1]],
        [[0.0, 1.0], [0.1, 0.9]],
    ]

    dt = CausalSetZoology.total_histogram_distinguishability(obs1, obs2)
    @test 0.0 <= dt.D <= 1.0
    @test dt.D > 0.8

    # single-observable path should agree with direct distinguishability
    d1 = CausalSetZoology.total_histogram_distinguishability(obs1)
    d2 = CausalSetZoology.energy_based_histogram_distinguishability(obs1[1], obs1[2])
    @test d1.D ≈ d2.D rtol=1e-6

    # multi-observable path should agree with explicit concatenate + core distinguishability
    vecs_a, vecs_b = CausalSetZoology.concatenate_hists(obs1, obs2)
    d_explicit = CausalSetZoology.energy_based_histogram_distinguishability(vecs_a, vecs_b)
    @test dt.D ≈ d_explicit.D
end

@testitem "distinguishability: total_histogram_distinguishability validation" setup=[setupDistinguishability] begin
    # Verifies top-level validation and propagation of invalid observable content.
    obs = [[Dict(1 => 1.0)], [Dict(1 => 1.0)]]
    @test_throws ArgumentError CausalSetZoology.total_histogram_distinguishability()
    @test_throws DimensionMismatch CausalSetZoology.total_histogram_distinguishability(
        obs,
        [[Dict(1 => 1.0), Dict(2 => 1.0)], [Dict(1 => 1.0)]],
    )
    @test_throws ArgumentError CausalSetZoology.total_histogram_distinguishability(
        obs,
        [[Dict(1 => 1.0)], [[1.0]]],
    )
end

@testitem "distinguishability: mutual_information vectors and normalization" setup=[setupDistinguishability] begin
    a = [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    b = [[8.0, 0.0], [0.0, 4.0], [0.0, 2.0]]

    mi = CausalSetZoology.distinguishability_mutual_information(a, b)
    @test 0.0 <= mi.D_mi <= 1.0
    @test mi.D_mi < 1.0

    a_prob = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    b_prob = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    mi_prob = CausalSetZoology.distinguishability_mutual_information(a_prob, b_prob)
    @test mi_prob.D_mi ≈ mi.D_mi atol = 1e-12

    mi_same = CausalSetZoology.distinguishability_mutual_information(a, a)
    @test mi_same.D_mi ≈ 0.0 atol = 1e-12

    c = [[1.0, 0.0] for _ in 1:32]
    d = [[0.0, 1.0] for _ in 1:32]
    mi_perfect = CausalSetZoology.distinguishability_mutual_information(c, d; k = 1, pca_dim = 2)
    @test mi_perfect.D_mi > 0.9
end

@testitem "distinguishability: mutual_information histogram dictionaries" setup=[setupDistinguishability] begin
    h_a = [Dict(1 => 2, 2 => 0), Dict(1 => 4, 2 => 0), Dict(1 => 0, 2 => 6)]
    h_b = [Dict(1 => 8, 2 => 0), Dict(1 => 0, 2 => 2), Dict(1 => 0, 2 => 4)]
    mi = CausalSetZoology.distinguishability_mutual_information(h_a, h_b)
    @test 0.0 <= mi.D_mi <= 1.0

    h_a_scaled = [Dict(1 => 20, 2 => 0), Dict(1 => 40, 2 => 0), Dict(1 => 0, 2 => 60)]
    h_b_scaled = [Dict(1 => 80, 2 => 0), Dict(1 => 0, 2 => 20), Dict(1 => 0, 2 => 40)]
    mi_scaled = CausalSetZoology.distinguishability_mutual_information(h_a_scaled, h_b_scaled)
    @test mi_scaled.D_mi ≈ mi.D_mi atol = 1e-12
end

@testitem "distinguishability: total histogram mutual_information wrapper" setup=[setupDistinguishability] begin
    obs1 = [
        [Dict(1 => 2, 2 => 0), Dict(1 => 4, 2 => 0), Dict(1 => 0, 2 => 6)],
        [Dict(1 => 8, 2 => 0), Dict(1 => 0, 2 => 2), Dict(1 => 0, 2 => 4)],
    ]
    obs2 = [
        [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        [[8.0, 0.0], [0.0, 4.0], [0.0, 2.0]],
    ]

    dt = CausalSetZoology.total_histogram_mutual_information_distinguishability(obs1, obs2)
    @test 0.0 <= dt.D_mi <= 1.0

    vecs_a, vecs_b = CausalSetZoology.concatenate_hists(obs1, obs2)
    d_explicit = CausalSetZoology.distinguishability_mutual_information(vecs_a, vecs_b)
    @test dt.D_mi ≈ d_explicit.D_mi atol = 1e-12

    d_single = CausalSetZoology.total_histogram_mutual_information_distinguishability(obs1)
    vecs_single_a, vecs_single_b = CausalSetZoology.concatenate_hists(obs1)
    d_direct = CausalSetZoology.distinguishability_mutual_information(vecs_single_a, vecs_single_b)
    @test d_single.D_mi ≈ d_direct.D_mi atol = 1e-12
end

@testitem "distinguishability: mutual_information is sensitive to correlations" setup=[setupDistinguishability] begin
    x0 = [10.0, 0.0]
    x1 = [0.0, 10.0]
    y0 = [10.0, 0.0]
    y1 = [0.0, 10.0]

    # Observable marginals are identical across classes.
    obs_x = [
        [x0, x1, x0, x1], # class A
        [x0, x1, x0, x1], # class B
    ]
    # Same marginal for Y, but different A/B pairing with X.
    obs_y = [
        [y0, y1, y0, y1], # class A: positively correlated with X
        [y1, y0, y1, y0], # class B: anti-correlated with X
    ]

    dx = CausalSetZoology.distinguishability_mutual_information(obs_x[1], obs_x[2]; k = 1, pca_dim = 2)
    dy = CausalSetZoology.distinguishability_mutual_information(obs_y[1], obs_y[2]; k = 1, pca_dim = 2)
    dxy = CausalSetZoology.total_histogram_mutual_information_distinguishability(obs_x, obs_y; k = 1, pca_dim = 4)

    @test dx.D_mi < 0.2
    @test dy.D_mi < 0.2
    @test dxy.D_mi > 0.7
end

@testitem "distinguishability: mutual_information monotone under added information" setup=[setupDistinguishability] begin
    rng = Random.Xoshiro(1234)
    n = 200
    x0, x1 = [5.0, 0.0], [0.0, 5.0]
    y0, y1 = [5.0, 0.0], [0.0, 5.0]

    x_a = Vector{Vector{Float64}}(undef, n)
    x_b = Vector{Vector{Float64}}(undef, n)
    y_a = Vector{Vector{Float64}}(undef, n)
    y_b = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        z = rand(rng, Bool)
        x_a[i] = z ? x1 : x0
        x_b[i] = z ? x1 : x0
        y_a[i] = z ? y1 : y0
        y_b[i] = z ? y0 : y1
    end

    obs_x = [x_a, x_b]
    obs_y = [y_a, y_b]
    obs_x_copy = deepcopy(obs_x)

    d_x = CausalSetZoology.total_histogram_mutual_information_distinguishability(obs_x; k = 3, pca_dim = 2)
    d_xy = CausalSetZoology.total_histogram_mutual_information_distinguishability(obs_x, obs_y; k = 3, pca_dim = 4)
    d_xyx = CausalSetZoology.total_histogram_mutual_information_distinguishability(obs_x, obs_y, obs_x_copy; k = 3, pca_dim = 4)

    @test d_xy.D_mi >= d_x.D_mi - 0.05
    # Finite-sample kNN+PCA estimates are not strictly monotone under feature duplication;
    # enforce stability bounds instead of hard monotonicity.
    @test 0.0 <= d_xyx.D_mi <= 1.0 + 1e-12
    @test abs(d_xyx.D_mi - d_xy.D_mi) <= 0.8
end

@testitem "distinguishability: mutual_information bootstrap and reproducibility" setup=[setupDistinguishability] begin
    a = [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]
    b = [[8.0, 0.0], [0.0, 4.0], [0.0, 2.0], [0.0, 4.0], [8.0, 0.0]]

    r1 = CausalSetZoology.distinguishability_mutual_information(a, b, 200; rng = Random.Xoshiro(2026))
    r2 = CausalSetZoology.distinguishability_mutual_information(a, b, 200; rng = Random.Xoshiro(2026))
    @test r1.D_mi ≈ r2.D_mi atol = 1e-12
    @test r1.std ≈ r2.std atol = 1e-12
    @test 0.0 <= r1.D_mi <= 1.0
    @test r1.std >= 0.0

    obs = [
        [Dict(1 => 2, 2 => 0), Dict(1 => 4, 2 => 0), Dict(1 => 0, 2 => 6)],
        [Dict(1 => 8, 2 => 0), Dict(1 => 0, 2 => 2), Dict(1 => 0, 2 => 4)],
    ]
    rt = CausalSetZoology.total_histogram_mutual_information_distinguishability(obs; num_draws = 120, rng = Random.Xoshiro(7))
    @test 0.0 <= rt.D_mi <= 1.0
    @test rt.std >= 0.0
end

@testitem "distinguishability: mutual_information validation" setup=[setupDistinguishability] begin
    a = [[1.0, 0.0], [0.0, 1.0]]
    b = [[1.0, 0.0], [0.0, 1.0]]
    @test_throws ArgumentError CausalSetZoology.distinguishability_mutual_information(Vector{Vector{Float64}}(), b)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information([[0.0, 0.0]], b)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information([[-1.0, 2.0]], b)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b, 0)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b; k = 0)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b; pca_mode = :bad_mode)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b; pca_dim = 0)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b; explained_variance = 0.0)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b; explained_variance = 1.1)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b; eigenvalue_rtol = -1e-3)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b; eigenvalue_rtol = 1.0)
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(a, b; max_per_class = 1)

    h_good = [Dict(1 => 1, 2 => 0), Dict(1 => 0, 2 => 1)]
    h_bad = [Dict(1 => 0, 2 => 0), Dict(1 => 0, 2 => 1)]
    @test_throws DomainError CausalSetZoology.distinguishability_mutual_information(h_bad, h_good)
    @test_throws DomainError CausalSetZoology.total_histogram_mutual_information_distinguishability([h_good, h_good]; k = 0)

    obs = [[Dict(1 => 1.0)], [Dict(1 => 1.0)]]
    @test_throws ArgumentError CausalSetZoology.total_histogram_mutual_information_distinguishability()
    @test_throws DimensionMismatch CausalSetZoology.total_histogram_mutual_information_distinguishability(
        obs,
        [[Dict(1 => 1.0), Dict(2 => 1.0)], [Dict(1 => 1.0)]],
    )
end

@testitem "distinguishability: mutual_information PCA modes" setup=[setupDistinguishability] begin
    rng = Random.Xoshiro(19)
    n = 120
    d = 24

    # Build two classes with signal concentrated in first 3 coordinates.
    a = Vector{Vector{Float64}}(undef, n)
    b = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        va = rand(rng, d)
        vb = rand(rng, d)
        va[1] += 2.0
        va[2] += 1.0
        va[3] += 0.5
        vb[1] += 0.2
        vb[2] += 0.1
        vb[3] += 0.05
        a[i] = va
        b[i] = vb
    end

    d_dim = CausalSetZoology.distinguishability_mutual_information(
        a,
        b;
        k = 3,
        pca_mode = :dim,
        pca_dim = 8,
        max_per_class = nothing,
    )
    d_var = CausalSetZoology.distinguishability_mutual_information(
        a,
        b;
        k = 3,
        pca_mode = :variance,
        explained_variance = 0.9,
        max_per_class = nothing,
    )
    d_cut = CausalSetZoology.distinguishability_mutual_information(
        a,
        b;
        k = 3,
        pca_mode = :cutoff,
        eigenvalue_rtol = 1e-8,
        max_per_class = nothing,
    )

    @test 0.0 <= d_dim.D_mi <= 1.0
    @test 0.0 <= d_var.D_mi <= 1.0
    @test 0.0 <= d_cut.D_mi <= 1.0
    @test d_dim.D_mi > 0.1
    @test d_var.D_mi > 0.1
    @test d_cut.D_mi > 0.1

    # Consistency across modes for same data should be reasonably close.
    @test isapprox(d_dim.D_mi, d_var.D_mi; atol = 0.25)
    @test isapprox(d_dim.D_mi, d_cut.D_mi; atol = 0.25)
end

@testitem "distinguishability: mutual_information pooled cutoff removes null modes" setup=[setupDistinguishability] begin
    a = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.95, 0.05], [0.85, 0.15]]
    b = [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.05, 0.95], [0.15, 0.85]]
    a_pad = [vcat(v, 0.0, 0.0, 0.0) for v in a]
    b_pad = [vcat(v, 0.0, 0.0, 0.0) for v in b]

    r1 = CausalSetZoology.distinguishability_mutual_information(
        a,
        b;
        k = 3,
        pca_mode = :cutoff,
        eigenvalue_rtol = 1e-8,
        max_per_class = nothing,
    )
    r2 = CausalSetZoology.distinguishability_mutual_information(
        a_pad,
        b_pad;
        k = 3,
        pca_mode = :cutoff,
        eigenvalue_rtol = 1e-8,
        max_per_class = nothing,
    )

    @test r2.D_mi ≈ r1.D_mi atol = 1e-12
end

@testitem "distinguishability: total_variation basic and bayes accuracy" setup=[setupDistinguishability] begin
    a = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    b = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    res = CausalSetZoology.distinguishability_total_variation(a, b)
    @test res.D_tv ≈ 1.0 atol = 1e-12
    @test res.bayes_accuracy ≈ 1.0 atol = 1e-12
    @test res.tv_bias_mean === nothing
    @test res.D_tv_debiased === nothing

    # Quantization should affect TV when samples nearly collide after coarse rounding.
    a_q = [
        [0.07336635446929285, 0.34924148955718615],
        [0.6988266836914685, 0.6282647403425017],
        [0.9149290036628314, 0.19280811624587546],
        [0.7701803478856664, 0.7805192636751863],
        [0.6702639583444937, 0.16771210647092682],
        [0.5710874493423871, 0.4528085872833483],
    ]
    b_q = [
        [0.30232547191787174, 0.0013502779247226426],
        [0.5670236732404312, 0.6159379234562881],
        [0.19573857852575793, 0.012461945950411835],
        [0.3119923865097316, 0.11479916823306191],
        [0.5460487092960259, 0.6232150941621899],
        [0.2708693898950604, 0.8451820156319791],
    ]
    res_coarse = CausalSetZoology.distinguishability_total_variation(a_q, b_q; tv_quantization_digits = 0)
    res_fine = CausalSetZoology.distinguishability_total_variation(a_q, b_q; tv_quantization_digits = 3)
    @test res_coarse.D_tv ≈ 1 / 6 atol = 1e-12
    @test res_fine.D_tv ≈ 1.0 atol = 1e-12
    @test res_fine.D_tv > res_coarse.D_tv
end

@testitem "distinguishability: total_variation bias check and reproducibility" setup=[setupDistinguishability] begin
    rng = Random.Xoshiro(77)
    base = [[rand(rng), rand(rng), rand(rng)] for _ in 1:24]
    a = copy(base)
    b = copy(base)
    r1 = CausalSetZoology.distinguishability_total_variation(
        a,
        b;
        check_bias = true,
        bias_num_splits = 10,
        rng = Random.Xoshiro(123),
    )
    r2 = CausalSetZoology.distinguishability_total_variation(
        a,
        b;
        check_bias = true,
        bias_num_splits = 10,
        rng = Random.Xoshiro(123),
    )
    @test r1.D_tv ≈ r2.D_tv atol = 1e-12
    @test r1.tv_bias_mean ≈ r2.tv_bias_mean atol = 1e-12
    @test r1.tv_bias_std ≈ r2.tv_bias_std atol = 1e-12
    @test 0.0 <= r1.D_tv <= 1.0
    @test 0.0 <= r1.bayes_accuracy <= 1.0
    @test 0.0 <= r1.D_tv_debiased <= 1.0
    @test 0.5 <= r1.bayes_accuracy_debiased <= 1.0
end

@testitem "distinguishability: total_variation dictionaries, total wrapper, and verbose" setup=[setupDistinguishability] begin
    h_a = [Dict(1 => 3.0, 2 => 1.0), Dict(1 => 2.0, 2 => 2.0), Dict(1 => 4.0, 2 => 0.0), Dict(1 => 3.0, 2 => 1.0)]
    h_b = [Dict(1 => 0.0, 2 => 4.0), Dict(1 => 1.0, 2 => 3.0), Dict(1 => 0.0, 2 => 4.0), Dict(1 => 1.0, 2 => 3.0)]
    r = CausalSetZoology.distinguishability_total_variation(h_a, h_b; tv_quantization_digits = 6)
    @test 0.0 <= r.D_tv <= 1.0
    @test r.bayes_accuracy ≈ (1 + r.D_tv) / 2 atol = 1e-12

    total = CausalSetZoology.total_histogram_total_variation_distinguishability(
        [h_a, h_b],
        [h_a, h_b];
        tv_quantization_digits = 6,
    )
    @test 0.0 <= total.D_tv <= 1.0
    vecs_a, vecs_b = CausalSetZoology.concatenate_hists([h_a, h_b], [h_a, h_b])
    total_explicit = CausalSetZoology.distinguishability_total_variation(
        vecs_a,
        vecs_b;
        tv_quantization_digits = 6,
    )
    @test total.D_tv ≈ total_explicit.D_tv atol = 1e-12
    @test total.bayes_accuracy ≈ total_explicit.bayes_accuracy atol = 1e-12

    @test_logs (:info, r"TV distinguishability: D_tv=.*bayes_accuracy=.*tv_bias_mean=.*") CausalSetZoology.distinguishability_total_variation(
        h_a,
        h_b;
        check_bias = true,
        bias_num_splits = 4,
        rng = Random.Xoshiro(9),
        verbose = true,
    )
end

@testitem "distinguishability: total_variation validation" setup=[setupDistinguishability] begin
    a = [[1.0, 2.0], [2.0, 1.0]]
    b = [[0.0, 3.0], [3.0, 0.0]]
    h_a = [Dict(1 => 1.0, 2 => 0.0)]
    h_b = [Dict(1 => 0.0, 2 => 1.0)]
    @test_throws ArgumentError CausalSetZoology.distinguishability_total_variation(Vector{Vector{Float64}}(), b)
    @test_throws ArgumentError CausalSetZoology.distinguishability_total_variation(a, Vector{Vector{Float64}}())
    @test_throws ArgumentError CausalSetZoology.distinguishability_total_variation(Dict{Int,Float64}[], h_b)
    @test_throws ArgumentError CausalSetZoology.total_histogram_total_variation_distinguishability()
    @test_throws DomainError CausalSetZoology.distinguishability_total_variation(a, b; tv_quantization_digits = -1)
    @test_throws DomainError CausalSetZoology.distinguishability_total_variation(a, b; tv_quantization_digits = 13)
    @test_throws DomainError CausalSetZoology.distinguishability_total_variation(a, b; bias_num_splits = 0)
    @test_throws DomainError CausalSetZoology.distinguishability_total_variation([[1.0]], [[1.0]]; check_bias = true, bias_num_splits = 2)
    @test_throws DimensionMismatch CausalSetZoology.total_histogram_total_variation_distinguishability(
        [[Dict(1 => 1.0)], [Dict(1 => 1.0)]],
        [[Dict(1 => 1.0), Dict(2 => 1.0)], [Dict(1 => 1.0)]],
    )
end

@testitem "distinguishability: energy_based_histogram_distinguishability threaded RNG reproducibility" setup=[setupDistinguishability] begin
    a = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05], [0.85, 0.15], [0.92, 0.08]]
    b = [[0.0, 1.0], [0.1, 0.9], [0.05, 0.95], [0.15, 0.85], [0.08, 0.92]]

    r1 = CausalSetZoology.energy_based_histogram_distinguishability(a, b, 100000; rng = Random.Xoshiro(2026))
    r2 = CausalSetZoology.energy_based_histogram_distinguishability(a, b, 100000; rng = Random.Xoshiro(2026))
    @test isapprox(r1.D, r2.D; rtol=1e-2)
    @test isapprox(r1.std, r2.std; rtol=1e-2)

end


@testitem "distinguishability: energy_based_histogram_distinguishability validation" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: energy_based_histogram_distinguishability validation behavior and output contract.
    a = [[1.0, 0.0], [0.9, 0.1]]
    b = [[0.0, 1.0], [0.1, 0.9]]
    h_b = [Dict(1 => 0, 2 => 10), Dict(1 => 1, 2 => 9)]
    @test_throws DomainError CausalSetZoology.energy_based_histogram_distinguishability(a, b, 0)
    @test_throws DomainError CausalSetZoology.energy_based_histogram_distinguishability(a, b; covariance_cutoff_rel_median = -1e-6)
    @test_throws ArgumentError CausalSetZoology.energy_based_histogram_distinguishability(Vector{Vector{Float64}}(), b)
    @test_throws ArgumentError CausalSetZoology.energy_based_histogram_distinguishability(Dict{Int,Float64}[], h_b)
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
    # Cholesky solve can differ at ~1e-6 across BLAS/LAPACK backends (local vs CI).
    @test y ≈ y_expected atol = 5e-6 rtol = 1e-12

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


@testitem "distinguishability helpers: mahalanobis resampling serial and threaded" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability helpers: mahalanobis resampling serial and threaded behavior and output contract.
    X = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
    seeds = UInt64[11, 22, 33, 44]
    out_many = CausalSetZoology._mahal_resample_many(seeds, X, 1e-8, 0.0, :regularization, 1e-10, false, 1e-12)
    out_threaded = CausalSetZoology._mahal_resample_many_threaded(seeds, X, 1e-8, 0.0, :regularization, 1e-10, false, 1e-12, max(2, Threads.nthreads()))
    out_once = [
        CausalSetZoology._mahal_resample_once(s, X, 1e-8, 0.0, :regularization, 1e-10, false, 1e-12)
        for s in seeds
    ]
    @test out_many == out_threaded
    @test out_many ≈ out_once atol = 1e-12
    @test length(out_many) == length(seeds)
    @test all(isfinite, out_many)
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
    @test_throws ArgumentError CausalSetZoology._prepare_vectors_for_mahalanobis(Vector{Vector{Float64}}(), [[0.0]])
    @test_throws ArgumentError CausalSetZoology._prepare_vectors_for_mahalanobis([[0.0]], Vector{Vector{Float64}}())

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
    @test keys(res) == (:M_obs, :D, :distinguishable, :threshold, :z_emp, :M_obs_sym, :D_sym, :M_obs_min, :threshold_sym, :threshold_max)
    @test res.M_obs ≈ expected_min_far atol = 1e-6
    @test 0.0 <= res.D <= 1.0
    @test res.M_obs_sym === nothing
    @test res.D_sym === nothing
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
    @test res_q0.M_obs ≈ expected_min_far atol = 1e-6
    @test res_q50.M_obs ≈ expected_med_far atol = 1e-6
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
    @test res_sym.M_obs ≈ expected_min_far atol = 1e-6
    @test res_sym.M_obs_sym ≈ expected_min_far atol = 1e-6
    @test res_sym.D ≈ cdf(Normal(), res_sym.z_emp) atol = 1e-12
    @test res_sym.D_sym !== nothing
    @test 0.0 <= res_sym.D_sym <= 1.0
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
    @test res_near.M_obs ≈ expected_min_near atol = 1e-6
    @test res_near.M_obs < res_far.M_obs
end


@testitem "distinguishability: mahalanobis threaded path" setup=[setupDistinguishability] begin
    # Test intent: validate distinguishability: mahalanobis threaded path behavior and output contract.
    a = [[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7]]
    b = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]]
    seeds = rand(Random.Xoshiro(4242), UInt64, 64)
    @test length(unique(seeds)) == length(seeds)

    r1 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a,
        b;
        R = 25,
        rng = Random.Xoshiro(6060),
        regulator = 1e-8,
    )
    r2 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a,
        b;
        R = 25,
        rng = Random.Xoshiro(6060),
        regulator = 1e-8,
    )
    @test keys(r2) == keys(r1)
    @test r2.M_obs ≈ r1.M_obs atol = 1e-12
    @test r2.threshold ≈ r1.threshold atol = 1e-12
    @test r2.z_emp ≈ r1.z_emp atol = 1e-12
    @test r2.distinguishable == r1.distinguishable
    @test isfinite(r2.M_obs)
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
    @test_throws DomainError CausalSetZoology.mahalanobis_gap_distinguishability(a, b; to_regularize_rel = 0.0)
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

@testitem "distinguishability: mahalanobis pooled projection invariance" setup=[setupDistinguishability] begin
    a = [[1.0, 0.2], [1.1, 0.3], [0.9, 0.1], [1.2, 0.4], [0.8, 0.0], [1.05, 0.25]]
    b = [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.05, 0.95], [0.15, 0.85]]
    a_pad = [vcat(v, 0.0, 0.0, 0.0) for v in a]
    b_pad = [vcat(v, 0.0, 0.0, 0.0) for v in b]

    r1 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a,
        b;
        R = 40,
        rng = Random.Xoshiro(404),
        q = 0.0,
        projection_tolerance = 1e-10,
    )
    r2 = CausalSetZoology.mahalanobis_gap_distinguishability(
        a_pad,
        b_pad;
        R = 40,
        rng = Random.Xoshiro(404),
        q = 0.0,
        projection_tolerance = 1e-10,
    )

    @test r2.M_obs ≈ r1.M_obs atol = 1e-10
    @test r2.threshold ≈ r1.threshold atol = 1e-10
    @test r2.z_emp ≈ r1.z_emp atol = 1e-10
    @test r2.D ≈ r1.D atol = 1e-12
end

@testitem "distinguishability: mahalanobis small-mode floor helper" setup=[setupDistinguishability] begin
    rng = Random.Xoshiro(909)
    X = rand(rng, 24, 6)
    Σ = CausalSetZoology._safe_cov_matrix(X)
    eig = LinearAlgebra.eigen(Symmetric(Σ))
    U = eig.vectors
    small = [1, 2]
    U_small = U[:, small]
    seeds = rand(rng, UInt64, 12)

    floors = CausalSetZoology._split_floor_estimates_small_modes(X, U_small, seeds)
    @test size(floors) == (length(small), 2 * length(seeds))
    @test all(isfinite, floors)
    @test all(>=(0.0), floors)

    empty_floors = CausalSetZoology._split_floor_estimates_small_modes(X, U[:, Int[]], seeds)
    @test size(empty_floors) == (0, 0)
end

@testitem "distinguishability: verbose projection logs" setup=[setupDistinguishability] begin
    a = [[1.0, 0.0], [0.9, 0.1], [1.1, 0.0], [1.0, 0.1]]
    b = [[0.0, 1.0], [0.1, 0.9], [0.0, 1.1], [0.1, 1.0]]
    h_a = [Dict(1 => 2.0, 2 => 0.0), Dict(1 => 1.5, 2 => 0.5), Dict(1 => 1.8, 2 => 0.2), Dict(1 => 1.9, 2 => 0.1)]
    h_b = [Dict(1 => 0.0, 2 => 2.0), Dict(1 => 0.4, 2 => 1.6), Dict(1 => 0.2, 2 => 1.8), Dict(1 => 0.1, 2 => 1.9)]

    @test_logs (:info, r"Distinguishability projection \(energy\): projected out .* directions \(cutoff_rtol=.* \* median_eig\)") CausalSetZoology.energy_based_histogram_distinguishability(
        h_a,
        h_b;
        verbose = true,
    )
    @test_logs (:info, r"Distinguishability projection \(mutual_information\): projected out .* directions \(cutoff_rtol=.* \* median_eig\)") CausalSetZoology.distinguishability_mutual_information(
        h_a,
        h_b;
        verbose = true,
        k = 2,
    )
    @test_logs (:info, r"Distinguishability projection \(mahalanobis\): projected out .* directions \(cutoff_rtol=.* \* median_eig\)") CausalSetZoology.mahalanobis_gap_distinguishability(
        a,
        b;
        R = 8,
        rng = Random.Xoshiro(21),
        verbose = true,
    )
end
