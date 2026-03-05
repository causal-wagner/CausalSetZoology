@testsnippet setupUtils begin
    using Test
    using Statistics
    using Random
end

# normalize_hists (plain overload)
@testitem "utils: normalize_hists plain" setup=[setupUtils] begin
    h = [[Dict(1 => 2, 2 => 2), Dict(1 => 1, 2 => 3)]]

    # Probability mode normalizes each histogram to unit mass.
    np = CausalSetZoology.normalize_hists(h; normalization = :probability)
    @test sum(values(np[1][1])) ≈ 1.0
    @test sum(values(np[1][2])) ≈ 1.0

    # Max mode normalizes by histogram maximum.
    nm = CausalSetZoology.normalize_hists(h; normalization = :max)
    @test maximum(values(nm[1][1])) ≈ 1.0
    @test maximum(values(nm[1][2])) ≈ 1.0

    # Numeric mode divides by explicit constant.
    nc = CausalSetZoology.normalize_hists(h; normalization = 2.0)
    @test nc[1][1][1] ≈ 1.0
    @test nc[1][1][2] ≈ 1.0
end

@testitem "utils: normalize_hists plain validation" setup=[setupUtils] begin
    # Unsupported symbol mode.
    @test_throws ArgumentError CausalSetZoology.normalize_hists([[Dict(1 => 1)]]; normalization = :unknown)

    # Invalid numeric normalization constants.
    @test_throws DomainError CausalSetZoology.normalize_hists([[Dict(1 => 1)]]; normalization = 0.0)
    @test_throws DomainError CausalSetZoology.normalize_hists([[Dict(1 => 1)]]; normalization = Inf)
    @test_throws DomainError CausalSetZoology.normalize_hists([[Dict(1 => 1)]]; normalization = NaN)

    # Zero denominator in symbolic modes.
    @test_throws DomainError CausalSetZoology.normalize_hists([[Dict{Int,Int}()]]; normalization = :probability)
    @test_throws DomainError CausalSetZoology.normalize_hists([[Dict(1 => 0, 2 => 0)]]; normalization = :max)
end

# normalize_hists (histogram+scalar overload)
@testitem "utils: normalize_hists scalar" setup=[setupUtils] begin
    hs = [[(Dict(1 => 2, 2 => 2), 1.0), (Dict(1 => 1, 2 => 3), 2.0)]]

    # Without binning, keys are preserved.
    ns = CausalSetZoology.normalize_hists(hs; normalization = :probability)
    @test length(ns[1]) == 2
    @test ns[1][1][2] == 1.0
    @test ns[1][2][2] == 2.0

    # With scalar binning, keys become bin centers.
    ns_bin = CausalSetZoology.normalize_hists(hs; normalization = :probability, num_bins = 2)
    @test length(ns_bin[1]) == 2
    @test all(p -> p[2] isa Real, ns_bin[1])

    # Pre-binned scalars:
    # - same bin count => pass-through of existing centers
    # - fewer bins => regular rebinning
    hs_pre = [[
        (Dict(1 => 2, 2 => 2), 1.0),
        (Dict(1 => 3, 2 => 1), 1.0),
        (Dict(1 => 1, 2 => 3), 2.0),
        (Dict(1 => 2, 2 => 2), 2.0),
        (Dict(1 => 4, 2 => 0), 3.0),
    ]]
    ns_same = CausalSetZoology.normalize_hists(hs_pre; normalization = :probability, num_bins = 3)
    @test sort(unique(last.(ns_same[1]))) == [1.0, 2.0, 3.0]

    ns_smaller = CausalSetZoology.normalize_hists(hs_pre; normalization = :probability, num_bins = 2)
    @test sort(unique(last.(ns_smaller[1]))) == [1.5, 2.5]

    # Empty input returns empty output.
    @test CausalSetZoology.normalize_hists(Vector{Vector{Tuple{Dict{Int,Int},Float64}}}()) == Vector{Vector{Tuple{Dict{Int,Float64},Real}}}()
end

@testitem "utils: normalize_hists scalar validation" setup=[setupUtils] begin
    hs = [[(Dict(1 => 1), 1.0)]]

    @test_throws DomainError CausalSetZoology.normalize_hists(hs; num_bins = 0)
    @test_throws ArgumentError CausalSetZoology.normalize_hists(hs; normalization = :bad)
    @test_throws DomainError CausalSetZoology.normalize_hists(hs; normalization = 0.0)
    @test_throws DomainError CausalSetZoology.normalize_hists(hs; normalization = Inf)
    @test_throws DomainError CausalSetZoology.normalize_hists(hs; normalization = NaN)

    hs_pre = [[
        (Dict(1 => 1), 1.0),
        (Dict(1 => 1), 1.0),
        (Dict(1 => 1), 2.0),
        (Dict(1 => 1), 2.0),
        (Dict(1 => 1), 3.0),
    ]]
    @test_throws DomainError CausalSetZoology.normalize_hists(hs_pre; num_bins = 4)

    # Zero denominator in symbolic mode.
    @test_throws DomainError CausalSetZoology.normalize_hists([[(Dict(1 => 0), 1.0)]]; normalization = :max)
    @test_throws DomainError CausalSetZoology.normalize_hists([[(Dict(1 => 0), 1.0)]]; normalization = :probability)
end

# densify_hists
@testitem "utils: densify_hists values" setup=[setupUtils] begin
    dense = CausalSetZoology.densify_hists([Dict(0 => 1.0, 2 => 2.0), Dict(1 => 1.0)])
    @test size(dense) == (2, 3)
    @test dense[1, :] == [1.0, 0.0, 2.0]
    @test dense[2, :] == [0.0, 1.0, 0.0]

    dense1 = CausalSetZoology.densify_hists([Dict(1 => 2.0, 3 => 4.0)])
    @test size(dense1) == (1, 3)
    @test dense1[1, :] == [2.0, 0.0, 4.0]
end

@testitem "utils: densify_hists validation" setup=[setupUtils] begin
    @test_throws ArgumentError CausalSetZoology.densify_hists(Dict{Int,Float64}[])
    @test_throws ArgumentError CausalSetZoology.densify_hists([Dict{Int,Float64}()])
end

# histogram_to_dense_pair
@testitem "utils: histogram_to_dense_pair dict mode" setup=[setupUtils] begin
    obs = [
        [Dict(1 => 0.5, 3 => 0.5), Dict(2 => 1.0)],
        [Dict(1 => 1.0), Dict(3 => 1.0)],
    ]
    A, B = CausalSetZoology.histogram_to_dense_pair(obs, 1)
    @test size(A, 1) == 2
    @test size(B, 1) == 2
    @test size(A, 2) == size(B, 2)
    @test A[1, :] == [0.5, 0.0, 0.5]
    @test A[2, :] == [0.0, 1.0, 0.0]
    @test B[1, :] == [1.0, 0.0, 0.0]
    @test B[2, :] == [0.0, 0.0, 1.0]
end

@testitem "utils: histogram_to_dense_pair vector mode" setup=[setupUtils] begin
    obs = [
        [[1.0, 2.0], [3.0]],
        [[4.0], [5.0, 6.0, 7.0]],
    ]
    A, B = CausalSetZoology.histogram_to_dense_pair(obs, 2)
    @test size(A) == (2, 3)
    @test size(B) == (2, 3)
    @test A[1, :] == [1.0, 2.0, 0.0]
    @test A[2, :] == [3.0, 0.0, 0.0]
    @test B[1, :] == [4.0, 0.0, 0.0]
    @test B[2, :] == [5.0, 6.0, 7.0]
end

@testitem "utils: histogram_to_dense_pair validation" setup=[setupUtils] begin
    # Verifies shape/type guards for two-class observable inputs.
    @test_throws ArgumentError CausalSetZoology.histogram_to_dense_pair([[Dict(1 => 1.0)]], 1)
    @test_throws ArgumentError CausalSetZoology.histogram_to_dense_pair([Dict{Int,Float64}[], [Dict(1 => 1.0)]], 1)
    @test_throws ArgumentError CausalSetZoology.histogram_to_dense_pair([[[1.0]], Vector{Vector{Float64}}()], 1)
    @test_throws ArgumentError CausalSetZoology.histogram_to_dense_pair([[Dict(1 => 1.0)], [[1.0]]], 1)
end

@testitem "utils: histogram_to_dense_pair validation non-real vector values" setup=[setupUtils] begin
    # Verifies explicit non-real rejection for vector-mode samples in class A/B.
    bad_a = [[[1.0, "x"]], [[1.0]]]
    bad_b = [[[1.0]], [[1.0, "x"]]]
    @test_throws ArgumentError CausalSetZoology.histogram_to_dense_pair(bad_a, 3)
    @test_throws ArgumentError CausalSetZoology.histogram_to_dense_pair(bad_b, 3)
end

# concatenate_hists
@testitem "utils: concatenate_hists mixed observables" setup=[setupUtils] begin
    obs_dict = [
        [Dict(1 => 0.5, 2 => 0.5), Dict(2 => 1.0)],
        [Dict(1 => 1.0), Dict(2 => 1.0)],
    ]
    obs_vec = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0], [6.0, 7.0]],
    ]
    A, B = CausalSetZoology.concatenate_hists(obs_dict, obs_vec)
    @test length(A) == 2
    @test length(B) == 2
    @test length(A[1]) == length(B[1]) == 4
    @test A[1] == [0.5, 0.5, 1.0, 2.0]
    @test A[2] == [0.0, 1.0, 3.0, 4.0]
    @test B[1] == [1.0, 0.0, 5.0, 0.0]
    @test B[2] == [0.0, 1.0, 6.0, 7.0]
end

@testitem "utils: concatenate_hists validation" setup=[setupUtils] begin
    # Verifies top-level validation for empty input and cross-observable sample-count mismatch.
    obs = [[Dict(1 => 1.0)], [Dict(1 => 1.0)]]
    @test_throws ArgumentError CausalSetZoology.concatenate_hists()
    @test_throws DimensionMismatch CausalSetZoology.concatenate_hists(obs, [[Dict(1 => 1.0), Dict(2 => 1.0)], [Dict(1 => 1.0)]])
end

@testitem "utils: concatenate_hists validation propagates per-observable errors" setup=[setupUtils] begin
    # Verifies invalid observable content is rejected through histogram_to_dense_pair.
    obs_ok = [[Dict(1 => 1.0)], [Dict(1 => 1.0)]]
    obs_bad = [[Dict(1 => 1.0)], [[1.0]]]
    @test_throws ArgumentError CausalSetZoology.concatenate_hists(obs_ok, obs_bad)
end

# join_histograms (plain overload)
@testitem "utils: join_histograms plain" setup=[setupUtils] begin
    h1 = [
        [Dict(1 => 1, 2 => 2), Dict(1 => 3)],
        [Dict(1 => 4), Dict(2 => 5)],
    ]
    h2 = [
        [Dict(1 => 10, 3 => 1), Dict(1 => 7, 2 => 2)],
        [Dict(1 => 6, 2 => 1), Dict(2 => 8)],
    ]

    joined = CausalSetZoology.join_histograms(Vector{Vector{Vector{Dict}}}([h1, h2]))
    @test joined == [
        [Dict(1 => 11, 2 => 2, 3 => 1), Dict(1 => 10, 2 => 2)],
        [Dict(1 => 10, 2 => 1), Dict(2 => 13)],
    ]

    @test CausalSetZoology.join_histograms(Vector{Vector{Vector{Dict}}}()) == Vector{Vector{Dict}}()
end

# join_histograms (histogram+scalar overload)
@testitem "utils: join_histograms scalar" setup=[setupUtils] begin
    hs1 = [[(Dict(1 => 1.0), 10.0), (Dict(2 => 2.0), 20.0)]]
    hs2 = [[(Dict(1 => 3.0, 2 => 1.0), 10.0), (Dict(2 => 4.0), 20.0)]]

    joined = CausalSetZoology.join_histograms(Vector{Vector{Vector{Tuple{Dict,Real}}}}([hs1, hs2]))
    @test joined == [[(Dict(1 => 4.0, 2 => 1.0), 10.0), (Dict(2 => 6.0), 20.0)]]

    @test CausalSetZoology.join_histograms(Vector{Vector{Vector{Tuple{Dict,Real}}}}()) == Vector{Vector{Tuple{Dict,Float64}}}()
end

@testitem "utils: join_histograms validation" setup=[setupUtils] begin
    plain_a = [[Dict(1 => 1)::Dict]]
    plain_b_bad_groups = [[Dict(1 => 2)::Dict], [Dict(1 => 3)::Dict]]
    @test_throws DimensionMismatch CausalSetZoology.join_histograms(Vector{Vector{Vector{Dict}}}([plain_a, plain_b_bad_groups]))
    plain_b_bad_hists = [[Dict(1 => 2)::Dict, Dict(1 => 3)::Dict]]
    @test_throws DimensionMismatch CausalSetZoology.join_histograms(Vector{Vector{Vector{Dict}}}([plain_a, plain_b_bad_hists]))

    scalar_a = [[(Dict(1 => 1.0)::Dict, 5.0::Real)]]
    scalar_b_bad_groups = [[(Dict(1 => 2.0)::Dict, 5.0::Real)], [(Dict(1 => 3.0)::Dict, 5.0::Real)]]
    @test_throws DimensionMismatch CausalSetZoology.join_histograms(Vector{Vector{Vector{Tuple{Dict,Real}}}}([scalar_a, scalar_b_bad_groups]))
    scalar_b_bad_hists = [[(Dict(1 => 2.0)::Dict, 5.0::Real), (Dict(1 => 3.0)::Dict, 5.0::Real)]]
    @test_throws DimensionMismatch CausalSetZoology.join_histograms(Vector{Vector{Vector{Tuple{Dict,Real}}}}([scalar_a, scalar_b_bad_hists]))

    # Scalar mismatch for same (j,k) position.
    t1 = (Dict{Int,Float64}(1 => 1.0)::Dict, 5.0::Real)
    t2 = (Dict{Int,Float64}(1 => 2.0)::Dict, 6.0::Real)
    @test_throws DomainError CausalSetZoology.join_histograms(Vector{Vector{Vector{Tuple{Dict,Real}}}}([[[t1]], [[t2]]]))
end

# average_histogram_with_std (plain overload)
@testitem "utils: average_histogram_with_std plain" setup=[setupUtils] begin
    meanh, stdh = CausalSetZoology.average_histogram_with_std([
        Dict(1 => 1, 2 => 3),
        Dict(1 => 3, 2 => 1),
    ])
    @test meanh == [2.0, 2.0]
    @test all(x -> x ≥ 0, stdh)

    # Empty input path.
    me, se = CausalSetZoology.average_histogram_with_std(Dict{Int,Int}[])
    @test isempty(me)
    @test isempty(se)
end

# average_vectors_with_std (plain overload)
@testitem "utils: average_vectors_with_std plain" setup=[setupUtils] begin
    meanv, stdv = CausalSetZoology.average_vectors_with_std([[1.0, 2.0], [3.0, 4.0]])
    @test meanv == [2.0, 3.0]
    @test all(x -> x > 0, stdv)

    # Nested input branch.
    mean_nested, std_nested = CausalSetZoology.average_vectors_with_std([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    @test length(mean_nested) == 2
    @test all(x -> x >= 0, std_nested)

    # Empty input path.
    me, se = CausalSetZoology.average_vectors_with_std(Vector{Vector{Float64}}())
    @test isempty(me)
    @test isempty(se)
end

@testitem "utils: average_vectors_with_std plain validation" setup=[setupUtils] begin
    @test_throws ArgumentError CausalSetZoology.average_vectors_with_std([[1.0], [1.0, 2.0]])
    @test_throws ArgumentError CausalSetZoology.average_vectors_with_std(Any[1.0, 2.0])
end

# average_vectors_with_std (vector+scalar overload)
@testitem "utils: average_vectors_with_std scalar" setup=[setupUtils] begin
    grouped = CausalSetZoology.average_vectors_with_std([
        ([1.0, 2.0], 1.0),
        ([2.0, 3.0], 1.0),
        ([3.0, 4.0], 2.0),
    ])
    @test length(grouped) == 2
    @test grouped[1][1] == 1.0
    @test grouped[2][1] == 2.0

    # Binned-scalar branch.
    grouped_b = CausalSetZoology.average_vectors_with_std([
        ([1.0, 2.0], 1.0),
        ([2.0, 3.0], 2.0),
        ([3.0, 4.0], 3.0),
    ]; num_bins = 2)
    @test length(grouped_b) == 2

    # Empty input path.
    @test isempty(CausalSetZoology.average_vectors_with_std(Tuple{Vector{Float64},Float64}[]))
end

@testitem "utils: average_vectors_with_std scalar validation" setup=[setupUtils] begin
    @test_throws DomainError CausalSetZoology.average_vectors_with_std([([1.0], 1.0)]; num_bins = 0)
end

# average_histogram_with_std (histogram+scalar overload)
@testitem "utils: average_histogram_with_std scalar" setup=[setupUtils] begin
    grouped_hist = CausalSetZoology.average_histogram_with_std([
        (Dict(1 => 1, 2 => 2), 1.0),
        (Dict(1 => 2, 2 => 1), 1.0),
    ])
    @test length(grouped_hist) == 1

    # Binned-scalar branch.
    grouped_hist_b = CausalSetZoology.average_histogram_with_std([
        (Dict(1 => 1, 2 => 2), 1.0),
        (Dict(1 => 2, 2 => 1), 2.0),
        (Dict(1 => 2, 2 => 2), 3.0),
    ]; num_bins = 2)
    @test length(grouped_hist_b) == 2

    # Empty input path.
    @test isempty(CausalSetZoology.average_histogram_with_std(Tuple{Dict{Int,Int},Float64}[]))
end

@testitem "utils: average_histogram_with_std scalar validation" setup=[setupUtils] begin
    @test_throws DomainError CausalSetZoology.average_histogram_with_std([(Dict(1 => 1), 1.0)]; num_bins = 0)
end

# replace_zeros
@testitem "utils: replace_zeros" setup=[setupUtils] begin
    rz = CausalSetZoology.replace_zeros([0.0, 2.0, 0.0]; ϵ = 1e-2)
    @test rz[2] == 2.0
    @test rz[1] > 0

    # No positive entries: unchanged.
    z = [0.0, 0.0]
    @test CausalSetZoology.replace_zeros(z; ϵ = 1e-3) == z

    @test_throws DomainError CausalSetZoology.replace_zeros([0.0, 1.0]; ϵ = 0.0)
end
