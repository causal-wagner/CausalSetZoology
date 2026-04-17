@testsnippet setupSparseLinksCauset begin
    using Test
    import Random
    import Statistics
    import CausalSets
    import CausalSetZoology
    import QuantumGrav

    # Transitively closed 3-chain: 1<2<3 plus closure edge 1->3.
    function _chain3_bitarray()
        future = [
            BitVector([0, 1, 1]),
            BitVector([0, 0, 1]),
            BitVector([0, 0, 0]),
        ]
        past = [
            BitVector([0, 0, 0]),
            BitVector([1, 0, 0]),
            BitVector([1, 1, 0]),
        ]
        return CausalSets.BitArrayCauset(3, future, past)
    end

    # Diamond with transitive closure edge 1->4 present.
    function _diamond_bitarray()
        future = [
            BitVector([0, 1, 1, 1]),
            BitVector([0, 0, 0, 1]),
            BitVector([0, 0, 0, 1]),
            BitVector([0, 0, 0, 0]),
        ]
        past = [
            BitVector([0, 0, 0, 0]),
            BitVector([1, 0, 0, 0]),
            BitVector([1, 0, 0, 0]),
            BitVector([1, 1, 1, 0]),
        ]
        return CausalSets.BitArrayCauset(4, future, past)
    end

    function _assert_same_sparse_links(actual, expected)
        @test actual.atom_count == expected.atom_count
        @test actual.future_links == expected.future_links
        @test actual.past_links == expected.past_links
    end

end

# Verifies direct field constructor stores values without mutation.
@testitem "SparseLinksCauset: direct constructor fields" setup=[setupSparseLinksCauset] begin
    sl = CausalSetZoology.SparseLinksCauset(
        Int64(3),
        [Int32[2], Int32[3], Int32[]],
        [Int32[], Int32[1], Int32[2]],
    )

    @test sl.atom_count == 3
    @test sl.future_links == [Int32[2], Int32[3], Int32[]]
    @test sl.past_links == [Int32[], Int32[1], Int32[2]]
end

# Verifies conversion from closure representation performs transitive reduction.
@testitem "SparseLinksCauset: convert from BitArrayCauset chain reduction" setup=[setupSparseLinksCauset] begin
    cset = _chain3_bitarray()
    sl = CausalSetZoology.SparseLinksCauset(cset)

    @test sl.atom_count == 3
    @test sl.future_links == [Int32[2], Int32[3], Int32[]]
    @test sl.past_links == [Int32[], Int32[1], Int32[2]]
end

# Verifies reduction removes transitive closure edge in a diamond pattern.
@testitem "SparseLinksCauset: convert from BitArrayCauset diamond reduction" setup=[setupSparseLinksCauset] begin
    cset = _diamond_bitarray()
    sl = CausalSetZoology.SparseLinksCauset(cset)

    @test sl.atom_count == 4
    @test sl.future_links[1] == Int32[2, 3]
    @test sl.future_links[2] == Int32[4]
    @test sl.future_links[3] == Int32[4]
    @test sl.future_links[4] == Int32[]
    @test !(Int32(4) in sl.future_links[1])
end

# Verifies antichain conversion yields empty link lists for every node.
@testitem "SparseLinksCauset: convert from BitArrayCauset antichain" setup=[setupSparseLinksCauset] begin
    n = 5
    future = [BitVector(fill(false, n)) for _ in 1:n]
    past = [BitVector(fill(false, n)) for _ in 1:n]
    cset = CausalSets.BitArrayCauset(n, future, past)
    sl = CausalSetZoology.SparseLinksCauset(cset)

    @test sl.atom_count == n
    @test all(isempty, sl.future_links)
    @test all(isempty, sl.past_links)
end

# Verifies manifold+s sprinkling constructor has correct cardinality and reciprocal link bookkeeping.
@testitem "SparseLinksCauset: manifold sprinkling constructor" setup=[setupSparseLinksCauset] begin
    manifold = CausalSets.MinkowskiManifold{2}()
    boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
    sprinkling = CausalSets.generate_sprinkling(manifold, boundary, 12)

    sl = CausalSetZoology.SparseLinksCauset(manifold, sprinkling)
    @test sl.atom_count == 12
    @test length(sl.future_links) == 12
    @test length(sl.past_links) == 12

    # Every future-link edge i->j must appear reciprocally in past_links[j].
    for i in 1:12
        for j in sl.future_links[i]
            @test Int32(i) in sl.past_links[Int(j)]
        end
    end
end

# Verifies SparseLinksCauset edges match CausalSets transitive reduction on a larger sampled causet.
@testitem "SparseLinksCauset: matches ToposortedDAG transitive reduction on Minkowski-256" setup=[setupSparseLinksCauset] begin
    n = 256
    manifold = CausalSets.MinkowskiManifold{2}()
    boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
    sprinkling = CausalSets.generate_sprinkling(manifold, boundary, n)
    cset = CausalSets.BitArrayCauset(manifold, sprinkling)

    tcg = CausalSets.ToposortedDAG(cset.atom_count, copy(cset.future_relations))
    trg = CausalSets.ToposortedDAG(cset.atom_count, [falses(cset.atom_count) for _ in 1:cset.atom_count])
    CausalSets.transitive_reduction!(tcg, trg)

    sl = CausalSetZoology.SparseLinksCauset(cset)
    @test sl.atom_count == n

    for i in 1:n
        expected = falses(n)
        for j in sl.future_links[i]
            expected[Int(j)] = true
        end
        @test trg.edges[i] == expected
    end
end

# Verifies simply-connected polynomial manifold wrapper can generate sparse links directly.
@testitem "SparseLinksCauset: make_polynomial_manifold_cset links mode" setup=[setupSparseLinksCauset] begin
    rng_bit = Random.MersenneTwister(11)
    rng_links = Random.MersenneTwister(11)

    cset, sprinkling_bit, coefs_bit = CausalSetZoology.make_polynomial_manifold_cset(
        16,
        rng_bit,
        4,
        2.0;
        d = 2,
        links = false,
    )
    links, sprinkling_links, coefs_links = CausalSetZoology.make_polynomial_manifold_cset(
        16,
        rng_links,
        4,
        2.0;
        d = 2,
        links = true,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test links isa CausalSetZoology.SparseLinksCauset
    @test sprinkling_bit == sprinkling_links
    @test coefs_bit == coefs_links
    _assert_same_sparse_links(links, CausalSetZoology.SparseLinksCauset(cset))
end

# Verifies simply-connected wrapper matches upstream QuantumGrav path for links=false.
@testitem "SparseLinksCauset: make_polynomial_manifold_cset closure matches upstream" setup=[setupSparseLinksCauset] begin
    rng_expected = Random.MersenneTwister(21)
    rng_actual = Random.MersenneTwister(21)

    expected_cset, expected_sprinkling, expected_coefs = QuantumGrav.make_polynomial_manifold_cset(
        16,
        rng_expected,
        4,
        2.0;
        d = 2,
    )
    actual_cset, actual_sprinkling, actual_coefs = CausalSetZoology.make_polynomial_manifold_cset(
        16,
        rng_actual,
        4,
        2.0;
        d = 2,
        links = false,
    )

    @test actual_cset.future_relations == expected_cset.future_relations
    @test actual_cset.past_relations == expected_cset.past_relations
    @test length(actual_sprinkling) == length(expected_sprinkling)
    for i in eachindex(actual_sprinkling, expected_sprinkling)
        @test actual_sprinkling[i][1] ≈ expected_sprinkling[i][1] atol = 1e-6
        @test actual_sprinkling[i][2] ≈ expected_sprinkling[i][2] atol = 1e-6
    end
    @test actual_coefs == expected_coefs
end

# Verifies branched polynomial manifold wrapper can generate sparse links directly.
@testitem "SparseLinksCauset: make_polynomial_manifold_cset_with_nontrivial_topology links mode" setup=[setupSparseLinksCauset] begin
    Random.seed!(2026)
    cset, sprinkling_bit, branch_info_bit, coefs_bit =
        CausalSetZoology.make_polynomial_manifold_cset_with_nontrivial_topology(
            18,
            1,
            0,
            Random.MersenneTwister(12),
            4,
            2.0;
            links = false,
        )
    Random.seed!(2026)
    links, sprinkling_links, branch_info_links, coefs_links =
        CausalSetZoology.make_polynomial_manifold_cset_with_nontrivial_topology(
            18,
            1,
            0,
            Random.MersenneTwister(12),
            4,
            2.0;
            links = true,
        )

    @test cset isa CausalSets.BitArrayCauset
    @test links isa CausalSetZoology.SparseLinksCauset
    @test sprinkling_links == sprinkling_bit
    @test branch_info_links == branch_info_bit
    @test coefs_links == coefs_bit
    _assert_same_sparse_links(links, CausalSetZoology.SparseLinksCauset(cset))
end

# Verifies grid wrapper can generate sparse links directly while preserving coordinates.
@testitem "SparseLinksCauset: create_grid_causet_in_boundary_2D links mode" setup=[setupSparseLinksCauset] begin
    boundary = CausalSets.BoxBoundary{2}(((0.0, -0.5), (1.0, 0.5)))
    manifold = CausalSets.MinkowskiManifold{2}()
    rng_bit = Random.MersenneTwister(13)
    rng_links = Random.MersenneTwister(13)

    cset, ok_bit, coords_bit = CausalSetZoology.create_grid_causet_in_boundary_2D(
        12,
        "quadratic",
        boundary,
        manifold;
        b = 1.0,
        gamma_deg = 10.0,
        rotate_deg = 0.0,
        rng = rng_bit,
        links = false,
    )
    links, ok_links, coords_links = CausalSetZoology.create_grid_causet_in_boundary_2D(
        12,
        "quadratic",
        boundary,
        manifold;
        b = 1.0,
        gamma_deg = 10.0,
        rotate_deg = 0.0,
        rng = rng_links,
        links = true,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test links isa CausalSetZoology.SparseLinksCauset
    @test ok_bit == true
    @test ok_links == true
    @test coords_bit == coords_links
    _assert_same_sparse_links(links, CausalSetZoology.SparseLinksCauset(cset))
end

# Verifies grid wrapper closure mode matches upstream QuantumGrav path.
@testitem "SparseLinksCauset: create_grid_causet_in_boundary_2D closure matches upstream" setup=[setupSparseLinksCauset] begin
    boundary = CausalSets.BoxBoundary{2}(((0.0, -0.5), (1.0, 0.5)))
    manifold = CausalSets.MinkowskiManifold{2}()
    rng_expected = Random.MersenneTwister(31)
    rng_actual = Random.MersenneTwister(31)

    expected_cset, expected_ok, expected_coords = QuantumGrav.create_grid_causet_in_boundary_2D(
        12,
        "quadratic",
        boundary,
        manifold;
        b = 1.0,
        gamma_deg = 10.0,
        rotate_deg = 0.0,
        rng = rng_expected,
    )
    actual_cset, actual_ok, actual_coords = CausalSetZoology.create_grid_causet_in_boundary_2D(
        12,
        "quadratic",
        boundary,
        manifold;
        b = 1.0,
        gamma_deg = 10.0,
        rotate_deg = 0.0,
        rng = rng_actual,
        links = false,
    )

    @test actual_ok == expected_ok
    @test actual_coords == expected_coords
    @test actual_cset.future_relations == expected_cset.future_relations
    @test actual_cset.past_relations == expected_cset.past_relations
end

# Verifies quasicrystal wrapper can generate sparse links directly from the same crystal.
@testitem "SparseLinksCauset: create_Minkowski_quasicrystal_cset links mode" setup=[setupSparseLinksCauset] begin
    crystal = QuantumGrav.quasicrystal(2.0)
    center = (0.5, 0.5)

    cset = CausalSetZoology.create_Minkowski_quasicrystal_cset(
        12,
        center;
        crystal = crystal,
        exact_size = true,
        deviation_from_mean_size = 0.1,
        max_iter = 100,
        links = false,
    )
    links = CausalSetZoology.create_Minkowski_quasicrystal_cset(
        12,
        center;
        crystal = crystal,
        exact_size = true,
        deviation_from_mean_size = 0.1,
        max_iter = 100,
        links = true,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test links isa CausalSetZoology.SparseLinksCauset
    _assert_same_sparse_links(links, CausalSetZoology.SparseLinksCauset(cset))
end

# Verifies quasicrystal wrapper closure mode matches upstream QuantumGrav path.
@testitem "SparseLinksCauset: create_Minkowski_quasicrystal_cset closure matches upstream" setup=[setupSparseLinksCauset] begin
    crystal = QuantumGrav.quasicrystal(2.0)
    center = (0.5, 0.5)

    expected = QuantumGrav.create_Minkowski_quasicrystal_cset(
        12,
        center;
        crystal = crystal,
        exact_size = true,
        deviation_from_mean_size = 0.1,
        max_iter = 100,
    )
    actual = CausalSetZoology.create_Minkowski_quasicrystal_cset(
        12,
        center;
        crystal = crystal,
        exact_size = true,
        deviation_from_mean_size = 0.1,
        max_iter = 100,
        links = false,
    )

    @test actual.future_relations == expected.future_relations
    @test actual.past_relations == expected.past_relations
end

# Verifies new wrapper methods retain upstream validation behavior.
@testitem "SparseLinksCauset: wrapper validation" setup=[setupSparseLinksCauset] begin
    boundary = CausalSets.BoxBoundary{2}(((0.0, -0.5), (1.0, 0.5)))
    manifold = CausalSets.MinkowskiManifold{2}()

    @test_throws ArgumentError CausalSetZoology.make_polynomial_manifold_cset(
        0,
        Random.MersenneTwister(1),
        4,
        2.0;
        links = true,
    )
    @test_throws ArgumentError CausalSetZoology.make_polynomial_manifold_cset_with_nontrivial_topology(
        18,
        -1,
        0,
        Random.MersenneTwister(1),
        4,
        2.0;
        links = true,
    )
    @test_throws ArgumentError CausalSetZoology.create_grid_causet_in_boundary_2D(
        0,
        "quadratic",
        boundary,
        manifold;
        links = true,
    )
    @test_throws ErrorException CausalSetZoology.create_Minkowski_quasicrystal_cset(
        12,
        (0.5, 0.5);
        links = true,
    )
end
