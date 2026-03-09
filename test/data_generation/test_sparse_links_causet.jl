@testsnippet setupSparseLinksCauset begin
    using Test
    import CausalSets
    import CausalSetZoology

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
