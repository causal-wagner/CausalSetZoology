@testsnippet setupDataGenerationGraphObservables begin
    using Test
    using LinearAlgebra
    import CausalSetZoology
    import CausalSets

    sparse_links_file = joinpath(@__DIR__, "..", "..", "src", "data_generation", "SparseLinksCauset.jl")
    graph_obs_file = joinpath(@__DIR__, "..", "..", "src", "data_generation", "graph_observables.jl")

    if !isdefined(CausalSetZoology, :SparseLinksCauset)
        Base.include(CausalSetZoology, sparse_links_file)
    end
    if !isdefined(CausalSetZoology, :normalized_lap_eigs_symmetrized_links)
        Base.include(CausalSetZoology, graph_obs_file)
    end

    # Transitively closed 3-chain: 1<2<3 with closure edge 1->3.
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

    # Link graph (transitive reduction) for 1<2<3.
    function _chain3_sparse_links()
        return CausalSetZoology.SparseLinksCauset(
            Int64(3),
            [Int32[2], Int32[3], Int32[]],
            [Int32[], Int32[1], Int32[2]],
        )
    end
end

# Verifies strict-upper to strict-lower copy and in-place return contract.
@testitem "graph_observables: symmetrize_strictly_upper_triangular! basic" setup=[setupDataGenerationGraphObservables] begin
    M = [10.0 2.0 3.0; 7.0 20.0 4.0; 8.0 9.0 30.0]
    out = CausalSetZoology.symmetrize_strictly_upper_triangular!(M)
    @test out === M
    @test M == [10.0 2.0 3.0; 2.0 20.0 4.0; 3.0 4.0 30.0]
end

# Verifies non-square guard.
@testitem "graph_observables: symmetrize_strictly_upper_triangular! validation" setup=[setupDataGenerationGraphObservables] begin
    @test_throws DimensionMismatch CausalSetZoology.symmetrize_strictly_upper_triangular!(ones(2, 3))
end

# Verifies known normalized-Laplacian spectrum and in-place mutation.
@testitem "graph_observables: sym_norm_lap_eigs! basic" setup=[setupDataGenerationGraphObservables] begin
    W = [0.0 1.0; 1.0 0.0]
    vals = sort(CausalSetZoology.sym_norm_lap_eigs!(W))
    @test vals ≈ [0.0, 2.0] atol = 1e-12
    @test W ≈ [1.0 -1.0; -1.0 1.0] atol = 1e-12
end

# Verifies matrix-shape and symmetry guards.
@testitem "graph_observables: sym_norm_lap_eigs! validation" setup=[setupDataGenerationGraphObservables] begin
    @test_throws DimensionMismatch CausalSetZoology.sym_norm_lap_eigs!(ones(2, 3))
    @test_throws ArgumentError CausalSetZoology.sym_norm_lap_eigs!([0.0 1.0; 0.0 0.0])
end

# Verifies end-to-end link reduction + symmetrization + normalized-Laplacian path.
@testitem "graph_observables: normalized_lap_eigs_symmetrized_links basic" setup=[setupDataGenerationGraphObservables] begin
    vals = sort(CausalSetZoology.normalized_lap_eigs_symmetrized_links(_chain3_bitarray()))
    @test vals ≈ [0.0, 1.0, 2.0] atol = 1e-12
end

# Verifies degree counts for closure-based BitArrayCauset path.
@testitem "graph_observables: degrees bitarray" setup=[setupDataGenerationGraphObservables] begin
    in_deg, out_deg = CausalSetZoology.degrees(_chain3_bitarray())
    @test in_deg == Int32[0, 1, 2]
    @test out_deg == Int32[2, 1, 0]
end

# Verifies degree counts for sparse-link path.
@testitem "graph_observables: degrees sparse links" setup=[setupDataGenerationGraphObservables] begin
    in_deg, out_deg = CausalSetZoology.degrees(_chain3_sparse_links())
    @test in_deg == Int32[0, 1, 1]
    @test out_deg == Int32[1, 1, 0]
end

# Verifies relation-density scalar on fully related and empty-relation cases.
@testitem "graph_observables: connectivity" setup=[setupDataGenerationGraphObservables] begin
    cset_chain = _chain3_bitarray()
    @test CausalSetZoology.connectivity(cset_chain) ≈ 1.0 atol = 1e-12

    n = 4
    future = [BitVector(fill(false, n)) for _ in 1:n]
    past = [BitVector(fill(false, n)) for _ in 1:n]
    cset_antichain = CausalSets.BitArrayCauset(n, future, past)
    @test CausalSetZoology.connectivity(cset_antichain) ≈ 0.0 atol = 1e-12
end

# Verifies domain guard for undersized causal sets.
@testitem "graph_observables: connectivity validation" setup=[setupDataGenerationGraphObservables] begin
    cset_singleton = CausalSets.BitArrayCauset(1, [BitVector([0])], [BitVector([0])])
    @test_throws DomainError CausalSetZoology.connectivity(cset_singleton)
end

# Verifies maximum path-length logic on closure edges.
@testitem "graph_observables: height bitarray" setup=[setupDataGenerationGraphObservables] begin
    cset = _chain3_bitarray()
    @test CausalSetZoology.height(cset, 1) == 2
    @test CausalSetZoology.height(cset, 2) == 1
    @test CausalSetZoology.height(cset, 3) == 0
end

# Verifies bounds guard on closure-edge path.
@testitem "graph_observables: height bitarray validation" setup=[setupDataGenerationGraphObservables] begin
    cset = _chain3_bitarray()
    @test_throws BoundsError CausalSetZoology.height(cset, 0)
    @test_throws BoundsError CausalSetZoology.height(cset, 4)
end

# Verifies maximum path-length logic on sparse-link edges.
@testitem "graph_observables: height sparse links" setup=[setupDataGenerationGraphObservables] begin
    links = _chain3_sparse_links()
    @test CausalSetZoology.height(links, 1) == 2
    @test CausalSetZoology.height(links, 2) == 1
    @test CausalSetZoology.height(links, 3) == 0
end

# Verifies bounds guard on sparse-link path.
@testitem "graph_observables: height sparse links validation" setup=[setupDataGenerationGraphObservables] begin
    links = _chain3_sparse_links()
    @test_throws BoundsError CausalSetZoology.height(links, 0)
    @test_throws BoundsError CausalSetZoology.height(links, 4)
end
