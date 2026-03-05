@testsnippet setupDataCreationUtils begin
    using Test
    using LinearAlgebra
    import CausalSetZoology
    import CausalSets

    utils_file = joinpath(@__DIR__, "..", "..", "src", "data_generation", "utils.jl")
    if !isdefined(CausalSetZoology, :sym_norm_lap_eigs!)
        Base.include(CausalSetZoology, utils_file)
    end

    # Transitively closed 3-chain: 1<2<3 with closure edge 1->3.
    function _chain3_cset()
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
end

# Covers transitive_reduction! core behavior and overwrite contract.
@testitem "data_creation utils: transitive_reduction! basic" setup=[setupDataCreationUtils] begin
    tcg = _chain3_cset()
    trg = CausalSets.empty_graph(3)
    trg.edges[1] .= true
    trg.edges[2] .= true
    trg.edges[3] .= true

    out = CausalSets.transitive_reduction!(tcg, trg)
    @test out === trg
    @test trg.edges[1] == BitVector([0, 1, 0]) # 1->3 removed as transitive
    @test trg.edges[2] == BitVector([0, 0, 1])
    @test trg.edges[3] == BitVector([0, 0, 0])
end

# Covers transitive_reduction! input-size validation.
@testitem "data_creation utils: transitive_reduction! validation" setup=[setupDataCreationUtils] begin
    tcg = _chain3_cset()
    trg_bad = CausalSets.empty_graph(2)
    @test_throws DimensionMismatch CausalSets.transitive_reduction!(tcg, trg_bad)
end

# Covers strict-upper to strict-lower copy and in-place return.
@testitem "data_creation utils: symmetrize_strictly_upper_triangular! basic" setup=[setupDataCreationUtils] begin
    M = [10.0 2.0 3.0; 7.0 20.0 4.0; 8.0 9.0 30.0]
    out = CausalSetZoology.symmetrize_strictly_upper_triangular!(M)
    @test out === M
    @test M == [10.0 2.0 3.0; 2.0 20.0 4.0; 3.0 4.0 30.0]
end

# Covers non-square guard for symmetrize_strictly_upper_triangular!.
@testitem "data_creation utils: symmetrize_strictly_upper_triangular! validation" setup=[setupDataCreationUtils] begin
    @test_throws DimensionMismatch CausalSetZoology.symmetrize_strictly_upper_triangular!(ones(2, 3))
end

# Covers normalized Laplacian on a known graph and verifies in-place mutation.
@testitem "data_creation utils: sym_norm_lap_eigs! basic" setup=[setupDataCreationUtils] begin
    W = [0.0 1.0; 1.0 0.0]
    vals = sort(CausalSetZoology.sym_norm_lap_eigs!(W))
    @test vals ≈ [0.0, 2.0] atol = 1e-12
    @test W ≈ [1.0 -1.0; -1.0 1.0] atol = 1e-12
end

# Covers matrix-shape and symmetry guards for sym_norm_lap_eigs!.
@testitem "data_creation utils: sym_norm_lap_eigs! validation" setup=[setupDataCreationUtils] begin
    @test_throws DimensionMismatch CausalSetZoology.sym_norm_lap_eigs!(ones(2, 3))
    @test_throws ArgumentError CausalSetZoology.sym_norm_lap_eigs!([0.0 1.0; 0.0 0.0])
end

# Covers end-to-end link reduction + symmetrization + normalized-Laplacian spectrum.
@testitem "data_creation utils: normalized_lap_eigs_symmetrized_links basic" setup=[setupDataCreationUtils] begin
    vals = sort(CausalSetZoology.normalized_lap_eigs_symmetrized_links(_chain3_cset()))
    @test vals ≈ [0.0, 1.0, 2.0] atol = 1e-12
end
