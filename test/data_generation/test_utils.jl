@testsnippet setupDataGenerationUtils begin
    using Test
    using Statistics
    import CausalSetZoology

    sparse_links_file = joinpath(@__DIR__, "..", "..", "src", "data_generation", "SparseLinksCauset.jl")
    utils_file = joinpath(@__DIR__, "..", "..", "src", "data_generation", "utils.jl")

    if !isdefined(CausalSetZoology, :SparseLinksCauset)
        Base.include(CausalSetZoology, sparse_links_file)
    end
    if !isdefined(CausalSetZoology, :ev_summary)
        Base.include(CausalSetZoology, utils_file)
    end

    # Link graph for 1<2<3 with links 1->2, 2->3.
    function _chain3_sparse_links()
        return CausalSetZoology.SparseLinksCauset(
            Int64(3),
            [Int32[2], Int32[3], Int32[]],
            [Int32[], Int32[1], Int32[2]],
        )
    end
end

# Verifies dense adjacency reconstruction from sparse future-link storage.
@testitem "data_generation utils: dense_future_links basic" setup=[setupDataGenerationUtils] begin
    A = CausalSetZoology.dense_future_links(_chain3_sparse_links())
    @test size(A) == (3, 3)
    @test eltype(A) == Bool
    @test A == BitMatrix([0 1 0; 0 0 1; 0 0 0])
end

# Validates shape/bounds checks for dense adjacency reconstruction.
@testitem "data_generation utils: dense_future_links validation" setup=[setupDataGenerationUtils] begin
    bad_shape = CausalSetZoology.SparseLinksCauset(
        Int64(3),
        [Int32[2], Int32[3]],
        [Int32[], Int32[1], Int32[2]],
    )
    @test_throws DimensionMismatch CausalSetZoology.dense_future_links(bad_shape)

    bad_index = CausalSetZoology.SparseLinksCauset(
        Int64(3),
        [Int32[2], Int32[4], Int32[]],
        [Int32[], Int32[1], Int32[2]],
    )
    @test_throws BoundsError CausalSetZoology.dense_future_links(bad_index)
end

# Verifies histogram sparsification keeps nonzero bins only and preserves counts.
@testitem "data_generation utils: sparse_hist basic" setup=[setupDataGenerationUtils] begin
    @test CausalSetZoology.sparse_hist([0, 2, 0, 3]) == Dict(2 => 2, 4 => 3)
    @test CausalSetZoology.sparse_hist([0, 0, 0]) == Dict{Int, Int}()
    @test CausalSetZoology.sparse_hist([1, 1, 1]) == Dict(1 => 1, 2 => 1, 3 => 1)
end

# Validates count-domain checks for sparse_hist.
@testitem "data_generation utils: sparse_hist validation" setup=[setupDataGenerationUtils] begin
    @test_throws DomainError CausalSetZoology.sparse_hist([0, -1, 2])
end

# Verifies all ev_summary components against explicit expected statistics.
@testitem "data_generation utils: ev_summary basic" setup=[setupDataGenerationUtils] begin
    ev = [-2.0, 0.0, 2.0, 4.0]
    s = CausalSetZoology.ev_summary(ev)

    @test s[1] == ev
    @test s[2] == 1
    @test s[3] == 2.0
    @test s[4] == minimum(ev)
    @test s[5] == maximum(ev)
    @test s[6] == mean(ev)
    @test s[7] == quantile(ev, 0.25)
    @test s[8] == quantile(ev, 0.75)
    @test s[9] == quantile(ev, 0.5)
end

# Verifies ev_summary zero-eigenvalue handling when all values are numerically zero.
@testitem "data_generation utils: ev_summary all-zero edge case" setup=[setupDataGenerationUtils] begin
    ev = [0.0, 0.0, 0.0]
    s = CausalSetZoology.ev_summary(ev)

    @test s[2] == 3
    @test isnan(s[3])
    @test s[4] == 0.0
    @test s[5] == 0.0
    @test s[6] == 0.0
end

# Validates ev_summary input guards.
@testitem "data_generation utils: ev_summary validation" setup=[setupDataGenerationUtils] begin
    @test_throws ArgumentError CausalSetZoology.ev_summary(Float64[])
    @test_throws DomainError CausalSetZoology.ev_summary([0.0, Inf])
end
