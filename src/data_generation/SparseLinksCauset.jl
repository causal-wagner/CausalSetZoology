"""
    SparseLinksCauset

Sparse causal set representation that stores only link relations
(transitive reduction), not full transitive closure.
"""
struct SparseLinksCauset <: CausalSets.AbstractCauset
    atom_count::Int64
    future_links::Vector{Vector{Int32}} # future link neighbors
    past_links::Vector{Vector{Int32}}   # past link neighbors
end

"""
    SparseLinksCauset(manifold, sprinkling)

Create a `SparseLinksCauset` from a manifold sprinkling. Conversion computes
the transitive reduction automatically.
"""
function SparseLinksCauset(
    manifold::CausalSets.AbstractManifold{N},
    sprinkling::Vector{CausalSets.Coordinates{N}},
)::SparseLinksCauset where {N}
    return convert(SparseLinksCauset, CausalSets.ManifoldCauset(manifold, sprinkling))
end

function SparseLinksCauset(causet::CausalSets.AbstractCauset)::SparseLinksCauset
    return convert(SparseLinksCauset, causet)
end

"""
    Base.convert(::Type{SparseLinksCauset}, causet)

Convert any `CausalSets.AbstractCauset` to `SparseLinksCauset`, computing
transitive reduction (links) during conversion.

Conversion uses `CausalSets.in_past_of` directly and does not materialize an
intermediate `BitArrayCauset`.
"""
function Base.convert(
    ::Type{SparseLinksCauset},
    causet::CausalSets.AbstractCauset,
)::SparseLinksCauset
    atom_count = length(causet)

    future_links = [Int32[] for _ in 1:atom_count]
    past_links = [Int32[] for _ in 1:atom_count]
    tls_future = [Int32[] for _ in 1:Threads.maxthreadid()]

    # Thread-safe first pass: each iteration only writes to future_links[i].
    # Candidate caching avoids re-checking in_past_of(causet, i, j) for every k.
    Threads.@threads :dynamic for i in 1:atom_count
        future_i = tls_future[Threads.threadid()]
        empty!(future_i)

        @inbounds for k in (i + 1):atom_count
            CausalSets.in_past_of(causet, i, k) || continue
            push!(future_i, Int32(k))
        end

        row = future_links[i]
        @inbounds for k32 in future_i
            k = Int(k32)
            is_link = true
            for j32 in future_i
                j = Int(j32)
                j >= k && break
                if CausalSets.in_past_of(causet, j, k)
                    is_link = false
                    break
                end
            end
            is_link && push!(row, k32)
        end
    end

    # Build past links from future links.
    @inbounds for i in 1:atom_count
        for k in future_links[i]
            push!(past_links[Int(k)], Int32(i))
        end
    end

    return SparseLinksCauset(atom_count, future_links, past_links)
end
