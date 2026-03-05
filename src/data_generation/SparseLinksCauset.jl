"""
    SparseLinksCauset

Sparse causal set representation that stores only link relations
(transitive reduction), not full transitive closure.
"""
struct SparseLinksCauset <: CausalSets.AbstractCauset
    atom_count::Int32
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

    # Thread-safe first pass: each thread only writes to future_links[i].
    Threads.@threads for i in 1:atom_count
        for k in (i + 1):atom_count
            CausalSets.in_past_of(causet, i, k) || continue
            is_link = true
            for j in (i + 1):(k - 1)
                if CausalSets.in_past_of(causet, i, j) && CausalSets.in_past_of(causet, j, k)
                    is_link = false
                    break
                end
            end
            if is_link
                push!(future_links[i], k)
            end
        end
    end

    # Build past links from future links.
    @inbounds for i in 1:atom_count
        for k in future_links[i]
            push!(past_links[k], Int32(i))
        end
    end

    return SparseLinksCauset(atom_count, future_links, past_links)
end