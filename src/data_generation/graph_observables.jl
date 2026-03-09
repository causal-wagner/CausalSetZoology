"""
    symmetrize_strictly_upper_triangular!(M)

Copy the strict upper triangle of `M` into the strict lower triangle in-place.

# Arguments
- `M`: Square matrix whose upper triangle defines the symmetric entries.

# Returns
- `M`: Mutated symmetric matrix.

# Throws
- `DimensionMismatch`: If `M` is not square.
"""
function symmetrize_strictly_upper_triangular!(M::AbstractMatrix)
    n, m = size(M)
    n == m || throw(DimensionMismatch("symmetrize_strictly_upper_triangular! requires a square matrix, got size ($(n), $(m))"))

    @inbounds for i in 1:n
        for j in (i + 1):n
            M[j, i] = M[i, j]
        end
    end
    return M
end

"""
    sym_norm_lap_eigs!(W)

Compute eigenvalues of the symmetrically normalized Laplacian in-place from
adjacency-like matrix `W`.

The function mutates `W` to `L_sym = I - D^{-1/2} W D^{-1/2}` and returns
`eigvals(Hermitian(W))`.

# Arguments
- `W`: Real square symmetric matrix.

# Returns
- `λ::Vector{Float64}`: Eigenvalues of the normalized Laplacian.

# Throws
- `DimensionMismatch`: If `W` is not square.
- `ArgumentError`: If `W` is not symmetric.
"""
function sym_norm_lap_eigs!(W::AbstractMatrix{<:Real})::Vector{Float64}
    n, m = size(W)
    if n != m
        throw(DimensionMismatch("sym_norm_lap_eigs! requires a square matrix, got size ($n, $m)"))
    end
    if !LinearAlgebra.issymmetric(W)
        throw(ArgumentError("sym_norm_lap_eigs! requires a symmetric matrix"))
    end
    n = size(W, 1)
    deg = vec(sum(W, dims = 2))
    dinvsqrt = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        di = deg[i]
        dinvsqrt[i] = di > 0.0 ? inv(sqrt(di)) : 0.0
    end

    LinearAlgebra.lmul!(LinearAlgebra.Diagonal(dinvsqrt), W)
    LinearAlgebra.rmul!(W, LinearAlgebra.Diagonal(dinvsqrt))
    W .*= -1.0
    @inbounds for i in 1:n
        W[i, i] += 1.0
    end

    return LinearAlgebra.eigvals(LinearAlgebra.Hermitian(W))
end

"""
    normalized_lap_eigs_symmetrized_links(cset)

Compute normalized-Laplacian eigenvalues from the undirected link graph of `cset`.

# Arguments
- `cset`: Input causal set.

# Returns
- `λ::Vector{Float64}`: Eigenvalues of the normalized Laplacian of the symmetrized links.
"""
function normalized_lap_eigs_symmetrized_links(cset::CausalSets.BitArrayCauset)::Vector{Float64}
    # Build the transitive-reduction link graph directly from closure relations.
    # This avoids relying on an external BitArrayCauset->ToposortedDAG method.
    links = CausalSets.empty_graph(cset.atom_count)
    for i in 1:cset.atom_count
        links.edges[i] .= false
    end
    for i in 1:cset.atom_count
        for k in i+1:cset.atom_count
            if cset.future_relations[i][k]
                is_link = true
                for j in i+1:k-1
                    if cset.future_relations[i][j] && cset.future_relations[j][k]
                        is_link = false
                        break
                    end
                end
                links.edges[i][k] = is_link
            end
        end
    end

    # links.edges stores one outgoing row per vertex; transpose hcat result so
    # matrix rows/cols align with (source, target) indexing.
    W_sym = Float64.(transpose(reduce(hcat, links.edges)))
    symmetrize_strictly_upper_triangular!(W_sym)
    return sym_norm_lap_eigs!(W_sym)
end

"""
    degrees(cset::CausalSets.BitArrayCauset)

Compute in/out degrees from closure relations of a `BitArrayCauset`.

# Arguments
- `cset`: Input causal set.

# Returns
- `in_deg::Vector{Int}`: In-degree per node.
- `out_deg::Vector{Int}`: Out-degree per node.
"""
function degrees(cset::CausalSets.BitArrayCauset)::Tuple{Vector{Int}, Vector{Int}}
    n = cset.atom_count
    in_deg  = Vector{Int}(undef, n)
    out_deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        in_deg[i]  = CausalSets.bitvector_count_ones(cset.past_relations[i])
        out_deg[i] = CausalSets.bitvector_count_ones(cset.future_relations[i])
    end
    return in_deg, out_deg
end

"""
    degrees(links::SparseLinksCauset)

Compute in/out degrees from sparse link adjacency lists.

# Arguments
- `links`: Sparse-links causal set.

# Returns
- `in_deg::Vector{Int}`: In-degree per node.
- `out_deg::Vector{Int}`: Out-degree per node.
"""
function degrees(links::SparseLinksCauset)::Tuple{Vector{Int}, Vector{Int}}
    n = links.atom_count
    in_deg  = Vector{Int}(undef, n)
    out_deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        in_deg[i]  = length(links.past_links[i])
        out_deg[i] = length(links.future_links[i])
    end
    return in_deg, out_deg
end

"""
    connectivity(cset::CausalSets.BitArrayCauset)

Compute the directed-edge density (connectivity) of a causal set.

# Arguments
- `cset`: Causal set in transitive-closure representation.

# Returns
- `ρ::Float64`: Fraction of realized relations among all possible directed
  relations for `n` labeled events, i.e.
  `count_relations(cset) / (n * (n - 1) / 2)`.

# Throws
- `DomainError`: If `cset.atom_count < 2`.
"""
function connectivity(cset::CausalSets.BitArrayCauset)::Float64
    n = cset.atom_count
    n >= 2 || throw(DomainError(n, "connectivity is undefined for atom_count < 2"))
    return CausalSets.count_relations(cset) / (n * (n - 1) / 2)
end

"""
    height(cset, source)

Calculate the height (maximum path length) from `source`.
Supported `cset` types:
- `CausalSets.BitArrayCauset` (closure edges)
- `SparseLinksCauset` (link edges)

# Arguments
- `cset`: Input causal set (closure or sparse-links representation).
- `source`: 1-based source index.

# Returns
- `h::Int`: Maximum directed path length reachable from `source`.

# Throws
- `BoundsError`: If `source` is outside `1:atom_count`.
"""
function height(cset::CausalSets.BitArrayCauset, source::Int)::Int
    n = cset.atom_count
    (1 <= source <= n) || throw(BoundsError(1:n, source))

    dist = fill(-1, n)
    dist[source] = 0

    @inbounds for u in source:n
        du = dist[u]
        du < 0 && continue
        fu = cset.future_relations[u]
        for v in (u + 1):n
            fu[v] || continue
            cand = du + 1
            cand > dist[v] && (dist[v] = cand)
        end
    end

    return maximum(dist)
end

function height(links::SparseLinksCauset, source::Int)::Int
    n = links.atom_count
    (1 <= source <= n) || throw(BoundsError(1:n, source))

    dist = fill(-1, n)
    dist[source] = 0

    @inbounds for u in source:n
        du = dist[u]
        du < 0 && continue
        for v in links.future_links[u]
            cand = du + 1
            cand > dist[Int(v)] && (dist[Int(v)] = cand)
        end
    end

    return maximum(dist)
end
