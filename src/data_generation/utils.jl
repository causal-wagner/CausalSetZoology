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
    links = CausalSets.empty_graph(cset.atom_count)
    CausalSets.transitive_reduction!(cset, links)

    # links.edges stores one outgoing row per vertex; transpose hcat result so
    # matrix rows/cols align with (source, target) indexing.
    W_sym = Float64.(transpose(reduce(hcat, links.edges)))
    symmetrize_strictly_upper_triangular!(W_sym)
    return sym_norm_lap_eigs!(W_sym)
end

function degrees(cset::CausalSets.BitArrayCauset)::Tuple{Vector{Int32}, Vector{Int32}}
    n = cset.atom_count
    in_deg  = Vector{Int}(undef, n)
    out_deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        in_deg[i]  = CausalSets.bitvector_count_ones(cset.past_relations[i])
        out_deg[i] = CausalSets.bitvector_count_ones(cset.future_relations[i])
    end
    return in_deg, out_deg
end

function degrees(links::SparseLinksCauset)::Tuple{Vector{Int32}, Vector{Int32}}
    n = links.atom_count
    in_deg  = Vector{Int}(undef, n)
    out_deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        in_deg[i]  = length(links.past_links[i])
        out_deg[i] = length(links.future_links[i])
    end
    return in_deg, out_deg
end

function dense_future_links(cset::SparseLinksCauset)::BitMatrix
    n = Int(cset.atom_count)
    A = falses(n, n)
    @inbounds for i in 1:n
        for j in cset.future_links[i]
            A[i, Int(j)] = true
        end
    end
    return A
end