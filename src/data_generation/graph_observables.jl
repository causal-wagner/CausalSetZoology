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

# Throws
- `DomainError`: If `cset.atom_count < 1`.
"""
function normalized_lap_eigs_symmetrized_links(cset::CausalSets.BitArrayCauset)::Vector{Float64}
    cset.atom_count >= 1 || throw(DomainError(cset.atom_count, "normalized_lap_eigs_symmetrized_links requires atom_count >= 1"))
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
    imag_antisym_out_lap_eigs(cset)

Compute the eigenvalues of `im` times the antisymmetric part of the directed
normalized out-Laplacian
`L_out = I - A D_out^{-1}`.

The identity term drops out of the antisymmetric part, so the spectrum is real
because `im * (L_out - L_out') / 2` is Hermitian.

# Arguments
- `cset`: Input causal set in closure (BitArrayCauset) or link (SparseLinksCauset) representation.

# Returns
- `λ::Vector{Float64}`: Eigenvalues of `im * (L_out - L_out') / 2`.
"""
function imag_antisym_out_lap_eigs(cset::Union{CausalSets.BitArrayCauset,SparseLinksCauset})::Vector{Float64}
    return Float64.(LinearAlgebra.eigvals(LinearAlgebra.Hermitian(imag_antisym_out_lap(cset))))
end

"""
    imag_antisym_in_lap_eigs(cset)

Compute the eigenvalues of `im` times the antisymmetric part of the directed
normalized in-Laplacian
`L_in = I - D_in^{-1} A'`.

# Arguments
- `cset`: Input causal set in closure (BitArrayCauset) or link (SparseLinksCauset) representation.

# Returns
- `λ::Vector{Float64}`: Eigenvalues of `im * (L_in - L_in') / 2`.
"""
function imag_antisym_in_lap_eigs(cset::Union{CausalSets.BitArrayCauset,SparseLinksCauset})::Vector{Float64}
    return Float64.(LinearAlgebra.eigvals(LinearAlgebra.Hermitian(imag_antisym_in_lap(cset))))
end

"""
    imag_antisym_out_lap(cset)

Construct `im` times the antisymmetric part of the directed normalized
out-Laplacian
`L_out = I - A D_out^{-1}`.

The returned matrix is Hermitian by construction and can be diagonalized with
`eigvals(Hermitian(...))`.

# Arguments
- `cset`: Input causal set in closure (`BitArrayCauset`) representation.

# Returns
- `H::Matrix{ComplexF64}`: Hermitian matrix
  `im * (L_out - L_out') / 2`.
"""
function imag_antisym_out_lap(cset::CausalSets.BitArrayCauset)::Matrix{ComplexF64}
    n = cset.atom_count
    dout = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        dout[i] = CausalSets.bitvector_count_ones(cset.future_relations[i])
    end

    H = zeros(ComplexF64, n, n)
    @inbounds for i in 1:n
        fi = cset.future_relations[i]
        for j in (i + 1):n
            fi[j] || continue
            hij = -0.5im * (dout[j] > 0.0 ? inv(dout[j]) : 0.0)
            H[i, j] = hij
            H[j, i] = conj(hij)
        end
    end
    return H
end

"""
    imag_antisym_out_lap(links)

Construct `im` times the antisymmetric part of the directed normalized
out-Laplacian for a sparse-links causal set.

# Arguments
- `links`: Input causal set in link (`SparseLinksCauset`) representation.

# Returns
- `H::Matrix{ComplexF64}`: Hermitian matrix
  `im * (L_out - L_out') / 2`.
"""
function imag_antisym_out_lap(links::SparseLinksCauset)::Matrix{ComplexF64}
    n = links.atom_count
    dout = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        dout[i] = length(links.future_links[i])
    end

    H = zeros(ComplexF64, n, n)
    @inbounds for i in 1:n
        for j in links.future_links[i]
            hij = -0.5im * (dout[j] > 0.0 ? inv(dout[j]) : 0.0)
            H[i, j] = hij
            H[j, i] = conj(hij)
        end
    end
    return H
end

"""
    imag_antisym_in_lap(cset)

Construct `im` times the antisymmetric part of the directed normalized
in-Laplacian
`L_in = I - D_in^{-1} A'`.

The returned matrix is Hermitian by construction and can be diagonalized with
`eigvals(Hermitian(...))`.

# Arguments
- `cset`: Input causal set in closure (`BitArrayCauset`) representation.

# Returns
- `H::Matrix{ComplexF64}`: Hermitian matrix
  `im * (L_in - L_in') / 2`.
"""
function imag_antisym_in_lap(cset::CausalSets.BitArrayCauset)::Matrix{ComplexF64}
    n = cset.atom_count
    din = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        din[i] = CausalSets.bitvector_count_ones(cset.past_relations[i])
    end

    H = zeros(ComplexF64, n, n)
    @inbounds for i in 1:n
        fi = cset.future_relations[i]
        pi = cset.past_relations[i]
        scale_i = din[i] > 0.0 ? inv(din[i]) : 0.0
        for j in (i + 1):n
            hij = 0.0im
            pi[j] && (hij -= 0.5im * scale_i)
            fi[j] && (hij += 0.5im * (din[j] > 0.0 ? inv(din[j]) : 0.0))
            H[i, j] = hij
            H[j, i] = conj(hij)
        end
    end
    return H
end

"""
    imag_antisym_in_lap(links)

Construct `im` times the antisymmetric part of the directed normalized
in-Laplacian for a sparse-links causal set.

# Arguments
- `links`: Input causal set in link
  (`SparseLinksCauset`) representation.

# Returns
- `H::Matrix{ComplexF64}`: Hermitian matrix
  `im * (L_in - L_in') / 2`.
"""
function imag_antisym_in_lap(links::SparseLinksCauset)::Matrix{ComplexF64}
    n = links.atom_count
    din = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        din[i] = length(links.past_links[i])
    end

    H = zeros(ComplexF64, n, n)
    @inbounds for i in 1:n
        scale_i = din[i] > 0.0 ? inv(din[i]) : 0.0
        for j in links.past_links[i]
            hij = -0.5im * scale_i
            H[i, j] = hij
            H[j, i] = conj(hij)
        end
    end
    return H
end

"""
    communicability_row_sums(cset)

Compute the row sums of the communicability matrix `exp(A)` for the adjacency
representation induced by `cset`.

For topologically sorted causal sets the adjacency is strictly upper triangular,
so `exp(A) * 1` reduces to a finite Taylor series. The implementation follows
the matrix-function-action viewpoint from the communicability literature and
uses repeated adjacency-vector products rather than materializing `exp(A)`.

# Arguments
- `cset`: Input causal set in closure or sparse-link representation.

# Returns
- `tc::Vector{Float64}`: Row sums of `exp(A)`, i.e. `exp(A) * ones(n)`.

# Throws
- `DomainError`: If `atom_count < 1`.
"""
function communicability_row_sums(cset::Union{CausalSets.BitArrayCauset,SparseLinksCauset})::Vector{Float64}
    n = cset.atom_count
    n >= 1 || throw(DomainError(n, "communicability_row_sums is undefined for atom_count < 1"))

    tc = ones(Float64, n)
    term = ones(Float64, n)
    next_term = zeros(Float64, n)

    @inbounds for k in 1:(n - 1)
        _adjacency_mul!(next_term, cset, term)
        next_term ./= k
        tc .+= next_term
        all(iszero, next_term) && break
        term, next_term = next_term, term
    end

    return tc
end

"""
    _adjacency_mul!(y, cset, x)

Apply the directed adjacency operator of `cset` to `x` and store the result in `y`.

# Arguments
- `y`: Output buffer of length `cset.atom_count`.
- `cset`: Input causal set in closure representation.
- `x`: Input vector of length `cset.atom_count`.

# Returns
- `y::AbstractVector{Float64}`: Updated output buffer.

# Throws
- `DimensionMismatch`: If `x` or `y` has length different from `cset.atom_count`.
"""
function _adjacency_mul!(
    y::AbstractVector{Float64},
    cset::CausalSets.BitArrayCauset,
    x::AbstractVector{Float64},
)
    n = cset.atom_count
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)) but atom_count=$n"))
    length(y) == n || throw(DimensionMismatch("y has length $(length(y)) but atom_count=$n"))

    fill!(y, 0.0)
    @inbounds for i in 1:n
        fi = cset.future_relations[i]
        acc = 0.0
        for j in (i + 1):n
            fi[j] || continue
            acc += x[j]
        end
        y[i] = acc
    end
    return y
end

"""
    _adjacency_mul!(y, links, x)

Apply the directed adjacency operator of `links` to `x` and store the result in `y`.

# Arguments
- `y`: Output buffer of length `links.atom_count`.
- `links`: Input sparse-links causal set.
- `x`: Input vector of length `links.atom_count`.

# Returns
- `y::AbstractVector{Float64}`: Updated output buffer.

# Throws
- `DimensionMismatch`: If `x` or `y` has length different from `links.atom_count`.
"""
function _adjacency_mul!(
    y::AbstractVector{Float64},
    links::SparseLinksCauset,
    x::AbstractVector{Float64},
)
    n = links.atom_count
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)) but atom_count=$n"))
    length(y) == n || throw(DimensionMismatch("y has length $(length(y)) but atom_count=$n"))

    fill!(y, 0.0)
    @inbounds for i in 1:n
        acc = 0.0
        for j32 in links.future_links[i]
            acc += x[Int(j32)]
        end
        y[i] = acc
    end
    return y
end

"""
    degrees(cset::CausalSets.BitArrayCauset)

Compute in/out degrees and full degree from closure relations of a `BitArrayCauset`.

# Arguments
- `cset`: Input causal set.

# Returns
- `in_deg::Vector{Int}`: In-degree per node.
- `out_deg::Vector{Int}`: Out-degree per node.
- `deg::Vector{Int}`: Full degree per node, computed as `in_deg + out_deg`.
"""
function degrees(cset::CausalSets.BitArrayCauset)::Tuple{Vector{Int}, Vector{Int}, Vector{Int}}
    n = cset.atom_count
    in_deg  = Vector{Int}(undef, n)
    out_deg = Vector{Int}(undef, n)
    deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        in_deg[i]  = CausalSets.bitvector_count_ones(cset.past_relations[i])
        out_deg[i] = CausalSets.bitvector_count_ones(cset.future_relations[i])
        deg[i] = in_deg[i] + out_deg[i]
    end
    return in_deg, out_deg, deg
end

"""
    degrees(links::SparseLinksCauset)

Compute in/out degrees and full degree from sparse link adjacency lists.

# Arguments
- `links`: Sparse-links causal set.

# Returns
- `in_deg::Vector{Int}`: In-degree per node.
- `out_deg::Vector{Int}`: Out-degree per node.
- `deg::Vector{Int}`: Full degree per node, computed as `in_deg + out_deg`.
"""
function degrees(links::SparseLinksCauset)::Tuple{Vector{Int}, Vector{Int}, Vector{Int}}
    n = links.atom_count
    in_deg  = Vector{Int}(undef, n)
    out_deg = Vector{Int}(undef, n)
    deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        in_deg[i]  = length(links.past_links[i])
        out_deg[i] = length(links.future_links[i])
        deg[i] = in_deg[i] + out_deg[i]
    end
    return in_deg, out_deg, deg
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

"""
    height_distribution(links)

Compute the longest link-path distance from the past boundary for every
element. Sources have height `0`; every other element receives the maximum
path length from any source to that element.
"""
function height_distribution(links::SparseLinksCauset)::Vector{Int}
    n = links.atom_count
    dist = zeros(Int, n)

    @inbounds for u in 1:n
        du = dist[u]
        for v in links.future_links[u]
            vi = Int(v)
            cand = du + 1
            cand > dist[vi] && (dist[vi] = cand)
        end
    end

    return dist
end
