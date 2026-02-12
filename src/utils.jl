# Utilities for analysis statistics on BitArrayCauset and ToposortedDAG.

import Arpack
import CausalSets
import LinearAlgebra
import SparseArrays
import StatsBase
import Statistics

# --------------------------
# Basic stats helpers
# --------------------------

@inline function minmaxmean(v::AbstractVector{<:Real})
    n = length(v)
    @inbounds begin
        x1 = v[1]
        minv = x1
        maxv = x1
        s = float(x1)
        for i in 2:n
            x = v[i]
            if x < minv
                minv = x
            elseif x > maxv
                maxv = x
            end
            s += x
        end
    end
    return minv, maxv, s / n
end

@inline function quartiles(v::AbstractVector{<:Real})
    q = StatsBase.quantile(v, (0.25, 0.5, 0.75))
    return q[1], q[2], q[3]
end

@inline function summary_stats(v::AbstractVector{<:Real})
    minv, maxv, meanv = minmaxmean(v)
    q25, q50, q75 = quartiles(v)
    return minv, maxv, meanv, q25, q75, q50
end

# Fast histogram that returns a sparse Dict (value => count).
# For integer data with small range it uses a dense counter internally.
function histmap(v::AbstractVector{<:Integer})
    minv = minimum(v)
    maxv = maximum(v)
    range = maxv - minv + 1
    n = length(v)

    if range <= 4n && range <= 1_000_000
        counts = zeros(Int, range)
        @inbounds for x in v
            counts[x - minv + 1] += 1
        end
        d = Dict{Int, Int}()
        @inbounds for (i, c) in enumerate(counts)
            c == 0 && continue
            d[minv + i - 1] = c
        end
        return d
    end

    d = Dict{Int, Int}()
    @inbounds for x in v
        d[x] = get(d, x, 0) + 1
    end
    return d
end

function histmap(v::AbstractVector)
    d = Dict{eltype(v), Int}()
    @inbounds for x in v
        d[x] = get(d, x, 0) + 1
    end
    return d
end

@inline function summary_with_hist(v::AbstractVector{<:Real})
    hist = histmap(v)
    minv, maxv, meanv = minmaxmean(v)
    q25, q50, q75 = quartiles(v)
    return hist, minv, maxv, meanv, q25, q75, q50
end

# Sparsify a histogram/count array by storing only nonzero entries.
# Returns Dict{Int,Int} where the key is the linear index.
function sparse_hist_counts(v::AbstractArray{<:Integer})
    d = Dict{Int, Int}()
    for (k, count) in enumerate(v)
        count == 0 && continue
        d[k] = count
    end
    return d
end

# --------------------------
# Degree utilities
# --------------------------

function degree_vectors(cset::CausalSets.BitArrayCauset)
    n = cset.atom_count
    in_deg = Vector{Int}(undef, n)
    out_deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        in_deg[i] = CausalSets.bitvector_count_ones(cset.past_relations[i])
        out_deg[i] = CausalSets.bitvector_count_ones(cset.future_relations[i])
    end
    return in_deg, out_deg
end

function degree_vectors(links::CausalSets.ToposortedDAG)
    n = links.count
    in_deg = zeros(Int, n)
    out_deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        row = links.edges[i]
        out_deg[i] = CausalSets.bitvector_count_ones(row)
        for j in i+1:n
            if row[j]
                in_deg[j] += 1
            end
        end
    end
    return in_deg, out_deg
end

@inline function degree_stats(in_deg::AbstractVector{<:Real}, out_deg::AbstractVector{<:Real})
    return summary_with_hist(in_deg)..., summary_with_hist(out_deg)...
end

@inline function source_sink_counts(in_deg::AbstractVector{<:Real}, out_deg::AbstractVector{<:Real})
    num_sources = count(==(0), in_deg)
    num_sinks = count(==(0), out_deg)
    return num_sources, num_sinks
end

# --------------------------
# Bitvector-backed adjacency with fast mul!
# --------------------------

struct BitAdjacency{T<:AbstractVector{BitVector}}
    rows::T
    n::Int
end

BitAdjacency(rows::AbstractVector{BitVector}) = BitAdjacency(rows, length(rows))
BitAdjacency(graph::CausalSets.ToposortedDAG) = BitAdjacency(graph.edges, graph.count)

Base.size(A::BitAdjacency) = (A.n, A.n)

function LinearAlgebra.mul!(y::AbstractVector, A::BitAdjacency, x::AbstractVector)
    n = A.n
    @boundscheck length(y) == n || throw(DimensionMismatch("y has length $(length(y)), expected $n"))
    @boundscheck length(x) == n || throw(DimensionMismatch("x has length $(length(x)), expected $n"))
    @inbounds for i in 1:n
        row = A.rows[i]
        s = zero(eltype(y))
        @inbounds for (chunk_idx, bits) in enumerate(row.chunks)
            b = bits
            while b != 0
                tz = trailing_zeros(b)
                j = (chunk_idx - 1) * 64 + tz + 1
                s += x[j]
                b &= b - 1
            end
        end
        y[i] = s
    end
    return y
end

# --------------------------
# Krylov expmv and row sums
# --------------------------

function expmv_krylov(A::BitAdjacency, v::AbstractVector{<:Real}; m::Int=30, tol::Real=1e-8)
    n = A.n
    @boundscheck length(v) == n || throw(DimensionMismatch("v has length $(length(v)), expected $n"))
    beta = LinearAlgebra.norm(v)
    beta == 0 && return zeros(Float64, n)

    V = Matrix{Float64}(undef, n, m + 1)
    H = zeros(Float64, m + 1, m)
    @inbounds V[:, 1] .= v ./ beta

    w = Vector{Float64}(undef, n)
    m_eff = m
    for j in 1:m
        LinearAlgebra.mul!(w, A, @view V[:, j])
        for i in 1:j
            hij = LinearAlgebra.dot(@view V[:, i], w)
            H[i, j] = hij
            @inbounds @. w = w - hij * V[:, i]
        end
        hnext = LinearAlgebra.norm(w)
        H[j + 1, j] = hnext
        if hnext < tol
            m_eff = j
            break
        end
        @inbounds V[:, j + 1] .= w ./ hnext
    end

    Hm = H[1:m_eff, 1:m_eff]
    expHm = LinearAlgebra.exp(Hm)
    e1 = zeros(Float64, m_eff)
    e1[1] = 1.0
    y = V[:, 1:m_eff] * (expHm * e1)
    return beta .* y
end

function exp_row_sums(links::CausalSets.ToposortedDAG; m::Int=30, tol::Real=1e-8)
    A = BitAdjacency(links)
    v = ones(Float64, A.n)
    return expmv_krylov(A, v; m=m, tol=tol)
end

function total_communicability_hist(links::CausalSets.ToposortedDAG; m::Int=30, tol::Real=1e-8)
    rowsums = exp_row_sums(links; m=m, tol=tol)
    return histmap(rowsums)
end

function total_communicability_stats(links::CausalSets.ToposortedDAG; m::Int=30, tol::Real=1e-8)
    rowsums = exp_row_sums(links; m=m, tol=tol)
    hist = histmap(rowsums)
    minv, maxv, meanv, q25, q75, q50 = summary_stats(rowsums)
    return hist, minv, maxv, meanv, q25, q75, q50
end

# --------------------------
# Chung normalized Laplacian (lazy random walk)
# --------------------------

function lazy_transition_sparse(links::CausalSets.ToposortedDAG)
    n = links.count
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    @inbounds for i in 1:n
        row = links.edges[i]
        outdeg = CausalSets.bitvector_count_ones(row)
        if outdeg == 0
            push!(rows, i); push!(cols, i); push!(vals, 1.0)
            continue
        end
        push!(rows, i); push!(cols, i); push!(vals, 0.5)
        inv2d = 0.5 / outdeg
        @inbounds for (chunk_idx, bits) in enumerate(row.chunks)
            b = bits
            while b != 0
                tz = trailing_zeros(b)
                j = (chunk_idx - 1) * 64 + tz + 1
                push!(rows, i); push!(cols, j); push!(vals, inv2d)
                b &= b - 1
            end
        end
    end
    return SparseArrays.sparse(rows, cols, vals, n, n)
end

function stationary_distribution(P::SparseArrays.SparseMatrixCSC{Float64, Int};
    tol::Real=1e-10,
    maxiter::Int=10_000,
    floor::Real=1e-15,
)
    n = size(P, 1)
    φ = fill(1.0 / n, n)
    tmp = similar(φ)
    for _ in 1:maxiter
        tmp .= P' * φ
        tmp ./= sum(tmp)
        if LinearAlgebra.norm(tmp - φ, 1) < tol
            φ = tmp
            break
        end
        φ, tmp = tmp, φ
    end
    @inbounds for i in 1:n
        if φ[i] < floor
            φ[i] = floor
        end
    end
    φ ./= sum(φ)
    return φ
end

function chung_laplacian_normalized_lazy(links::CausalSets.ToposortedDAG;
    tol::Real=1e-10,
    maxiter::Int=10_000,
    floor::Real=1e-15,
)
    P = lazy_transition_sparse(links)
    φ = stationary_distribution(P; tol=tol, maxiter=maxiter, floor=floor)
    s = sqrt.(φ)
    s_inv = 1.0 ./ s
    Φ12 = LinearAlgebra.Diagonal(s)
    Φm12 = LinearAlgebra.Diagonal(s_inv)
    S = Φ12 * P * Φm12
    n = size(P, 1)
    L = SparseArrays.spdiagm(0 => ones(n)) - 0.5 * (S + S')
    return L
end

function chung_laplacian_eigenvalues_full(links::CausalSets.ToposortedDAG;
    tol::Real=1e-10,
    maxiter::Int=10_000,
    floor::Real=1e-15,
)
    L = chung_laplacian_normalized_lazy(links; tol=tol, maxiter=maxiter, floor=floor)
    vals = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(Matrix(L)))
    return vals
end

function chung_laplacian_eigenvalues_cutoff(links::CausalSets.ToposortedDAG;
    cutoff::Real,
    tol::Real=1e-10,
    maxiter::Int=10_000,
    floor::Real=1e-15,
    nev_start::Int=12,
    nev_step::Int=12,
)
    L = chung_laplacian_normalized_lazy(links; tol=tol, maxiter=maxiter, floor=floor)
    n = size(L, 1)
    nev = min(n - 1, nev_start)
    vals = Float64[]
    while true
        d, _ = Arpack.eigs(LinearAlgebra.Symmetric(L); nev=nev, which=:SM)
        vals = sort(real(d))
        if isempty(vals)
            return vals
        end
        if vals[end] > cutoff || nev >= n - 1
            return vals[vals .> cutoff]
        end
        nev = min(n - 1, nev + nev_step)
    end
end

function LinearAlgebra.mul!(
    y::AbstractVector,
    A::BitAdjacency,
    x::AbstractVector,
    α::Number,
    β::Number,
)
    n = A.n
    @boundscheck length(y) == n || throw(DimensionMismatch("y has length $(length(y)), expected $n"))
    @boundscheck length(x) == n || throw(DimensionMismatch("x has length $(length(x)), expected $n"))
    @inbounds for i in 1:n
        row = A.rows[i]
        s = zero(eltype(y))
        @inbounds for (chunk_idx, bits) in enumerate(row.chunks)
            b = bits
            while b != 0
                tz = trailing_zeros(b)
                j = (chunk_idx - 1) * 64 + tz + 1
                s += x[j]
                b &= b - 1
            end
        end
        y[i] = α * s + β * y[i]
    end
    return y
end

# --------------------------
# Laplacian eigenvalue stats (dense, optional)
# --------------------------

@inline function bitmatrix_from_rows(rows::Vector{BitVector})
    return transpose(reduce(hcat, rows))
end

function laplacian_eigen_stats_from_rows(adj_rows::Vector{BitVector}, in_deg, out_deg)
    adj = bitmatrix_from_rows(adj_rows)
    in_laplacian = LinearAlgebra.Diagonal(in_deg) - transpose(adj)
    out_laplacian = LinearAlgebra.Diagonal(out_deg) - adj

    ev_in = LinearAlgebra.eigen(Matrix(in_laplacian)).values
    ev_out = LinearAlgebra.eigen(Matrix(out_laplacian)).values

    return summary_with_hist(ev_in)..., summary_with_hist(ev_out)...
end

# --------------------------
# Max path length stats (bitvector-native)
# --------------------------

@inline function _max_pathlen_from_row(row::BitVector, longest::Vector{Int})
    maxlen = 0
    @inbounds for (chunk_idx, bits) in enumerate(row.chunks)
        b = bits
        while b != 0
            tz = trailing_zeros(b)
            j = (chunk_idx - 1) * 64 + tz + 1
            l = longest[j] + 1
            if l > maxlen
                maxlen = l
            end
            b &= b - 1
        end
    end
    return maxlen
end

function max_pathlens_from_rows(rows::Vector{BitVector})
    n = length(rows)
    longest = zeros(Int, n)
    @inbounds for i in n:-1:1
        longest[i] = _max_pathlen_from_row(rows[i], longest)
    end
    return longest
end

function max_pathlen_stats_from_links(links::CausalSets.ToposortedDAG, in_deg, out_deg)
    sources = findall(in_deg .== 0)
    sinks = findall(out_deg .== 0)

    longest = max_pathlens_from_rows(links.edges)
    max_pathlens = longest[sources]

    return histmap(max_pathlens), length(sources), length(sinks), summary_stats(max_pathlens)...
end

# --------------------------
# Transitive reduction for BitArrayCauset -> ToposortedDAG
# --------------------------

function transitive_reduction!(cset::CausalSets.BitArrayCauset, links::CausalSets.ToposortedDAG)
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
end
