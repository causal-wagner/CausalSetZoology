"""
    dense_future_links(cset::SparseLinksCauset)::BitMatrix

Materialize the sparse future-link adjacency of `cset` as a dense `BitMatrix`.

# Arguments
- `cset`: Sparse-links causal set with `future_links` adjacency lists.

# Returns
- `A::BitMatrix`: Dense adjacency matrix where `A[i, j] == true` iff `j` is in `future_links[i]`.

# Throws
- `DimensionMismatch`: If `future_links` row count does not match `atom_count`.
- `BoundsError`: If a stored target index is outside `1:atom_count`.
"""
function dense_future_links(cset::SparseLinksCauset)::BitMatrix
    n = cset.atom_count
    if length(cset.future_links) != n
        throw(
            DimensionMismatch(
                "future_links has $(length(cset.future_links)) rows but atom_count=$n",
            ),
        )
    end
    A = falses(n, n)
    @inbounds for i in 1:n
        for j in cset.future_links[i]
            jj = Int(j)
            if !(1 <= jj <= n)
                throw(BoundsError(1:n, jj))
            end
            A[i, jj] = true
        end
    end
    return A
end

"""
    sparse_hist(v::AbstractVector{<:Integer})::Dict{Int,Int}

Convert a dense histogram count vector to a sparse dictionary, dropping zero bins.

# Arguments
- `v`: Dense histogram counts indexed from 1.

# Returns
- `d::Dict{Int,Int}`: Sparse nonzero bins as `bin_index => count`.

# Throws
- `DomainError`: If any count is negative.
"""
function sparse_hist(v::AbstractVector{<:Integer})::Dict{Int,Int}
    if any(<(0), v)
        throw(DomainError(minimum(v), "histogram counts must be nonnegative"))
    end
    d = Dict{Int,Int}()
    for (k, count) in enumerate(v)
        count == 0 && continue
        d[k] = count
    end
    return d
end

"""
    ev_summary(ev::AbstractVector{<:Real})

Compute summary statistics for an eigenvalue vector.

Returned tuple fields are:
`(ev, num_zero_ev, min_abs_nonzero_ev, minimum, maximum, mean, q25, q75, median)`.

# Arguments
- `ev`: Eigenvalue vector.

# Returns
- `summary::Tuple`: Summary tuple described above.

# Throws
- `ArgumentError`: If `ev` is empty.
- `DomainError`: If `ev` contains non-finite values.
"""
function ev_summary(ev::AbstractVector{<:Real})
    if isempty(ev)
        throw(ArgumentError("ev must be non-empty"))
    end
    if any(!isfinite, ev)
        throw(DomainError(ev, "ev must contain only finite values"))
    end
    abs_ev = abs.(ev)
    num_zero_ev = count(abs_ev .<= 1e-10)
    min_abs_nonzero_ev = let nz = abs_ev[abs_ev .> 1e-10]
        isempty(nz) ? NaN : minimum(nz)
    end
    return (
        ev,
        num_zero_ev,
        min_abs_nonzero_ev,
        minimum(ev),
        maximum(ev),
        Statistics.mean(ev),
        Statistics.quantile(ev, 0.25),
        Statistics.quantile(ev, 0.75),
        Statistics.quantile(ev, 0.5),
    )
end
