"""
    minkowski_cardinality_abundance(
        n::Int,
        m::Int,
        d::Real;
        jmax::Int = 1000,
        result_type::Type{<:AbstractFloat} = Float64,
        compute_type::Type{<:AbstractFloat} = BigFloat,
        rel_tol::Union{Nothing,Real} = nothing,
    )

Compute the exact interval abundance from Eq. (5) of 2510.19403v1
(using the infinite series representation truncated at `jmax`).

The interval size `m` is inclusive (endpoints counted).

# Behavior
- `m == 1`: returns `n` converted to the selected result type.
- `m ≥ 2`: evaluates the truncated alternating series.

# Arguments
- `m`: Inclusive interval size (`m ≥ 1`).
- `n`: Causal set size / sprinkling count (`n > 0`).
- `d`: Spacetime dimension (`d ≥ 2`).

# Keyword Arguments
- `jmax`: Truncation index for the infinite series (default: 10, must be nonnegative).
- `result_type`: Floating-point type used for the returned value.
- `compute_type`: Floating-point type used for the internal summation.
- `rel_tol`: Optional relative-tolerance stopping criterion for early sum termination.

# Returns
- `result_type`: Predicted interval abundance cast to the requested floating type.

# Throws
- `DomainError`: Raised when numeric input preconditions fail.
"""
function minkowski_cardinality_abundance(
    n::Int,
    m::Int,
    d::Real;
    jmax::Int = 1000,
    result_type::Type{RT} = Float64,
    compute_type::Type{CT} = BigFloat,
    rel_tol::Union{Nothing,Real} = nothing,
) where {RT<:AbstractFloat, CT<:AbstractFloat}
    if !isfinite(n)
        throw(DomainError(n, "n must be finite"))
    end
    if !isfinite(d)
        throw(DomainError(d, "d must be finite"))
    end
    if !(m >= 1)
        throw(DomainError(m, "m must be >= 1"))
    end
    if !(n > 0)
        throw(DomainError(n, "n must be > 0"))
    end
    if !(d >= 2)
        throw(DomainError(d, "d must be >= 2"))
    end
    if !(jmax >= 0)
        throw(DomainError(jmax, "jmax must be >= 0"))
    end
    if rel_tol !== nothing && !(isfinite(rel_tol) && rel_tol > 0)
        throw(DomainError(rel_tol, "rel_tol must be finite and > 0"))
    end

    if m == 1
        return RT(n)
    end

    mp_int = m - 2  # shift to paper convention
    mp = CT(mp_int)
    d = CT(d)
    n = CT(n)

    prefactor = SpecialFunctions.gamma(d)^2 / SpecialFunctions.gamma(mp + one(CT)) * n^CT(mp)
    rtol = rel_tol === nothing ? nothing : CT(rel_tol)

    s = zero(CT)
    for j_int in 0:jmax
        j = CT(j_int)
        sign = isodd(j_int) ? -one(CT) : one(CT)

        term = sign * n^(j + CT(2)) /
            ( SpecialFunctions.gamma(j + one(CT)) *
              (j + mp + one(CT)) *
              (j + mp + CT(2)) ) *
              SpecialFunctions.gamma(d/CT(2) * (j + mp)           + one(CT)) *
              SpecialFunctions.gamma(d/CT(2) * (j + mp + one(CT)) + one(CT)) /
            ( SpecialFunctions.gamma(d/CT(2) * (j + mp + CT(2)) ) *
              SpecialFunctions.gamma(d/CT(2) * (j + mp + CT(3)) ) )
        

        s += term
        if rtol !== nothing
            thresh = rtol * max(abs(s), one(CT))
            if abs(term) <= thresh
                break
            end
        end
    end

    return RT(prefactor * s)
end

"""
    minkowski_cardinality_abundance_2D(
        n::Int,
        m::Int;
        jmax::Int = 1000,
        result_type::Type{<:AbstractFloat} = Float64,
        compute_type::Type{<:AbstractFloat} = BigFloat,
        rel_tol::Union{Nothing,Real} = nothing,
    )

Compute the exact interval abundance from Eq. (5) of 2510.19403v1
in 2D (using the infinite series representation truncated at `jmax`).

The interval size `m` is inclusive (endpoints counted).

# Behavior
- `m == 1`: returns `n` converted to the selected result type.
- `m ≥ 2`: evaluates the truncated alternating series.

# Arguments
- `m`: Inclusive interval size (`m ≥ 1`).
- `n`: Causal set size / sprinkling count (`n > 0`).

# Keyword Arguments
- `jmax`: Truncation index for the infinite series (default: 10, must be nonnegative).
- `result_type`: Floating-point type used for the returned value.
- `compute_type`: Floating-point type used for the internal summation.
- `rel_tol`: Optional relative-tolerance stopping criterion for early sum termination.

# Returns
- `result_type`: Predicted interval abundance cast to the requested floating type.

# Throws
- `DomainError`: Raised when numeric input preconditions fail.
"""
function minkowski_cardinality_abundance_2D(
    n::Int,
    m::Int;
    jmax::Int = 1000,
    result_type::Type{RT} = Float64,
    compute_type::Type{CT} = BigFloat,
    rel_tol::Union{Nothing,Real} = nothing,
) where {RT<:AbstractFloat, CT<:AbstractFloat}
    if !isfinite(n)
        throw(DomainError(n, "n must be finite"))
    end
    if !(m >= 1)
        throw(DomainError(m, "m must be >= 1"))
    end
    if !(n > 0)
        throw(DomainError(n, "n must be > 0"))
    end
    if !(jmax >= 0)
        throw(DomainError(jmax, "jmax must be >= 0"))
    end
    if rel_tol !== nothing && !(isfinite(rel_tol) && rel_tol > 0)
        throw(DomainError(rel_tol, "rel_tol must be finite and > 0"))
    end

    if m == 1
        return RT(n)
    end

    mp_int = m - 2  # shift to paper convention
    mp = CT(mp_int)
    n = CT(n)

    prefactor = SpecialFunctions.gamma(CT(2))^2 / SpecialFunctions.gamma(mp + one(CT)) * n^CT(mp)
    rtol = rel_tol === nothing ? nothing : CT(rel_tol)

    s = zero(CT)
    for j_int in 0:jmax
        j = CT(j_int)
        sign = isodd(j_int) ? -one(CT) : one(CT)
        term = sign * n^(j + CT(2)) /
            ( SpecialFunctions.gamma(j + one(CT)) *
              (j + mp + one(CT))^2 *
              (j + mp +   CT(2))^2 )

        s += term
        if rtol !== nothing
            thresh = rtol * max(abs(s), one(CT))
            if abs(term) <= thresh
                break
            end
        end
    end

    return RT(prefactor * s)
end

"""
    minkowski_cardinality_abundances_2D_asymptotic(
        m::Real,
        n::Real,
    )::Float64

Compute the large-n cardinality abundance in a 2D Minkowski geometry 
with causal-diamond boundary.

The interval size `m` is inclusive (endpoints counted).

# Behavior
- `m == 1`: returns `Float64(n)` exactly.
- `m > 1`: uses the asymptotic closed-form expression implemented below.

# Arguments
- `m`: Inclusive interval size (`m >= 1`).
- `n`: Causal set size / sprinkling count (`n > 0`).

# Returns
- `Float64`: Predicted interval abundance.

# Throws
- `DomainError`: Raised when numeric input preconditions fail."""
function minkowski_cardinality_abundances_2D_asymptotic(
    n::Int,
    m::Int,
)::Float64
    if !isfinite(m)
        throw(DomainError(m, "m must be finite"))
    end
    if !isfinite(n)
        throw(DomainError(n, "n must be finite"))
    end
    if !(m >= 1)
        throw(DomainError(m, "m must be >= 1"))
    end
    if !(n > 0)
        throw(DomainError(n, "n must be > 0"))
    end
    if m == 1
        return float(n)
    end

    mp = m - 2

    return 1 + 2 * mp - 2n + (1 + mp + n) * (log(n) - SpecialFunctions.polygamma(0, 1 + mp))
end