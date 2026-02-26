"""
    minkowski_interval_abundance_2d_inclusive_asymptotic(
        m::Real,
        n::Real,
    )::Float64

Compute the asymptotic inclusive interval abundance in 2D Minkowski spacetime.

The interval size `m` is inclusive (endpoints counted).

# Behavior
- `m == 1`: returns `Float64(n)` exactly.
- `m > 1`: uses the asymptotic closed-form expression implemented below.

# Arguments
- `m`: Inclusive interval size (`m >= 1`).
- `n`: Causal set size / sprinkling count (`n > 0`).

# Returns
- `Float64`: Predicted interval abundance.

# Keyword Arguments
- This method has no keyword arguments.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function minkowski_interval_abundance_2d_inclusive_asymptotic(
    m::Real,
    n::Real,
)::Float64
    @assert m ≥ 1
    @assert n > 0

    if m == 1
        return float(n)
    end

    mp = m - 2

    return 1 + 2 * mp -2n + (1 + mp + n) * (log(n) - SpecialFunctions.polygamma(0, 1 + mp))
end
