@testsnippet setupMinkowski begin
    using Test
    using SpecialFunctions
end

# Verifies exact branch behavior at m=1 for all exposed formulas.
@testitem "minkowski_abundance_analytical: m equals one branch" setup=[setupMinkowski] begin
    # For m=1, all variants should return n exactly as Float64 by default.
    @test CausalSetZoology.minkowski_cardinality_abundance(10, 1, 2) == 10.0
    @test CausalSetZoology.minkowski_cardinality_abundance_2D(10, 1) == 10.0
    @test CausalSetZoology.minkowski_cardinality_abundances_2D_asymptotic(10, 1) == 10.0
end

# Verifies typed output behavior for new `result_type` / `compute_type` keywords.
@testitem "minkowski_abundance_analytical: typed outputs for exact formulas" setup=[setupMinkowski] begin
    # m=1 branch should honor requested result type.
    @test CausalSetZoology.minkowski_cardinality_abundance(10, 1, 2; result_type = Float32) isa Float32
    @test CausalSetZoology.minkowski_cardinality_abundance(10, 1, 2; result_type = BigFloat) isa BigFloat
    @test CausalSetZoology.minkowski_cardinality_abundance_2D(10, 1; result_type = Float32) isa Float32
    @test CausalSetZoology.minkowski_cardinality_abundance_2D(10, 1; result_type = BigFloat) isa BigFloat

    # Nontrivial branch should support mixed compute/result types.
    v64 = CausalSetZoology.minkowski_cardinality_abundance(3, 2, 2; jmax = 30, compute_type = BigFloat, result_type = Float64)
    v32 = CausalSetZoology.minkowski_cardinality_abundance(3, 2, 2; jmax = 30, compute_type = BigFloat, result_type = Float32)
    @test v64 isa Float64
    @test v32 isa Float32
    @test Float64(v32) ≈ v64 rtol = 1e-5

    w64 = CausalSetZoology.minkowski_cardinality_abundance_2D(3, 2; jmax = 30, compute_type = BigFloat, result_type = Float64)
    w32 = CausalSetZoology.minkowski_cardinality_abundance_2D(3, 2; jmax = 30, compute_type = BigFloat, result_type = Float32)
    @test w64 isa Float64
    @test w32 isa Float32
    @test Float64(w32) ≈ w64 rtol = 1e-5
end

# Verifies numeric sanity/type for representative nontrivial inputs.
@testitem "minkowski_abundance_analytical: numeric behavior for exact expressions" setup=[setupMinkowski] begin
    vals = [
        CausalSetZoology.minkowski_cardinality_abundance(2, 2, 2; jmax = 20, compute_type = BigFloat),
        CausalSetZoology.minkowski_cardinality_abundance_2D(2, 2; jmax = 20, compute_type = BigFloat),
        CausalSetZoology.minkowski_cardinality_abundance(5, 4, 2; jmax = 30, compute_type = BigFloat),
        CausalSetZoology.minkowski_cardinality_abundance_2D(5, 4; jmax = 30, compute_type = BigFloat),
    ]
    @test all(isfinite, vals)
    @test all(v -> v isa Float64, vals)
end

# Checks that the general-d exact formula reduces to the dedicated 2D exact formula.
@testitem "minkowski_abundance_analytical: exact d formula equals exact 2D formula" setup=[setupMinkowski] begin
    # Compare for several low n,m with m of order 1.
    ns = [2, 3, 5, 8, 12, 20]
    ms = [1, 2, 3, 4]
    for n in ns, m in ms
        v_d2 = CausalSetZoology.minkowski_cardinality_abundance(n, m, 2; jmax = 80, compute_type = BigFloat)
        v_2d = CausalSetZoology.minkowski_cardinality_abundance_2D(n, m; jmax = 80, compute_type = BigFloat)
        @test isapprox(v_d2, v_2d; rtol = 1e-8, atol = 1e-8)
    end
end

# Checks requested approximation quality at the four specific (n,m) pairs.
@testitem "minkowski_abundance_analytical: asymptotic approximates exact 2D at n20 n30 m2 m3" setup=[setupMinkowski] begin
    # Requested points: (n,m) = (20,2), (20,3), (30,2), (30,3).
    pairs = [(20, 2), (20, 3), (30, 2), (30, 3)]
    for (n, m) in pairs
        exact = CausalSetZoology.minkowski_cardinality_abundance_2D(n, m; jmax = 120, compute_type = BigFloat)
        approx = CausalSetZoology.minkowski_cardinality_abundances_2D_asymptotic(n, m)
        relerr = abs(approx - exact) / max(abs(exact), 1e-12)
        @test relerr < 0.20
    end
end

# Validates argument preconditions on all three exposed functions.
@testitem "minkowski_abundance_analytical: validation" setup=[setupMinkowski] begin
    # m must satisfy m >= 1.
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance(10, 0, 2)
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance_2D(10, 0)
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundances_2D_asymptotic(10, 0)

    # n must satisfy n > 0.
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance(0, 2, 2)
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance_2D(0, 2)
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundances_2D_asymptotic(0, 2)

    # d must satisfy d >= 2 in the general formula.
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance(10, 2, 1)

    # jmax must be nonnegative in summation-based formulas.
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance(10, 2, 2; jmax = -1)
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance_2D(10, 2; jmax = -1)

    # rel_tol must be finite and positive in exact formulas.
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance(10, 2, 2; rel_tol = 0.0)
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance(10, 2, 2; rel_tol = -1e-3)
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance_2D(10, 2; rel_tol = 0.0)
    @test_throws DomainError CausalSetZoology.minkowski_cardinality_abundance_2D(10, 2; rel_tol = -1e-3)

    # Non-Int n/m are rejected by dispatch for Int-typed APIs.
    @test_throws MethodError CausalSetZoology.minkowski_cardinality_abundance(Inf, 2, 2)
    @test_throws MethodError CausalSetZoology.minkowski_cardinality_abundance_2D(Inf, 2)
    @test_throws MethodError CausalSetZoology.minkowski_cardinality_abundances_2D_asymptotic(Inf, 2)
    @test_throws MethodError CausalSetZoology.minkowski_cardinality_abundances_2D_asymptotic(2, Inf)

    # Type parameters must be floating-point types for exact formulas.
    @test_throws MethodError CausalSetZoology.minkowski_cardinality_abundance(10, 2, 2; result_type = Int)
    @test_throws MethodError CausalSetZoology.minkowski_cardinality_abundance(10, 2, 2; compute_type = Int)
    @test_throws MethodError CausalSetZoology.minkowski_cardinality_abundance_2D(10, 2; result_type = Int)
    @test_throws MethodError CausalSetZoology.minkowski_cardinality_abundance_2D(10, 2; compute_type = Int)
end
