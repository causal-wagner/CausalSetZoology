@testsnippet setupHistogramFitting begin
    using Test
    using Statistics
    using Random
    using LinearAlgebra
    using Optim
end

# Checks helper round-trip conversions between NamedTuple parameters and vectors.
@testitem "histogram_fitting helpers: param vector conversion" setup=[setupHistogramFitting] begin
    # Round-trip: NamedTuple -> vector -> NamedTuple preserves parameter values/order.
    syms = (:a, :b, :c)
    nt = (a = 2.0, b = -1.0, c = 0.5)
    v = CausalSetZoology._fit_curve_to_vec(syms, nt)
    @test v == [2.0, -1.0, 0.5]
    @test CausalSetZoology._fit_curve_to_nt(syms, v) == nt
end

# Verifies bounds mode construction, conversion, and clamping behavior.
@testitem "histogram_fitting helpers: bounds modes" setup=[setupHistogramFitting] begin
    # `nothing` bounds should map to no-bounds mode and back to `nothing`.
    nb = CausalSetZoology._fit_curve_make_bounds_mode(nothing)
    @test nb isa CausalSetZoology._NoBounds
    @test CausalSetZoology._fit_curve_bounds_tuple(nb) === nothing

    # Tuple bounds should be normalized to a boxed-bounds mode.
    b = CausalSetZoology._fit_curve_make_bounds_mode(([-1, 0], [2, 3]))
    @test b isa CausalSetZoology._BoxBounds
    @test b.lower == [-1.0, 0.0]
    @test b.upper == [2.0, 3.0]
    @test CausalSetZoology._fit_curve_bounds_tuple(b) == ([-1.0, 0.0], [2.0, 3.0])

    # Applying bounds should clamp component-wise.
    x = [10.0, -5.0]
    @test CausalSetZoology._fit_curve_apply_bounds(x, b) == [2.0, 0.0]
    @test CausalSetZoology._fit_curve_apply_bounds(x, nb) === x
end

# Validates error paths for malformed bounds input and mismatched clamp dimensions.
@testitem "histogram_fitting helpers: bounds modes validation" setup=[setupHistogramFitting] begin
    # Lower/upper size mismatch should fail.
    @test_throws ArgumentError CausalSetZoology._fit_curve_make_bounds_mode(([-1.0], [1.0, 2.0]))

    # Component-wise lower > upper should fail.
    @test_throws ArgumentError CausalSetZoology._fit_curve_make_bounds_mode(([0.0, 2.0], [1.0, 1.0]))

    # Clamping requires equal dimensionality between x and stored bounds.
    b = CausalSetZoology._fit_curve_make_bounds_mode(([-1.0, 0.0], [2.0, 3.0]))
    @test_throws ArgumentError CausalSetZoology._fit_curve_apply_bounds([1.0], b)
end

# Verifies objective-mode construction and accessors for the unweighted branch.
@testitem "histogram_fitting helpers: weighting modes objective unweighted" setup=[setupHistogramFitting] begin
    # No stds and no chi-squared minimization should produce unweighted mode.
    m_obj_u = CausalSetZoology._fit_curve_make_objective_mode(nothing, nothing, false)
    @test m_obj_u isa CausalSetZoology._UnweightedMode
    @test CausalSetZoology._fit_curve_stds(m_obj_u) === nothing
    @test CausalSetZoology._fit_curve_std_fn(m_obj_u) === nothing
    @test !CausalSetZoology._fit_curve_has_stds(m_obj_u)
    @test CausalSetZoology._fit_curve_label(m_obj_u) == "rel_rms"
end

# Verifies objective-mode construction and accessors for the weighted branch.
@testitem "histogram_fitting helpers: weighting modes objective weighted" setup=[setupHistogramFitting] begin
    # Stds plus chi-squared minimization should produce weighted mode with stored std_fn.
    stds = [0.2, 0.3, 0.4]
    sfn = (y, yhat, s, p) -> s
    m_obj_w = CausalSetZoology._fit_curve_make_objective_mode(stds, sfn, true)
    @test m_obj_w isa CausalSetZoology._WeightedMode
    @test CausalSetZoology._fit_curve_stds(m_obj_w) == stds
    @test CausalSetZoology._fit_curve_std_fn(m_obj_w) === sfn
    @test CausalSetZoology._fit_curve_has_stds(m_obj_w)
    @test CausalSetZoology._fit_curve_label(m_obj_w) == "χ²"
end

# Verifies diagnostics-mode selection from presence/absence of standard deviations.
@testitem "histogram_fitting helpers: weighting modes diagnostics" setup=[setupHistogramFitting] begin
    stds = [0.2, 0.3, 0.4]
    sfn = (y, yhat, s, p) -> s

    # Diagnostics mode is unweighted without stds and weighted with stds.
    m_diag_u = CausalSetZoology._fit_curve_make_diagnostics_mode(nothing, nothing)
    m_diag_w = CausalSetZoology._fit_curve_make_diagnostics_mode(stds, sfn)
    @test m_diag_u isa CausalSetZoology._UnweightedMode
    @test m_diag_w isa CausalSetZoology._WeightedMode
end

# Validates weighting mode throw path when chi-squared minimization is requested without stds.
@testitem "histogram_fitting helpers: weighting modes validation" setup=[setupHistogramFitting] begin
    # Objective mode requires stds if `minimize_χ²=true`.
    @test_throws ArgumentError CausalSetZoology._fit_curve_make_objective_mode(nothing, nothing, true)
end

# Checks sigma resolution and residual-bundle assembly with explicit expected values.
@testitem "histogram_fitting helpers: sigma and residual bundle" setup=[setupHistogramFitting] begin
    # Setup a simple linear model with one bounded parameter candidate.
    xs = [1.0, 2.0, 3.0]
    ys = [3.0, 5.0, 7.0]
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    b = CausalSetZoology._fit_curve_make_bounds_mode(([1.0, 0.5], [2.0, 2.0]))
    u = CausalSetZoology._UnweightedMode()
    w = CausalSetZoology._WeightedMode([1.0, 2.0, 3.0], (y, yhat, s, p) -> 2 .* s)

    # Unweighted mode should return no sigma vector.
    @test CausalSetZoology._fit_curve_sigma(u, ys, ys, (a = 1.0, b = 1.0)) === nothing

    # Weighted mode should return std_fn-transformed sigmas.
    σ = CausalSetZoology._fit_curve_sigma(w, ys, ys, (a = 1.0, b = 1.0))
    @test σ == [2.0, 4.0, 6.0]

    # Residual bundle should include clamped params, predictions, residuals, and effective sigmas.
    bundle = CausalSetZoology._fit_curve_residual_bundle([10.0, -5.0], ys, f, xs, syms, b, w)
    @test bundle.x == [2.0, 0.5]
    @test bundle.params == (a = 2.0, b = 0.5)
    @test bundle.preds == [2.5, 4.5, 6.5]
    @test bundle.residuals == [0.5, 0.5, 0.5]
    @test bundle.σ == [2.0, 4.0, 6.0]
end

# Validates sigma/residual-bundle error paths for bad dimensions and invalid sigmas.
@testitem "histogram_fitting helpers: sigma and residual bundle validation" setup=[setupHistogramFitting] begin
    xs = [1.0, 2.0, 3.0]
    ys = [3.0, 5.0, 7.0]
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    nb = CausalSetZoology._NoBounds()

    # Effective stds length must match ys length.
    w_bad_len = CausalSetZoology._WeightedMode([1.0, 2.0, 3.0], (y, yhat, s, p) -> [1.0, 2.0])
    @test_throws ArgumentError CausalSetZoology._fit_curve_sigma(w_bad_len, ys, ys, (a = 1.0, b = 1.0))

    # Effective stds must be finite and strictly positive.
    w_bad_vals = CausalSetZoology._WeightedMode([1.0, 2.0, 3.0], (y, yhat, s, p) -> [1.0, 0.0, Inf])
    @test_throws ArgumentError CausalSetZoology._fit_curve_sigma(w_bad_vals, ys, ys, (a = 1.0, b = 1.0))

    # Residual bundle validates input lengths.
    @test_throws ArgumentError CausalSetZoology._fit_curve_residual_bundle([1.0, 1.0], ys, f, xs[1:2], syms, nb, CausalSetZoology._UnweightedMode())
    @test_throws ArgumentError CausalSetZoology._fit_curve_residual_bundle([1.0], ys, f, xs, syms, nb, CausalSetZoology._UnweightedMode())
end

# Checks explicit objective/score scalar formulas for weighted and unweighted cases.
@testitem "histogram_fitting helpers: objective and score values" setup=[setupHistogramFitting] begin
    residuals = [1.0, -2.0, 3.0]
    ys = [2.0, 4.0, 8.0]
    σ = [1.0, 2.0, 4.0]

    # Objective is sum(residuals^2) unweighted and sum((residuals/σ)^2) weighted.
    @test CausalSetZoology._fit_curve_objective_value(residuals, nothing) == 14.0
    # (1/1)^2 + (2/2)^2 + (3/4)^2 = 1 + 1 + 9/16 = 41/16 = 2.5625
    @test CausalSetZoology._fit_curve_objective_value(residuals, σ) == 2.5625

    # Score is relative RMS in unweighted mode.
    expected_rel = sqrt(mean((residuals ./ ys) .^ 2))
    @test CausalSetZoology._fit_curve_score_value(residuals, ys, nothing, 2) ≈ expected_rel atol = 1e-12

    # Score is reduced chi-squared in weighted mode.
    # Reduced chi-squared uses the same numerator 41/16 = 2.5625.
    expected_chi = 2.5625 / (length(ys) - 1)
    @test CausalSetZoology._fit_curve_score_value(residuals, ys, σ, 1) ≈ expected_chi atol = 1e-12

    # dof <= 0 branch returns NaN for weighted score.
    @test isnan(CausalSetZoology._fit_curve_score_value(residuals[1:2], ys[1:2], σ[1:2], 2))
end

# Validates objective/score value throw paths.
@testitem "histogram_fitting helpers: objective and score values validation" setup=[setupHistogramFitting] begin
    residuals = [1.0, -2.0, 3.0]
    ys = [2.0, 4.0, 8.0]

    # Objective validates sigma length.
    @test_throws ArgumentError CausalSetZoology._fit_curve_objective_value(residuals, [1.0, 2.0])

    # Score validates p and vector lengths.
    @test_throws ArgumentError CausalSetZoology._fit_curve_score_value(residuals, ys, nothing, -1)
    @test_throws ArgumentError CausalSetZoology._fit_curve_score_value(residuals[1:2], ys, nothing, 1)
    @test_throws ArgumentError CausalSetZoology._fit_curve_score_value(residuals, ys, [1.0, 2.0], 1)
end

# Verifies parameter-key validation helper for setup NamedTuples.
@testitem "histogram_fitting helpers: require parameter keys" setup=[setupHistogramFitting] begin
    # Complete parameter sets should pass and return `nothing`.
    @test CausalSetZoology._fit_curve_require_param_keys((a = 1.0, b = 2.0), (:a, :b), "init") === nothing
end

# Validates parameter-key helper throw path.
@testitem "histogram_fitting helpers: require parameter keys validation" setup=[setupHistogramFitting] begin
    # Missing key should throw.
    @test_throws ArgumentError CausalSetZoology._fit_curve_require_param_keys((a = 1.0,), (:a, :b), "init")
end

# Checks input normalization and config construction from _fit_curve_prepare_inputs.
@testitem "histogram_fitting helpers: prepare inputs" setup=[setupHistogramFitting] begin
    # Setup includes zero std for replace_zeros path and explicit bounds/init mapping.
    ys = [2.0, 4.0, 6.0]
    xs, ys_out, cfg = CausalSetZoology._fit_curve_prepare_inputs(
        ys,
        (:a, :b);
        stds = [1.0, 0.0, 2.0],
        minimize_χ² = true,
        init = (a = 3.0, b = -1.0),
        bounds = ((a = 0.0, b = -2.0), (a = 5.0, b = 2.0)),
        ϵ = 1e-2,
        multistart = 4,
        rng = Random.Xoshiro(7),
        goodness_of_fit = true,
        return_cov = true,
        bootstrap_errorbars = true,
        n_boot = 11,
    )

    # x/y passthrough and config basics.
    @test xs == [1, 2, 3]
    @test ys_out == ys
    @test cfg.param_syms == (:a, :b)
    @test cfg.init_vec == [3.0, -1.0]
    @test cfg.multistart == 4
    @test cfg.goodness_of_fit
    @test cfg.return_cov
    @test cfg.bootstrap_errorbars
    @test cfg.n_boot == 11

    # Bounds/objective/diagnostics modes should be fully initialized.
    @test cfg.bounds_mode isa CausalSetZoology._BoxBounds
    @test cfg.objective_mode isa CausalSetZoology._WeightedMode
    @test cfg.diagnostics_mode isa CausalSetZoology._WeightedMode
    @test CausalSetZoology._fit_curve_stds(cfg.objective_mode) == [1.0, 0.01, 2.0]
    @test CausalSetZoology._fit_curve_std_fn(cfg.objective_mode) === nothing
end

# Validates throw paths for _fit_curve_prepare_inputs preconditions.
@testitem "histogram_fitting helpers: prepare inputs validation" setup=[setupHistogramFitting] begin
    ys = [2.0, 4.0, 6.0]
    syms = (:a, :b)

    # std_fn requires stds.
    @test_throws ArgumentError CausalSetZoology._fit_curve_prepare_inputs(ys, syms; std_fn = (y, yhat, s, p) -> s)

    # Scalar option checks.
    @test_throws DomainError CausalSetZoology._fit_curve_prepare_inputs(ys, syms; ϵ = 0.0)
    @test_throws ArgumentError CausalSetZoology._fit_curve_prepare_inputs(ys, syms; multistart = 0)
    @test_throws ArgumentError CausalSetZoology._fit_curve_prepare_inputs([2.0, NaN, 6.0], syms)

    # Weighted setup checks.
    @test_throws ArgumentError CausalSetZoology._fit_curve_prepare_inputs(ys, syms; minimize_χ² = true)
    @test_throws ArgumentError CausalSetZoology._fit_curve_prepare_inputs(ys, syms; stds = [1.0])

    # init/bounds key checks.
    @test_throws ArgumentError CausalSetZoology._fit_curve_prepare_inputs(ys, syms; init = (a = 1.0,))
    @test_throws ArgumentError CausalSetZoology._fit_curve_prepare_inputs(ys, syms; bounds = ((a = 0.0,), (a = 1.0, b = 2.0)))
    @test_throws ArgumentError CausalSetZoology._fit_curve_prepare_inputs(ys, syms; bounds = ((a = 2.0, b = 0.0), (a = 1.0, b = 1.0)))

    # x/y length mismatch check.
    @test_throws DimensionMismatch CausalSetZoology._fit_curve_prepare_inputs(ys, syms; x_values = [1.0, 2.0])
end

# Verifies multistart run summary and finalized internal result assembly.
@testitem "histogram_fitting helpers: run and finalize" setup=[setupHistogramFitting] begin
    # Use a deterministic linear fit setup to test orchestration helpers.
    xs = collect(1.0:8.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    _, _, cfg = CausalSetZoology._fit_curve_prepare_inputs(
        ys,
        (:a, :b);
        x_values = xs,
        init = (a = 0.0, b = 0.0),
        multistart = 3,
        rng = Random.Xoshiro(123),
    )

    # Multistart should return a typed summary and callable solver callback.
    run, solve_local = CausalSetZoology._fit_curve_run_multistart(ys, f, xs, cfg)
    @test run isa CausalSetZoology._FitRunSummary
    @test length(run.best_x) == 2
    @test run.best_f >= 0.0
    @test solve_local isa Function

    # Finalized result should carry fitted params and residual vector.
    result = CausalSetZoology._fit_curve_finalize_result(ys, f, xs, cfg, run, solve_local)
    @test result isa CausalSetZoology._FitResult
    @test result.params.a ≈ 2.0 atol = 1e-2
    @test result.params.b ≈ 1.0 atol = 1e-2
    @test length(result.residuals) == length(ys)
    @test result.χ² === nothing
    @test result.cov === nothing
    @test result.stderr === nothing
end

# Verifies public output shaping across all combinations of diagnostics/covariance modes.
@testitem "histogram_fitting helpers: public output shaping" setup=[setupHistogramFitting] begin
    ys = [2.0, 4.0, 8.0]
    result = CausalSetZoology._FitResult(
        (a = 2.0, b = 1.0),
        [0.2, -0.4, 0.8],
        1.25,
        [1.0 0.0; 0.0 2.0],
        (a = 1.0, b = sqrt(2.0)),
    )

    # Plain mode: returns only params NamedTuple.
    _, _, cfg_plain = CausalSetZoology._fit_curve_prepare_inputs(ys, (:a, :b))
    out_plain = CausalSetZoology._fit_curve_public_output(result, ys, cfg_plain)
    @test out_plain == (a = 2.0, b = 1.0)

    # Covariance-only mode.
    _, _, cfg_cov = CausalSetZoology._fit_curve_prepare_inputs(ys, (:a, :b); return_cov = true)
    out_cov = CausalSetZoology._fit_curve_public_output(result, ys, cfg_cov)
    @test haskey(out_cov, :params)
    @test haskey(out_cov, :cov)
    @test haskey(out_cov, :stderr)
    @test !haskey(out_cov, :χ²)

    # Unweighted goodness-of-fit mode.
    _, _, cfg_gof = CausalSetZoology._fit_curve_prepare_inputs(ys, (:a, :b); goodness_of_fit = true)
    out_gof = CausalSetZoology._fit_curve_public_output(result, ys, cfg_gof)
    @test haskey(out_gof, :params)
    @test haskey(out_gof, :rel_residuals)
    @test !haskey(out_gof, :χ²)

    # Weighted goodness-of-fit + covariance mode.
    _, _, cfg_all = CausalSetZoology._fit_curve_prepare_inputs(
        ys,
        (:a, :b);
        stds = [0.1, 0.1, 0.1],
        goodness_of_fit = true,
        return_cov = true,
    )
    out_all = CausalSetZoology._fit_curve_public_output(result, ys, cfg_all)
    @test haskey(out_all, :params)
    @test haskey(out_all, :rel_residuals)
    @test haskey(out_all, :χ²)
    @test haskey(out_all, :cov)
    @test haskey(out_all, :stderr)
end

# Verifies unweighted/weighted objective behavior and bounds clamping effect.
@testitem "histogram_fitting helpers: objective" setup=[setupHistogramFitting] begin
    # Setup deterministic linear data so optimum is known exactly.
    xs = collect(1.0:8.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    stds = fill(0.2, length(xs))

    # Unweighted least-squares objective reaches zero at true parameters.
    obj = CausalSetZoology._fit_curve_objective(f, xs, ys, syms)
    @test obj([2.0, 1.0]) ≈ 0.0 atol = 1e-12
    @test obj([1.0, 1.0]) > 0

    # Weighted objective agrees on the same optimum.
    obj_w = CausalSetZoology._fit_curve_objective(
        f,
        xs,
        ys,
        syms;
        minimize_χ² = true,
        stds = stds,
    )
    @test obj_w([2.0, 1.0]) ≈ 0.0 atol = 1e-12

    # std_fn path should be usable and agree when it returns the provided stds.
    obj_w_fn = CausalSetZoology._fit_curve_objective(
        f,
        xs,
        ys,
        syms;
        minimize_χ² = true,
        stds = stds,
        std_fn = (y, yhat, s, p) -> s,
    )
    @test obj_w_fn([2.0, 1.0]) ≈ 0.0 atol = 1e-12

    # With identity std_fn, weighted residual objective should numerically match the plain weighted path.
    @test obj_w_fn([1.0, 1.0]) ≈ obj_w([1.0, 1.0]) atol = 1e-12
    
    # Controlled std_fn scaling: doubling σ should divide weighted sum-of-squares by 4.
    obj_w_fn_scaled = CausalSetZoology._fit_curve_objective(
        f,
        xs,
        ys,
        syms;
        minimize_χ² = true,
        stds = stds,
        std_fn = (y, yhat, s, p) -> 2 .* s,
    )
    @test obj_w_fn_scaled([1.0, 1.0]) ≈ obj_w([1.0, 1.0]) / 4 atol = 1e-12

    # Bounds are enforced via clamping inside the objective.
    bounds_vec = ([1.5, 0.5], [2.5, 1.5])
    obj_b = CausalSetZoology._fit_curve_objective(f, xs, ys, syms; bounds_vec = bounds_vec)
    @test obj_b([10.0, -5.0]) ≈ obj_b([2.5, 0.5]) atol = 1e-12
end

# Ensures a single optimization solve recovers linear-model parameters.
@testitem "histogram_fitting helpers: solve" setup=[setupHistogramFitting] begin
    # Solver should recover linear parameters from a poor initial guess.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))

    xopt, fmin = CausalSetZoology._fit_curve_solve(
        [0.0, 0.0],
        ys,
        f,
        xs,
        syms;
        method = Optim.NelderMead(),
    )
    @test xopt[1] ≈ 2.0 atol = 1e-2
    @test xopt[2] ≈ 1.0 atol = 1e-2
    @test fmin >= 0.0
end

# Validates score computation in unweighted and chi-squared modes, including dof edge case.
@testitem "histogram_fitting helpers: score" setup=[setupHistogramFitting] begin
    # Scores at true parameters should be exactly zero for noiseless data.
    xs = collect(1.0:8.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    stds = fill(0.2, length(xs))

    s0 = CausalSetZoology._fit_curve_score([2.0, 1.0], ys, f, xs, syms)
    @test s0 ≈ 0.0 atol = 1e-12

    sχ = CausalSetZoology._fit_curve_score([2.0, 1.0], ys, f, xs, syms; stds = stds)
    @test sχ ≈ 0.0 atol = 1e-12
    # std_fn override path: should match plain weighted score when std_fn returns stds unchanged.
    sχ_fn = CausalSetZoology._fit_curve_score(
        [2.0, 1.0],
        ys,
        f,
        xs,
        syms;
        stds = stds,
        std_fn = (y, yhat, s, p) -> s,
    )
    @test sχ_fn ≈ sχ atol = 1e-12

    # Controlled std_fn scaling: doubling σ should divide reduced chi-squared by 4.
    sχ_bad = CausalSetZoology._fit_curve_score([1.0, 1.0], ys, f, xs, syms; stds = stds)
    sχ_bad_scaled = CausalSetZoology._fit_curve_score(
        [1.0, 1.0],
        ys,
        f,
        xs,
        syms;
        stds = stds,
        std_fn = (y, yhat, s, p) -> 2 .* s,
    )
    @test sχ_bad_scaled ≈ sχ_bad / 4 atol = 1e-12

    # dof <= 0 branch: reduced chi-squared is undefined -> NaN.
    ys2 = ys[1:2]
    xs2 = xs[1:2]
    s_nan = CausalSetZoology._fit_curve_score([2.0, 1.0], ys2, f, xs2, syms; stds = stds[1:2])
    @test isnan(s_nan)
end

# Confirms multistart candidate generation for free and bounded initializations.
@testitem "histogram_fitting helpers: multistart candidate" setup=[setupHistogramFitting] begin
    # Free candidate has same dimension; bounded candidate stays inside bounds.
    rng = Random.Xoshiro(42)
    init = [1.0, 0.0]

    c_free = CausalSetZoology._fit_curve_multistart_candidate(init, nothing, rng)
    @test length(c_free) == 2

    # For finite bounds, candidate components are sampled as lo + rand() * (hi - lo).
    bounds_vec = ([-1.0, -2.0], [3.0, 2.0])
    rng_bounded = Random.Xoshiro(123)
    c_bounded = CausalSetZoology._fit_curve_multistart_candidate(init, bounds_vec, rng_bounded)
    rng_expected = Random.Xoshiro(123)
    expected_bounded = [
        bounds_vec[1][1] + rand(rng_expected) * (bounds_vec[2][1] - bounds_vec[1][1]),
        bounds_vec[1][2] + rand(rng_expected) * (bounds_vec[2][2] - bounds_vec[1][2]),
    ]
    @test bounds_vec[1][1] <= c_bounded[1] <= bounds_vec[2][1]
    @test bounds_vec[1][2] <= c_bounded[2] <= bounds_vec[2][2]
    @test c_bounded ≈ expected_bounded atol = 1e-15
end

# Checks finite-difference Jacobian against analytic derivatives for a linear model.
@testitem "histogram_fitting helpers: finite-difference jacobian" setup=[setupHistogramFitting] begin
    # Finite-difference Jacobian should match analytic derivatives for linear model.
    xs = collect(1.0:6.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    params = (a = 2.0, b = 1.0)

    J = CausalSetZoology._fit_curve_jacobian_fd(xs, params, f, syms)
    @test size(J) == (length(xs), 2)
    @test J[:, 1] ≈ xs rtol = 1e-5 atol = 1e-7
    @test J[:, 2] ≈ ones(length(xs)) rtol = 1e-5 atol = 1e-7

    # Explicit eps path: for a linear model, derivatives remain exact across step sizes.
    J_small = CausalSetZoology._fit_curve_jacobian_fd(xs, params, f, syms; eps = 1e-8)
    J_large = CausalSetZoology._fit_curve_jacobian_fd(xs, params, f, syms; eps = 1e-2)
    @test J_small[:, 1] ≈ xs rtol = 1e-5 atol = 1e-7
    @test J_small[:, 2] ≈ ones(length(xs)) rtol = 1e-5 atol = 1e-7
    @test J_large[:, 1] ≈ xs rtol = 1e-5 atol = 1e-7
    @test J_large[:, 2] ≈ ones(length(xs)) rtol = 1e-5 atol = 1e-7

    # Zero-parameter branch: when parameter is zero, step falls back to 1.0 * eps.
    params_zero = (a = 2.0, b = 0.0)
    J_zero = CausalSetZoology._fit_curve_jacobian_fd(xs, params_zero, f, syms; eps = 1e-6)
    @test J_zero[:, 2] ≈ ones(length(xs)) rtol = 1e-5 atol = 1e-7
end

# Tests covariance/stderr estimation via local linearization (weighted and unweighted).
@testitem "histogram_fitting helpers: covariance linearized structure" setup=[setupHistogramFitting] begin
    # Local linearization covariance should be symmetric with named stderr output.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    params_ref = (a = 2.0, b = 1.0)
    ys = f.(xs, Ref(params_ref))
    stds = fill(0.2, length(xs))

    solve_fn = (x0, ys_local) -> CausalSetZoology._fit_curve_solve(
        x0,
        ys_local,
        f,
        xs,
        syms;
        method = Optim.NelderMead(),
    )

    # Check both unweighted and weighted covariance paths.
    cov_u, stderr_u = CausalSetZoology._fit_curve_cov_and_stderr(
        xs,
        ys,
        params_ref,
        [2.0, 1.0],
        solve_fn,
        f,
        syms,
    )
    @test size(cov_u) == (2, 2)
    @test cov_u ≈ cov_u' atol = 1e-12
    @test haskey(stderr_u, :a)
    @test haskey(stderr_u, :b)

    cov_w, stderr_w = CausalSetZoology._fit_curve_cov_and_stderr(
        xs,
        ys,
        params_ref,
        [2.0, 1.0],
        solve_fn,
        f,
        syms;
        stds = stds,
    )
    @test size(cov_w) == (2, 2)
    @test cov_w ≈ cov_w' atol = 1e-12
    @test haskey(stderr_w, :a)
    @test haskey(stderr_w, :b)
end

# Uses closed-form covariance formulas to validate the linearized implementation numerically.
@testitem "histogram_fitting helpers: covariance linearized explicit formula" setup=[setupHistogramFitting] begin
    # Explicit covariance formula check on controlled noisy data:
    # cov = inv(J'J)*s² (unweighted), cov = inv(J'WJ)*s² (weighted).
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    params_ref = (a = 2.0, b = 1.0)
    ys = f.(xs, Ref(params_ref))
    stds = fill(0.2, length(xs))
    solve_fn = (x0, ys_local) -> CausalSetZoology._fit_curve_solve(
        x0,
        ys_local,
        f,
        xs,
        syms;
        method = Optim.NelderMead(),
    )
    δ = [0.1, -0.2, 0.0, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.05]
    ys_noisy = ys .+ δ
    cov_u_n, stderr_u_n = CausalSetZoology._fit_curve_cov_and_stderr(
        xs,
        ys_noisy,
        params_ref,
        [2.0, 1.0],
        solve_fn,
        f,
        syms,
    )
    J = hcat(xs, ones(length(xs)))
    dof = length(xs) - length(syms)
    s2_u = sum((ys_noisy .- ys) .^ 2) / dof
    cov_u_expected = inv(J' * J) * s2_u
    @test cov_u_n ≈ cov_u_expected atol = 1e-10
    @test stderr_u_n.a ≈ sqrt(abs(LinearAlgebra.diag(cov_u_expected)[1])) atol = 1e-10
    @test stderr_u_n.b ≈ sqrt(abs(LinearAlgebra.diag(cov_u_expected)[2])) atol = 1e-10

    cov_w_n, stderr_w_n = CausalSetZoology._fit_curve_cov_and_stderr(
        xs,
        ys_noisy,
        params_ref,
        [2.0, 1.0],
        solve_fn,
        f,
        syms;
        stds = stds,
    )
    W = LinearAlgebra.Diagonal(1.0 ./ (stds .^ 2))
    s2_w = sum(((ys_noisy .- ys) ./ stds) .^ 2) / dof
    cov_w_expected = inv(J' * W * J) * s2_w
    @test cov_w_n ≈ cov_w_expected atol = 1e-10
    @test stderr_w_n.a ≈ sqrt(abs(LinearAlgebra.diag(cov_w_expected)[1])) atol = 1e-10
    @test stderr_w_n.b ≈ sqrt(abs(LinearAlgebra.diag(cov_w_expected)[2])) atol = 1e-10
end

# Tests covariance/stderr estimation via bootstrap resampling with fixed RNG.
@testitem "histogram_fitting helpers: covariance bootstrap" setup=[setupHistogramFitting] begin
    # Bootstrap covariance path with fixed RNG should return valid covariance/stderr.
    rng = Random.Xoshiro(42)
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    stds = fill(0.2, length(xs))

    solve_fn = (x0, ys_local) -> CausalSetZoology._fit_curve_solve(
        x0,
        ys_local,
        f,
        xs,
        syms;
        method = Optim.NelderMead(),
    )

    cov_b, stderr_b = CausalSetZoology._fit_curve_cov_and_stderr(
        xs,
        ys,
        (a = 2.0, b = 1.0),
        [2.0, 1.0],
        solve_fn,
        f,
        syms;
        stds = stds,
        bootstrap_errorbars = true,
        n_boot = 12,
        rng = rng,
    )
    @test size(cov_b) == (2, 2)
    @test cov_b ≈ cov_b' atol = 1e-12
    @test haskey(stderr_b, :a)
    @test haskey(stderr_b, :b)
end

# Uses a deterministic solve callback so bootstrap covariance is known exactly (all-zero).
@testitem "histogram_fitting helpers: covariance bootstrap explicit" setup=[setupHistogramFitting] begin
    xs = collect(1.0:8.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    stds = fill(0.2, length(xs))

    # Deterministic solver ignores bootstrap sample and always returns same parameter vector.
    solve_const = (x0, ys_local) -> ([2.0, 1.0], 0.0)
    cov_z, stderr_z = CausalSetZoology._fit_curve_cov_and_stderr(
        xs,
        ys,
        (a = 2.0, b = 1.0),
        [2.0, 1.0],
        solve_const,
        f,
        syms;
        stds = stds,
        bootstrap_errorbars = true,
        n_boot = 12,
        rng = Random.Xoshiro(99),
    )
    @test cov_z ≈ zeros(2, 2) atol = 1e-15
    @test stderr_z.a ≈ 0.0 atol = 1e-15
    @test stderr_z.b ≈ 0.0 atol = 1e-15
end

# Validates throw paths for invalid bootstrap covariance configurations.
@testitem "histogram_fitting helpers: covariance validation" setup=[setupHistogramFitting] begin
    # Invalid bootstrap configuration should throw the documented errors.
    rng = Random.Xoshiro(42)
    xs = collect(1.0:8.0)
    f(x, p) = p.a * x + p.b
    syms = (:a, :b)
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))

    solve_fn = (x0, ys_local) -> CausalSetZoology._fit_curve_solve(
        x0,
        ys_local,
        f,
        xs,
        syms;
        method = Optim.NelderMead(),
    )

    @test_throws ArgumentError CausalSetZoology._fit_curve_cov_and_stderr(
        xs,
        ys,
        (a = 2.0, b = 1.0),
        [2.0, 1.0],
        solve_fn,
        f,
        syms;
        bootstrap_errorbars = true,
        n_boot = 8,
        rng = rng,
    )

    @test_throws DomainError CausalSetZoology._fit_curve_cov_and_stderr(
        xs,
        ys,
        (a = 2.0, b = 1.0),
        [2.0, 1.0],
        solve_fn,
        f,
        syms;
        stds = fill(0.2, length(xs)),
        bootstrap_errorbars = true,
        n_boot = 1,
        rng = rng,
    )
end

# End-to-end fit_curve smoke test on noiseless linear data.
@testitem "histogram_fitting: fit_curve basic" setup=[setupHistogramFitting] begin
    # End-to-end fit should recover known linear parameters.
    rng = Random.Xoshiro(123)
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))

    p = CausalSetZoology.fit_curve(
        ys,
        f,
        (:a, :b);
        x_values = xs,
        init = (a = 1.0, b = 0.0),
        multistart = 3,
        rng = rng,
    )
    @test p.a ≈ 2.0 atol = 1e-2
    @test p.b ≈ 1.0 atol = 1e-2
end

# Verifies goodness-of-fit diagnostics payload from fit_curve options.
@testitem "histogram_fitting: fit_curve diagnostics goodness-of-fit" setup=[setupHistogramFitting] begin
    # Goodness-of-fit payload should include χ² and residuals.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    stds = fill(0.1, length(ys))

    gof = CausalSetZoology.fit_curve(
        ys,
        f,
        (:a, :b);
        x_values = xs,
        stds = stds,
        minimize_χ² = true,
        goodness_of_fit = true,
        init = (a = 1.5, b = 0.5),
    )
    @test haskey(gof, :params)
    @test haskey(gof, :rel_residuals)
    @test haskey(gof, :χ²)
    @test length(gof.rel_residuals) == length(ys)
end

# Verifies covariance return payload from fit_curve options.
@testitem "histogram_fitting: fit_curve diagnostics covariance" setup=[setupHistogramFitting] begin
    # Covariance payload should include covariance matrix and stderr NamedTuple.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    cov = CausalSetZoology.fit_curve(
        ys,
        f,
        (:a, :b);
        x_values = xs,
        return_cov = true,
        init = (a = 1.0, b = 0.0),
    )
    @test haskey(cov, :params)
    @test haskey(cov, :cov)
    @test haskey(cov, :stderr)
    @test size(cov.cov) == (2, 2)
end

# Verifies combined weighted diagnostics and covariance payload from fit_curve options.
@testitem "histogram_fitting: fit_curve diagnostics combined" setup=[setupHistogramFitting] begin
    # Combined diagnostics path: weighted goodness-of-fit together with covariance output.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    stds = fill(0.1, length(ys))
    gof_cov = CausalSetZoology.fit_curve(
        ys,
        f,
        (:a, :b);
        x_values = xs,
        stds = stds,
        minimize_χ² = true,
        goodness_of_fit = true,
        return_cov = true,
        init = (a = 1.0, b = 0.0),
    )
    @test haskey(gof_cov, :params)
    @test haskey(gof_cov, :rel_residuals)
    @test haskey(gof_cov, :χ²)
    @test haskey(gof_cov, :cov)
    @test haskey(gof_cov, :stderr)
end

# Validates fit_curve argument checking and error handling.
@testitem "histogram_fitting: fit_curve validation" setup=[setupHistogramFitting] begin
    # Input/option validation for fit_curve.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))

    @test_throws ArgumentError CausalSetZoology.fit_curve(ys, f, (:a, :b); std_fn = (y, yhat, s, p) -> s)
    @test_throws DomainError CausalSetZoology.fit_curve(ys, f, (:a, :b); ϵ = 0.0)
    @test_throws ArgumentError CausalSetZoology.fit_curve(ys, f, (:a, :b); minimize_χ² = true)
    @test_throws ArgumentError CausalSetZoology.fit_curve(ys, f, (:a, :b); stds = [0.1], minimize_χ² = true)
    @test_throws ArgumentError CausalSetZoology.fit_curve(ys, f, (:a, :b); multistart = 0)
    @test_throws TypeError CausalSetZoology.fit_curve(ys, f, (:a, :b); bounds = (1.0, 2.0))
    @test_throws DimensionMismatch CausalSetZoology.fit_curve(ys, f, (:a, :b); x_values = xs[1:3])

    @test_throws ArgumentError CausalSetZoology.fit_curve(
        ys,
        f,
        (:a, :b);
        return_cov = true,
        bootstrap_errorbars = true,
        n_boot = 8,
    )

    @test_throws DomainError CausalSetZoology.fit_curve(
        ys,
        f,
        (:a, :b);
        stds = fill(0.1, length(ys)),
        return_cov = true,
        bootstrap_errorbars = true,
        n_boot = 1,
    )
end

# End-to-end bin-slice wrapper parameter recovery.
@testitem "histogram_fitting: fit_histogram_bins basic" setup=[setupHistogramFitting] begin
    # Bin-slice wrapper should recover parameters on noiseless linear data.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))

    fit = CausalSetZoology.fit_histogram_bins(
        ys,
        f,
        (:a, :b),
        3,
        8;
        x_values = xs,
        init = (a = 1.0, b = 0.0),
    )
    @test fit.a ≈ 2.0 atol = 1e-2
    @test fit.b ≈ 1.0 atol = 1e-2
end

# Verifies wrapper equivalence against direct fit_curve on the sliced data.
@testitem "histogram_fitting: fit_histogram_bins wrapper equivalence" setup=[setupHistogramFitting] begin
    # Wrapper equivalence: fitting through bin wrapper matches direct fit on the same slice.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    fit = CausalSetZoology.fit_histogram_bins(
        ys,
        f,
        (:a, :b),
        3,
        8;
        x_values = xs,
        init = (a = 1.0, b = 0.0),
    )
    ys_slice = ys[3:8]
    xs_slice = xs[3:8]
    fit_direct = CausalSetZoology.fit_curve(
        ys_slice,
        f,
        (:a, :b);
        x_values = xs_slice,
        init = (a = 1.0, b = 0.0),
    )
    @test fit.a ≈ fit_direct.a atol = 1e-10
    @test fit.b ≈ fit_direct.b atol = 1e-10
end

# Verifies goodness-of-fit diagnostics path through fit_histogram_bins.
@testitem "histogram_fitting: fit_histogram_bins diagnostics" setup=[setupHistogramFitting] begin
    # Goodness-of-fit payload should include fit params and chi-squared.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))
    gof = CausalSetZoology.fit_histogram_bins(
        ys,
        f,
        (:a, :b),
        2,
        9;
        x_values = xs,
        stds = fill(0.1, length(ys)),
        minimize_χ² = true,
        goodness_of_fit = true,
    )
    @test haskey(gof, :params)
    @test haskey(gof, :χ²)
end

# Validates fit_histogram_bins range and input-length error handling.
@testitem "histogram_fitting: fit_histogram_bins validation" setup=[setupHistogramFitting] begin
    # Bin-range and length validations for the wrapper API.
    xs = collect(1.0:10.0)
    f(x, p) = p.a * x + p.b
    ys = f.(xs, Ref((a = 2.0, b = 1.0)))

    @test_throws ArgumentError CausalSetZoology.fit_histogram_bins(ys, f, (:a, :b), 0, 3)
    @test_throws ArgumentError CausalSetZoology.fit_histogram_bins(ys, f, (:a, :b), 4, 3)
    @test_throws ArgumentError CausalSetZoology.fit_histogram_bins(ys, f, (:a, :b), 1, 20)
    @test_throws ArgumentError CausalSetZoology.fit_histogram_bins(ys, f, (:a, :b), 2, 6; stds = [0.1])
    @test_throws ArgumentError CausalSetZoology.fit_histogram_bins(ys, f, (:a, :b), 2, 6; x_values = xs[1:3])
end
