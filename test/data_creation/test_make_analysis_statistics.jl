@testsnippet setupMakeAnalysisStatistics begin
    using Test
    import CausalSetZoology

    function _extract_block(src::AbstractString, start_marker::AbstractString, end_marker::AbstractString)
        i = findfirst(start_marker, src)
        i === nothing && error("Start marker not found: $start_marker")
        j = findnext(end_marker, src, last(i) + 1)
        j === nothing && error("End marker not found: $end_marker")
        return src[first(i):first(j)-1]
    end

    function _eval_block_in_module(src::AbstractString; module_name::Symbol = gensym(:ScriptBlock))
        m = Module(module_name)
        Core.eval(m, :(import CausalSets))
        Core.eval(m, :(using LinearAlgebra))
        Core.eval(m, :(using Statistics))
        Core.eval(m, Meta.parseall(src))
        return m
    end

    function _run_capture(cmd::Cmd)
        out = Pipe()
        sep = Sys.iswindows() ? ";" : ":"
        base_lp = get(ENV, "JULIA_LOAD_PATH", "@")
        lp = occursin("@stdlib", base_lp) ? base_lp : string(base_lp, sep, "@stdlib")
        cmd_env = setenv(
            cmd,
            "JULIA_LOAD_PATH" => lp,
        )
        proc = run(pipeline(ignorestatus(cmd_env), stdout = out, stderr = out); wait = false)
        close(out.in)
        text = read(out, String)
        wait(proc)
        return proc.exitcode, text
    end

    stats_script = joinpath(@__DIR__, "..", "..", "src", "data_generation", "make_analysis_statistics.jl")
    stats_src = read(stats_script, String)
    utils_script = joinpath(@__DIR__, "..", "..", "src", "data_generation", "utils.jl")
    utils_src = read(utils_script, String)
end

@testitem "make_analysis_statistics: pure helper functions" setup=[setupMakeAnalysisStatistics] begin
    sparse_hist_block = _extract_block(
        stats_src,
        "@everywhere function sparse_hist(v)",
        "@everywhere function ev_summary(ev)",
    )
    ev_summary_block = _extract_block(
        stats_src,
        "@everywhere function ev_summary(ev)",
        "@everywhere function compute(",
    )
    lap_block = utils_src

    block = replace(sparse_hist_block * ev_summary_block, "@everywhere " => "") * lap_block
    m = _eval_block_in_module(block)

    @test m.sparse_hist([0, 2, 0, 3]) == Dict(2 => 2, 4 => 3)

    ev = [-2.0, 0.0, 2.0]
    summary = m.ev_summary(ev)
    @test summary[2] == 1
    @test summary[3] == 2.0
    @test summary[4] == -2.0
    @test summary[5] == 2.0

    W = [0.0 1.0; 1.0 0.0]
    vals = m.sym_norm_lap_eigs!(W)
    @test length(vals) == 2
    @test all(isfinite, vals)
    @test vals[1] ≈ 0.0 atol = 1e-10
    @test vals[2] ≈ 2.0 atol = 1e-10

    M = [1.0 2.0; 3.0 4.0]
    m.symmetrize!(M)
    @test M == [2.0 5.0; 5.0 8.0]

    U = [9.0 2.0 3.0; 7.0 8.0 4.0; 6.0 5.0 1.0]
    m.symmetrize_strictly_upper_triangular!(U)
    @test U == [0.0 2.0 3.0; 2.0 0.0 4.0; 3.0 4.0 0.0]

    # Causet-like overload: must agree with matrix path built from future_relations.
    Core.eval(m, :(struct MockBitArrayCauset; future_relations; end))
    cset = m.MockBitArrayCauset([BitVector([0, 1]), BitVector([0, 0])])
    vals_from_cset = m.sym_norm_lap_eigs!(cset)

    adj = transpose(reduce(hcat, cset.future_relations))
    W_sym = Float64.(adj)
    W_sym .+= transpose(W_sym)
    vals_from_matrix = m.sym_norm_lap_eigs!(W_sym)
    @test vals_from_cset ≈ vals_from_matrix atol = 1e-12
end

@testitem "make_analysis_statistics: CLI help and required flag paths" setup=[setupMakeAnalysisStatistics] begin
    c1, out1 = _run_capture(`$(Base.julia_cmd()) $stats_script --help`)
    @test c1 == 0
    @test occursin("Usage: julia make_analysis_statistics.jl", out1)

    # This script does not validate required args before first use; verify failure is explicit.
    c2, out2 = _run_capture(`$(Base.julia_cmd()) $stats_script`)
    @test c2 != 0
    @test occursin("UndefVarError", out2)
end

@testitem "make_analysis_statistics: source contracts for kinds and output metadata" setup=[setupMakeAnalysisStatistics] begin
    for kind in [
        "manifoldlike_simply_connected",
        "manifoldlike_non_simply_connected",
        "minkowski_quasicrystal",
        "destroyed",
        "merged",
        "grid",
        "layered",
    ]
        @test occursin("kind == \"$kind\"", stats_src)
    end

    for key in ["meta/batchsize", "meta/nbatches", "meta/N", "batches/\$b/csets", "batches/\$b/adjs", "batches/\$b/links"]
        @test occursin(key, stats_src)
    end

    for helper in ["connectivity(adj, size)", "function sparse_hist(v)", "function ev_summary(ev)", "function compute("]
        @test occursin(helper, stats_src)
    end
    @test occursin("include(joinpath(@__DIR__, \"utils.jl\"))", stats_src)
    @test occursin("function sym_norm_lap_eigs!(W::AbstractMatrix", utils_src)
end
