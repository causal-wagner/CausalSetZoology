@testsnippet setupMakeAnalysisDataset begin
    using Test
    using JLD2
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
        cmd_env = setenv(cmd, Dict("JULIA_LOAD_PATH" => lp))
        proc = run(pipeline(ignorestatus(cmd_env), stdout = out, stderr = out); wait = false)
        close(out.in)
        text = read(out, String)
        wait(proc)
        return proc.exitcode, text
    end

    root_dir = normpath(joinpath(@__DIR__, "..", ".."))
    src_project = joinpath(root_dir, "src")
    test_project = joinpath(root_dir, "test")
    sep = Sys.iswindows() ? ";" : ":"
    scripts_load_path = test_project * sep * src_project * sep * "@stdlib"

    function _run_dataset_script(; kind::String, size::Int, N::Int, batchsize::Int, seed::Int, num_processes::Int)
        tmp = mktempdir()
        out_path = joinpath(tmp, "dataset.jld2")
        cmd = `$(Base.julia_cmd()) $dataset_script --out $out_path --N $N --kind $kind --size $size --batchsize $batchsize --seed $seed --num_processes $num_processes`
        code, out = _run_capture(setenv(cmd, Dict("JULIA_LOAD_PATH" => scripts_load_path)))
        return out_path, code, out
    end

    function _load_all_adjs(path::String)
        JLD2.jldopen(path, "r") do f
            nbatches = f["meta/nbatches"]
            adjs = BitMatrix[]
            for b in 1:nbatches
                append!(adjs, f["batches/$b/adjs"])
            end
            return adjs
        end
    end

    dataset_script = joinpath(@__DIR__, "..", "..", "src", "data_generation", "make_analysis_dataset.jl")
    dataset_seq_script = joinpath(@__DIR__, "..", "..", "src", "data_generation", "make_analysis_dataset_sequential.jl")

    dataset_src = read(dataset_script, String)
    dataset_seq_src = read(dataset_seq_script, String)
end

@testitem "make_analysis_dataset: transitive reduction helper" setup=[setupMakeAnalysisDataset] begin
    block = _extract_block(
        dataset_src,
        "@everywhere function transitive_reduction!(mat::AbstractMatrix)",
        "################################################################################",
    )
    block = replace(block, "@everywhere " => "")
    m = _eval_block_in_module(block)

    mat = [0 1 1; 0 0 1; 0 0 0]
    m.transitive_reduction!(mat)
    @test mat == [0 1 0; 0 0 1; 0 0 0]

    mat2 = [0 1 0 1; 0 0 1 1; 0 0 0 1; 0 0 0 0]
    m.transitive_reduction!(mat2)
    @test mat2 == [0 1 0 0; 0 0 1 0; 0 0 0 1; 0 0 0 0]
end

@testitem "make_analysis_dataset(_sequential): CLI guards and help" setup=[setupMakeAnalysisDataset] begin
    c1, out1 = _run_capture(`$(Base.julia_cmd()) $dataset_script --help`)
    @test c1 == 0
    @test occursin("Usage: julia make_analysis_dataset.jl", out1)

    c2, out2 = _run_capture(`$(Base.julia_cmd()) $dataset_seq_script`)
    @test c2 != 0
    @test occursin("Error: --out is required.", out2)

    c3, out3 = _run_capture(`$(Base.julia_cmd()) $dataset_script --out /tmp/x.jld2 --N 1 --kind merged --size 8 --link_probability 1.1`)
    @test c3 != 0
    @test occursin("Error: --link_probability must be between 0.0 and 1.0.", out3)
end

@testitem "make_analysis_dataset(_sequential): source contracts for generation branches" setup=[setupMakeAnalysisDataset] begin
    kinds = [
        "minkowski_sprinkling",
        "minkowski_quasicrystal",
        "manifoldlike_simply_connected",
        "manifoldlike_non_simply_connected",
        "destroyed",
        "merged",
        "grid",
        "random",
        "layered",
    ]

    for kind in kinds
        @test occursin("kind == \"$kind\"", dataset_src)
    end

    # Sequential script should still cover representative special branches.
    for kind in ["random", "layered", "merged"]
        @test occursin("kind == \"$kind\"", dataset_seq_src)
    end

    for key in ["meta/batchsize", "meta/nbatches", "meta/N", "meta/config", "batches/\$b/csets", "batches/\$b/adjs", "batches/\$b/links"]
        @test occursin(key, dataset_src)
        @test occursin(key, dataset_seq_src)
    end

    @test occursin("function generate_batch(", dataset_src)
    @test occursin("if nprocs() - 1 < num_workers", dataset_src)
    @test occursin("Random.seed!(seed)", dataset_seq_src)
end

@testitem "make_analysis_dataset: multiprocessing batches produce distinct csets" setup=[setupMakeAnalysisDataset] begin
    # Check uniqueness across samples when using script multiprocessing + batching.
    out_path, code, out = _run_dataset_script(
        kind = "grid",
        size = 16,
        N = 8,
        batchsize = 2,
        seed = 11,
        num_processes = 2,
    )
    @test code == 0
    @test isfile(out_path)
    @test !isempty(out)
    (code == 0 && isfile(out_path)) || return

    adjs = _load_all_adjs(out_path)
    @test length(adjs) == 8
    @test length(unique(adjs)) == length(adjs)
end

@testitem "make_analysis_dataset: multiprocessing seed reproducibility and divergence" setup=[setupMakeAnalysisDataset] begin
    # Same seed should reproduce the exact sample sequence.
    p1, c1, _ = _run_dataset_script(
        kind = "grid",
        size = 16,
        N = 6,
        batchsize = 2,
        seed = 77,
        num_processes = 2,
    )
    p2, c2, _ = _run_dataset_script(
        kind = "grid",
        size = 16,
        N = 6,
        batchsize = 2,
        seed = 77,
        num_processes = 2,
    )
    @test c1 == 0
    @test c2 == 0
    @test isfile(p1)
    @test isfile(p2)
    (c1 == 0 && c2 == 0 && isfile(p1) && isfile(p2)) || return

    adjs1 = _load_all_adjs(p1)
    adjs2 = _load_all_adjs(p2)
    @test length(adjs1) == length(adjs2) == 6
    @test all(adjs1[i] == adjs2[i] for i in eachindex(adjs1))

    # Different seed should change at least one generated sample.
    p3, c3, _ = _run_dataset_script(
        kind = "grid",
        size = 16,
        N = 6,
        batchsize = 2,
        seed = 78,
        num_processes = 2,
    )
    @test c3 == 0
    @test isfile(p3)
    (c3 == 0 && isfile(p3)) || return
    adjs3 = _load_all_adjs(p3)
    @test any(adjs1[i] != adjs3[i] for i in eachindex(adjs1))
end
