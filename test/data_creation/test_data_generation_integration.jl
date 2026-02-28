@testsnippet setupDataGenerationIntegration begin
    using Test
    import CausalSetZoology
    using JLD2
    import QuantumGrav

    function _run_capture(cmd::Cmd)
        out = Pipe()
        proc = run(pipeline(ignorestatus(cmd), stdout = out, stderr = out); wait = false)
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

    dataset_seq_script = joinpath(root_dir, "src", "data_generation", "make_analysis_dataset_sequential.jl")
    stats_script = joinpath(root_dir, "src", "data_generation", "make_analysis_statistics.jl")
    quasicrystal_path_literal = "/Volumes/Causal Set Silo/causal_sets/crystals/spacetime_quasicrystal_5e8.jld2"

    function _build_temp_quasicrystal_file(; min_points::Int = 24)
        αin = Float64[]
        αout = Float64[]
        for ρ in (0.75, 1.0, 1.5, 2.0)
            αin, αout = QuantumGrav.quasicrystal(ρ)
            length(αin) >= min_points && break
        end
        length(αin) >= min_points || error("Could not build large-enough quasicrystal sample for tests.")
        if !(length(αin) == length(αout))
            throw(ArgumentError("assertion failed: length(αin) == length(αout)"))
        end
        tmp = mktempdir()
        path = joinpath(tmp, "test_quasicrystal.jld2")
        JLD2.jldopen(path, "w") do f
            f["big_set"] = (αin, αout)
        end
        return path
    end

    function _patched_seq_script_for_quasicrystal(quasicrystal_file::String)
        src = read(dataset_seq_script, String)
        patched = replace(src, quasicrystal_path_literal => quasicrystal_file)
        tmp = mktempdir()
        script_path = joinpath(tmp, "make_analysis_dataset_sequential_quasicrystal_test.jl")
        write(script_path, patched)
        return script_path
    end

    temp_quasicrystal_file = _build_temp_quasicrystal_file()
    dataset_seq_script_quasi = _patched_seq_script_for_quasicrystal(temp_quasicrystal_file)

    function _run_script(cmd::Cmd)
        code, out = _run_capture(setenv(cmd, Dict("JULIA_LOAD_PATH" => scripts_load_path)))
        return code, out
    end

    function _make_dataset(kind::String; size::Int=8, extra_args::Vector{String}=String[])
        tmp = mktempdir()
        dataset_path = joinpath(tmp, "dataset_$(kind).jld2")
        dataset_script = kind == "minkowski_quasicrystal" ? dataset_seq_script_quasi : dataset_seq_script
        cmd = `$(Base.julia_cmd()) $dataset_script --kind $kind --size $size --N 2 --seed 1 --batchsize 1 --out $dataset_path`
        !isempty(extra_args) && (cmd = `$cmd $(extra_args...)`)
        code, out = _run_script(cmd)
        return dataset_path, code, out
    end

    function _make_statistics(dataset_path::String)
        stats_path = dataset_path * ".stats.jld2"
        cmd = `$(Base.julia_cmd()) $stats_script --in $dataset_path --out $stats_path --num_processes 1`
        code, out = _run_script(cmd)
        return stats_path, code, out
    end
end

@testitem "data_generation integration: dataset + statistics tiny runs by kind" setup=[setupDataGenerationIntegration] begin
    dataset_base_keys = Set(["csets", "adjs", "links"])
    stats_base_fields = Ref{Union{Nothing, Set{Symbol}}}(nothing)

    kinds = [
        (
            kind = "minkowski_sprinkling",
            size = 8,
            extra = ["--D", "2"],
            dataset_keys = String[],
            stats_keys = String[],
        ),
        (
            kind = "minkowski_quasicrystal",
            size = 8,
            extra = String[],
            dataset_keys = ["trans_in", "trans_out"],
            stats_keys = ["trans_in", "trans_out"],
        ),
        (
            kind = "manifoldlike_simply_connected",
            size = 8,
            extra = ["--D", "2"],
            dataset_keys = ["r", "order"],
            stats_keys = ["r", "order"],
        ),
        (
            kind = "manifoldlike_non_simply_connected",
            size = 8,
            extra = ["--cut_restriction", "boundary_cuts"],
            dataset_keys = ["r", "order", "num_boundary_cuts", "genus"],
            stats_keys = ["r", "order", "num_boundary_cuts", "genus"],
        ),
        (
            kind = "destroyed",
            size = 8,
            extra = String[],
            dataset_keys = ["r", "order", "rel_num_flips"],
            stats_keys = ["r", "order", "rel_num_flips"],
        ),
        (
            kind = "merged",
            size = 8,
            extra = ["--link_probability", "0.5"],
            dataset_keys = ["r", "order", "rel_size_KR", "link_probability"],
            stats_keys = ["r", "order", "rel_size_KR", "link_probability"],
        ),
        (
            kind = "grid",
            size = 8,
            extra = String[],
            dataset_keys = ["r", "order", "segment_ratio", "segment_angle", "rotation_angle", "lattice"],
            stats_keys = ["r", "order", "segment_ratio", "segment_angle", "rotation_angle", "lattice"],
        ),
        (
            kind = "random",
            size = 8,
            extra = String[],
            dataset_keys = String[],
            stats_keys = String[],
        ),
        (
            kind = "layered",
            size = 64,
            extra = String[],
            dataset_keys = ["num_layers", "std"],
            stats_keys = ["num_layers", "standard_dev"],
        ),
    ]

    total_kinds = length(kinds)
    for (idx, spec) in enumerate(kinds)
        @info "integration progress: starting kind" index=idx total=total_kinds kind=spec.kind
        dataset_path, dataset_code, dataset_out = _make_dataset(spec.kind; size = spec.size, extra_args = spec.extra)
        @test dataset_code == 0
        dataset_code == 0 || continue
        @test isfile(dataset_path)
        @info "integration progress: dataset created" kind=spec.kind path=dataset_path

        JLD2.jldopen(dataset_path, "r") do f
            @test f["meta/N"] == 2
            @test f["meta/nbatches"] == 2
            @test f["meta/batchsize"] == 1
            @test f["meta/config"]["kind"] == spec.kind

            csets = f["batches/1/csets"]
            adjs = f["batches/1/adjs"]
            links = f["batches/1/links"]
            @test length(csets) == 1
            @test length(adjs) == 1
            @test length(links) == 1

            batch1_keys = Set(String(name) for name in keys(f["batches/1"]))
            expected_dataset_keys = union(dataset_base_keys, Set(spec.dataset_keys))
            @test batch1_keys == expected_dataset_keys

            for k in spec.dataset_keys
                @test haskey(f, "batches/1/$k")
                @test length(f["batches/1/$k"]) == 1
            end
        end

        stats_path, stats_code, stats_out = _make_statistics(dataset_path)
        @test stats_code == 0
        stats_code == 0 || continue
        @test isfile(stats_path)
        @info "integration progress: statistics created" kind=spec.kind path=stats_path

        JLD2.jldopen(stats_path, "r") do f
            @test f["meta/N"] == 2
            @test f["meta/nbatches"] == 2
            @test f["meta/batchsize"] == 1

            recs = f["batches/1"]
            @test length(recs) == 1
            rec = recs[1]
            rec_fields = Set(fieldnames(typeof(rec)))

            # One representative per base observable family from make_analysis_statistics.compute.
            base_representatives = (
                :n,
                :in_degree_hist,
                :out_degree_hist,
                :in_degree_hist_link,
                :out_degree_hist_link,
                :max_pathlen_hist,
                :max_pathlen_hist_link,
                :num_sources,
                :num_sinks,
                :num_sources_link,
                :num_sinks_link,
                :ev_sym_link,
                :connectivity,
                :cardinalities_hist,
                :chain_cardinalities_2_hist,
            )
            for fld in base_representatives
                @test hasfield(typeof(rec), fld)
            end

            extra_fields = Set(Symbol(k) for k in spec.stats_keys)
            if stats_base_fields[] === nothing
                stats_base_fields[] = setdiff(rec_fields, extra_fields)
            end
            @test rec_fields == union(stats_base_fields[], extra_fields)

            for k in spec.stats_keys
                @test hasfield(typeof(rec), Symbol(k))
            end
        end

        @test !isempty(dataset_out)
        @test !isempty(stats_out)
        @info "integration progress: completed kind" index=idx total=total_kinds kind=spec.kind
    end
end
