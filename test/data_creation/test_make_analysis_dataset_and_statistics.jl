@testsnippet setupMakeAnalysisDatasetAndStatistics begin
    using Test
    include(joinpath(@__DIR__, "test_support.jl"))

    pipeline_script = joinpath(@__DIR__, "..", "..", "src", "data_generation", "make_analysis_dataset_and_statistics.jl")
    pipeline_src = read(pipeline_script, String)
end

@testitem "make_analysis_dataset_and_statistics: arg parser help" setup=[setupMakeAnalysisDatasetAndStatistics] begin
    c, out = _run_capture(`$(Base.julia_cmd()) $pipeline_script --help`)
    @test c == 0
    @test occursin("--kind", out)
    @test occursin("--size", out)
    @test occursin("--num_csets", out)
    @test occursin("--outdir", out)
end

@testitem "make_analysis_dataset_and_statistics: source contracts" setup=[setupMakeAnalysisDatasetAndStatistics] begin
    flags = [
        "--kind",
        "--D",
        "--cut_restriction",
        "--link_probability",
        "--size",
        "--num_csets",
        "--seed",
        "--num_processes",
        "--dataset_multiprocessing",
        "--batchsize",
        "--outdir",
    ]
    for flag in flags
        @test occursin(flag, pipeline_src)
    end

    for path in [
        "dataset.jld2",
        "statistics.jld2",
        "config.yaml",
        "README.txt",
        "make_analysis_dataset.jl",
        "make_analysis_dataset_sequential.jl",
        "make_analysis_statistics.jl",
    ]
        @test occursin(path, pipeline_src)
    end

    @test occursin("YAML.write_file(config_out, config)", pipeline_src)
    @test occursin("cmd = `julia -O3 \$dataset_script_copy", pipeline_src)
    @test occursin("run(`julia -O3 \$stats_script_copy", pipeline_src)
    @test occursin("Pipeline completed successfully", pipeline_src)

    for cfg_key in [
        "\"kind\"",
        "\"size\"",
        "\"num_csets\"",
        "\"seed\"",
        "\"batchsize\"",
        "\"num_processes\"",
        "\"dataset_multiprocessing\"",
        "\"dataset_out\"",
        "\"stats_out\"",
        "\"datetime_utc\"",
    ]
        @test occursin(cfg_key, pipeline_src)
    end
end
