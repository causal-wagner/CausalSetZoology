@testsnippet setupMakeAnalysisDataset begin
    using Test
    include(joinpath(@__DIR__, "test_support.jl"))

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
