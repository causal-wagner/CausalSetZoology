@testsnippet setupGenerateDataset begin
    using Test
    using JLD2
    using Distributed
    import Random
    import Distributions
    import CausalSets
    import CausalSetZoology

    function _ensure_worker()
        if nworkers() < 1
            addprocs(1; exeflags = "--threads=1")
        end
    end

    # Verifies stored sparse links equal a fresh conversion from each generated causet.
    function _assert_links_match_generated_csets(data)
        @test length(data.links_b) == length(data.csets_b)
        for i in eachindex(data.csets_b)
            expected = CausalSetZoology.SparseLinksCauset(data.csets_b[i])
            @test data.links_b[i].atom_count == expected.atom_count
            @test data.links_b[i].future_links == expected.future_links
            @test data.links_b[i].past_links == expected.past_links
        end
    end
end

# Verifies Minkowski sprinkling branch produces csets/links without extra kind metadata arrays.
@testitem "generate_dataset: generate_batch minkowski_sprinkling" setup=[setupGenerateDataset] begin
    mink = CausalSets.MinkowskiManifold{2}()
    boundary = CausalSets.CausalDiamondBoundary{2}(1.0)

    data = CausalSetZoology.generate_batch(
        1,
        1,
        1,
        "minkowski_sprinkling",
        101;
        cset_size = 12,
        D = 2,
        mink = mink,
        causal_diamond_boundary = boundary,
    )

    @test length(data.csets_b) == 1
    @test length(data.links_b) == 1
    @test data.csets_b[1].atom_count == 12
    @test data.links_b[1].atom_count == 12
    _assert_links_match_generated_csets(data)
    @test isempty(data.r_b)
    @test isempty(data.order_b)
    @test isempty(data.trans_in_b)
    @test isempty(data.trans_out_b)
end

# Verifies manifoldlike simply-connected branch fills r/order metadata.
@testitem "generate_dataset: generate_batch manifoldlike_simply_connected" setup=[setupGenerateDataset] begin
    data = CausalSetZoology.generate_batch(
        1,
        1,
        1,
        "manifoldlike_simply_connected",
        102;
        cset_size = 16,
        D = 2,
        rdistr = Distributions.Uniform(2.0, 2.01),
    )

    @test length(data.csets_b) == 1
    @test length(data.links_b) == 1
    @test data.csets_b[1].atom_count == 16
    @test data.links_b[1].atom_count == 16
    _assert_links_match_generated_csets(data)
    @test length(data.r_b) == 1
    @test length(data.order_b) == 1
    @test 2.0 <= data.r_b[1] <= 2.01
    @test data.order_b[1] >= 2
    @test isempty(data.num_boundary_cuts_b)
    @test isempty(data.genus_b)
end

# Verifies non-simply-connected branch fills topology metadata and honors boundary-cuts mode.
@testitem "generate_dataset: generate_batch manifoldlike_non_simply_connected" setup=[setupGenerateDataset] begin
    data = CausalSetZoology.generate_batch(
        1,
        1,
        1,
        "manifoldlike_non_simply_connected",
        103;
        cset_size = 16,
        rdistr = Distributions.Uniform(2.0, 2.01),
        cut_restriction = "boundary_cuts",
        num_boundary_cuts_distr = Distributions.DiscreteUniform(1, 1),
        genus_distr = Distributions.DiscreteUniform(1, 1),
    )

    @test length(data.csets_b) == 1
    @test length(data.links_b) == 1
    @test data.csets_b[1].atom_count == 16
    @test data.links_b[1].atom_count == 16
    _assert_links_match_generated_csets(data)
    @test length(data.num_boundary_cuts_b) == 1
    @test length(data.genus_b) == 1
    @test length(data.r_b) == 1
    @test length(data.order_b) == 1
    @test data.num_boundary_cuts_b[1] == 1
    @test data.genus_b[1] == 0
end

# Verifies destroyed branch fills rel_num_flips metadata.
@testitem "generate_dataset: generate_batch destroyed" setup=[setupGenerateDataset] begin
    data = CausalSetZoology.generate_batch(
        1,
        1,
        1,
        "destroyed",
        104;
        cset_size = 16,
        rdistr = Distributions.Uniform(2.0, 2.01),
        non_manifoldlikeness_distr = Distributions.Uniform(0.05, 0.051),
    )

    @test length(data.csets_b) == 1
    @test length(data.links_b) == 1
    @test data.csets_b[1].atom_count == 16
    @test data.links_b[1].atom_count == 16
    _assert_links_match_generated_csets(data)
    @test length(data.rel_num_flips_b) == 1
    @test length(data.r_b) == 1
    @test length(data.order_b) == 1
    @test data.rel_num_flips_b[1] > 0.0
    @test data.rel_num_flips_b[1] < 1.0
end

# Verifies merged branch fills rel_size_KR and link_probability metadata.
@testitem "generate_dataset: generate_batch merged" setup=[setupGenerateDataset] begin
    data = CausalSetZoology.generate_batch(
        1,
        1,
        1,
        "merged",
        105;
        cset_size = 16,
        rdistr = Distributions.Uniform(2.0, 2.01),
        non_manifoldlikeness_distr = Distributions.Uniform(0.05, 0.051),
        link_probability = 0.4,
    )

    @test length(data.csets_b) == 1
    @test length(data.links_b) == 1
    @test data.csets_b[1].atom_count == 16
    @test data.links_b[1].atom_count == 16
    _assert_links_match_generated_csets(data)
    @test length(data.r_b) == 1
    @test length(data.order_b) == 1
    @test length(data.rel_size_KR_b) == 1
    @test length(data.link_probability_b) == 1
    @test data.rel_size_KR_b[1] > 0.0
    @test data.link_probability_b[1] ≈ 0.4 atol = 1e-12
end

# Verifies generate_batch grid path returns typed batches with per-sample metadata.
@testitem "generate_dataset: generate_batch grid basic" setup=[setupGenerateDataset] begin
    data = CausalSetZoology.generate_batch(
        1,
        2,
        2,
        "grid",
        123;
        cset_size = 12,
        rdistr = Distributions.Uniform(2.0, 2.001),
        lattice_distr = Distributions.DiscreteUniform(1, 1),
        lattices = ["quadratic"],
        segment_ratio_distr = Distributions.Uniform(1.0, 1.001),
        rotate_angle_distr = Distributions.Uniform(0.0, 0.001),
        oblique_angle_distr = Distributions.Uniform(10.0, 10.001),
    )

    @test length(data.csets_b) == 2
    @test length(data.links_b) == 2
    @test all(c -> c.atom_count == 12, data.csets_b)
    @test all(l -> l.atom_count == 12, data.links_b)
    @test all(l -> l isa CausalSetZoology.SparseLinksCauset, data.links_b)
    _assert_links_match_generated_csets(data)
    @test length(data.r_b) == 2
    @test length(data.order_b) == 2
    @test length(data.lattice_b) == 2
    @test length(data.segment_ratio_b) == 2
    @test length(data.segment_angle_b) == 2
    @test length(data.rotation_angle_b) == 2
    @test data.lattice_b == ["quadratic", "quadratic"]
    @test all(x -> 1.0 <= x <= 1.001, data.segment_ratio_b)
    @test all(x -> 0.0 <= x <= 0.001, data.rotation_angle_b)
    @test all(x -> 10.0 <= x <= 10.001, data.segment_angle_b)
end

# Verifies random branch returns a generated cset and link graph for sampled connectivity target.
@testitem "generate_dataset: generate_batch random" setup=[setupGenerateDataset] begin
    data = CausalSetZoology.generate_batch(
        1,
        1,
        1,
        "random",
        106;
        cset_size = 256,
        connectivity_distr = Distributions.Normal(0.5, 0.01),
    )

    @test length(data.csets_b) == 1
    @test length(data.links_b) == 1
    @test data.csets_b[1].atom_count == 256
    _assert_links_match_generated_csets(data)
    @test isempty(data.r_b)
    @test isempty(data.order_b)
    @test isempty(data.link_probability_b)
end

# Verifies variable-size branch consumes ndistr and produces the sampled size.
@testitem "generate_dataset: generate_batch variable-size path" setup=[setupGenerateDataset] begin
    data = CausalSetZoology.generate_batch(
        1,
        2,
        2,
        "layered",
        321;
        cset_size = nothing,
        ndistr = Distributions.DiscreteUniform(8, 8),
        layers_distr = Distributions.DiscreteUniform(2, 2),
        link_probability_distr = Distributions.Uniform(0.5, 0.5001),
    )

    @test length(data.csets_b) == 2
    @test all(c -> c.atom_count == 8, data.csets_b)
    @test length(data.links_b) == 2
    _assert_links_match_generated_csets(data)
    @test length(data.num_layers_b) == 2
    @test length(data.std_b) == 2
    @test all(==(2), data.num_layers_b)
    @test all(x -> x >= 0.0, data.std_b)
end

# Verifies quasicrystal branch guard when the required crystal input is missing.
@testitem "generate_dataset: generate_batch minkowski_quasicrystal validation" setup=[setupGenerateDataset] begin
    @test_throws ArgumentError CausalSetZoology.generate_batch(
        1,
        1,
        1,
        "minkowski_quasicrystal",
        1;
        cset_size = 8,
        big_crystal = nothing,
    )
end

# Verifies end-to-end writer creates expected metadata and batch payloads.
@testitem "generate_dataset: create_dataset_and_save basic" setup=[setupGenerateDataset] begin
    _ensure_worker()

    out_path = joinpath(mktempdir(), "dataset.jld2")
    config = Dict("kind" => "grid", "num_csets" => 2, "cset_size" => 16)

    CausalSetZoology.create_dataset_and_save(
        out_path,
        "grid",
        1,
        1,
        2,
        2,
        config,
        11;
        cset_size = 16,
        rdistr = Distributions.Uniform(2.0, 2.01),
        lattice_distr = Distributions.DiscreteUniform(1, 1),
        lattices = ["quadratic"],
        segment_ratio_distr = Distributions.Uniform(1.0, 1.01),
        rotate_angle_distr = Distributions.Uniform(0.0, 0.01),
        oblique_angle_distr = Distributions.Uniform(10.0, 10.01),
    )

    @test isfile(out_path)

    JLD2.jldopen(out_path, "r") do f
        @test f["meta/batchsize"] == 1
        @test f["meta/nbatches"] == 2
        @test f["meta/N"] == 2
        @test f["meta/config"]["kind"] == "grid"

        b1 = f["batches/1"]
        @test haskey(b1, "csets")
        @test haskey(b1, "links")
        @test length(b1["csets"]) == 1
        @test length(b1["links"]) == 1
        @test b1["links"][1] isa CausalSetZoology.SparseLinksCauset
    end
end

# Verifies worker-count validation catches invalid values and unavailable workers.
@testitem "generate_dataset: create_dataset_and_save worker validation" setup=[setupGenerateDataset] begin
    out_path = joinpath(mktempdir(), "dataset.jld2")
    config = Dict("kind" => "grid", "num_csets" => 1, "cset_size" => 16)

    @test_throws ErrorException CausalSetZoology.create_dataset_and_save(
        out_path,
        "grid",
        0,
        1,
        1,
        1,
        config,
        11;
        cset_size = 16,
        rdistr = Distributions.Uniform(2.0, 2.01),
        lattice_distr = Distributions.DiscreteUniform(1, 1),
        lattices = ["quadratic"],
        segment_ratio_distr = Distributions.Uniform(1.0, 1.01),
        rotate_angle_distr = Distributions.Uniform(0.0, 0.01),
        oblique_angle_distr = Distributions.Uniform(10.0, 10.01),
    )

    # Request more workers than currently available should fail fast.
    req_workers = max(2, nworkers() + 1)
    @test_throws ErrorException CausalSetZoology.create_dataset_and_save(
        out_path,
        "grid",
        req_workers,
        1,
        1,
        1,
        config,
        11;
        cset_size = 16,
        rdistr = Distributions.Uniform(2.0, 2.01),
        lattice_distr = Distributions.DiscreteUniform(1, 1),
        lattices = ["quadratic"],
        segment_ratio_distr = Distributions.Uniform(1.0, 1.01),
        rotate_angle_distr = Distributions.Uniform(0.0, 0.01),
        oblique_angle_distr = Distributions.Uniform(10.0, 10.01),
    )
end
