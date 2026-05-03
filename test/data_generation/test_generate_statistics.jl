@testsnippet setupGenerateStatistics begin
    using Test
    using JLD2
    using Distributed
    using Statistics
    import Distributions
    import CausalSets
    import CausalSetZoology

    function _ensure_worker()
        if nworkers() < 1
            addprocs(1; exeflags = "--threads=1")
        end
    end

    # Transitively closed 4-chain.
    function _chain4_bitarray()
        future = [
            BitVector([0, 1, 1, 1]),
            BitVector([0, 0, 1, 1]),
            BitVector([0, 0, 0, 1]),
            BitVector([0, 0, 0, 0]),
        ]
        past = [
            BitVector([0, 0, 0, 0]),
            BitVector([1, 0, 0, 0]),
            BitVector([1, 1, 0, 0]),
            BitVector([1, 1, 1, 0]),
        ]
        return CausalSets.BitArrayCauset(4, future, past)
    end

    function _chain4_sparse_links()
        return CausalSetZoology.SparseLinksCauset(
            Int64(4),
            [Int32[2], Int32[3], Int32[4], Int32[]],
            [Int32[], Int32[1], Int32[2], Int32[3]],
        )
    end

    function _write_tiny_dataset(path::String; kind::String = "grid")
        # Build fixture data through the same generator path as datasets to avoid
        # synthetic-edge-case mismatches in downstream CausalSets measurements.
        batch = CausalSetZoology.generate_batch(
            1,
            1,
            1,
            kind,
            17;
            cset_size = 16,
            ndistr = Distributions.DiscreteUniform(16, 16),
            rdistr = Distributions.Uniform(2.0, 2.01),
            genus_distr = Distributions.DiscreteUniform(1, 1),
            num_boundary_cuts_distr = Distributions.DiscreteUniform(1, 1),
            lattice_distr = Distributions.DiscreteUniform(1, 1),
            lattices = ["quadratic"],
            segment_ratio_distr = Distributions.Uniform(1.0, 1.01),
            rotate_angle_distr = Distributions.Uniform(0.0, 0.01),
            oblique_angle_distr = Distributions.Uniform(10.0, 10.01),
            non_manifoldlikeness_distr = Distributions.Uniform(0.05, 0.051),
            layers_distr = Distributions.DiscreteUniform(2, 2),
            link_probability_distr = Distributions.Uniform(0.5, 0.51),
            connectivity_distr = Distributions.Normal(0.5, 0.01),
        )
        cset = batch.csets_b[1]
        links = batch.links_b[1]
        adj = transpose(reduce(hcat, cset.future_relations))

        JLD2.jldopen(path, "w") do f
            f["meta/batchsize"] = 1
            f["meta/nbatches"] = 1
            f["meta/N"] = 1
            f["meta/config"] = Dict("kind" => kind)
            f["batches/1/csets"] = [cset]
            f["batches/1/adjs"] = [adj]
            f["batches/1/links"] = [links]

            if kind == "layered"
                f["batches/1/num_layers"] = [batch.num_layers_b[1]]
                f["batches/1/std"] = [batch.std_b[1]]
            elseif kind == "grid"
                f["batches/1/segment_ratio"] = [batch.segment_ratio_b[1]]
                f["batches/1/segment_angle"] = [batch.segment_angle_b[1]]
                f["batches/1/rotation_angle"] = [batch.rotation_angle_b[1]]
                f["batches/1/lattice"] = [batch.lattice_b[1]]
            elseif kind == "merged"
                f["batches/1/r"] = [batch.r_b[1]]
                f["batches/1/order"] = [batch.order_b[1]]
                f["batches/1/rel_size_KR"] = [batch.rel_size_KR_b[1]]
                f["batches/1/link_probability"] = [batch.link_probability_b[1]]
            end
        end
    end

    function _generated_fixture(; kind::String = "random")
        batch = CausalSetZoology.generate_batch(
            1,
            1,
            1,
            kind,
            23;
            cset_size = 16,
            ndistr = Distributions.DiscreteUniform(16, 16),
            rdistr = Distributions.Uniform(2.0, 2.01),
            genus_distr = Distributions.DiscreteUniform(1, 1),
            num_boundary_cuts_distr = Distributions.DiscreteUniform(1, 1),
            lattice_distr = Distributions.DiscreteUniform(1, 1),
            lattices = ["quadratic"],
            segment_ratio_distr = Distributions.Uniform(1.0, 1.01),
            rotate_angle_distr = Distributions.Uniform(0.0, 0.01),
            oblique_angle_distr = Distributions.Uniform(10.0, 10.01),
            non_manifoldlikeness_distr = Distributions.Uniform(0.05, 0.051),
            layers_distr = Distributions.DiscreteUniform(2, 2),
            link_probability_distr = Distributions.Uniform(0.5, 0.51),
            connectivity_distr = Distributions.Normal(0.5, 0.01),
        )
        return batch.csets_b[1], batch.links_b[1]
    end
end

# Verifies base statistics fields are present and numerically plausible for random-kind path.
@testitem "generate_statistics: compute_statistics basic" setup=[setupGenerateStatistics] begin
    cset, links = _generated_fixture(kind = "grid")
    rec = CausalSetZoology.compute_statistics(cset, links; kind = "grid")

    @test hasfield(typeof(rec), :n)
    @test hasfield(typeof(rec), :connectivity)
    @test hasfield(typeof(rec), :in_degree_hist)
    @test hasfield(typeof(rec), :out_degree_hist)
    @test hasfield(typeof(rec), :degree_hist)
    @test hasfield(typeof(rec), :in_degree_hist_link)
    @test hasfield(typeof(rec), :out_degree_hist_link)
    @test hasfield(typeof(rec), :degree_hist_link)
    @test hasfield(typeof(rec), :max_pathlen_hist)
    @test hasfield(typeof(rec), :max_pathlen_sources_hist)
    @test hasfield(typeof(rec), :ev_sym_link)
    @test hasfield(typeof(rec), :ev_imag_antisym_in_link)
    @test hasfield(typeof(rec), :ev_imag_antisym_in_mean_link)
    @test hasfield(typeof(rec), :communicability_link)
    @test hasfield(typeof(rec), :communicability_mean_link)

    @test rec.n == cset.atom_count
    @test rec.connectivity > 0
    @test rec.num_sources >= 1
    @test rec.num_sinks >= 1
    @test sum(values(rec.degree_hist)) == rec.n
    @test sum(values(rec.degree_hist_link)) == rec.n
    @test sum(values(rec.max_pathlen_hist)) == rec.n
    @test sum(values(rec.max_pathlen_sources_hist)) == rec.num_sources
    @test length(rec.ev_imag_antisym_in_link) == rec.n
    @test length(rec.communicability_link) == rec.n
end

# Verifies kind-specific augmentation fields are attached for representative kinds.
@testitem "generate_statistics: compute_statistics kind augmentation" setup=[setupGenerateStatistics] begin
    cset, links = _generated_fixture(kind = "layered")

    rec_layered = CausalSetZoology.compute_statistics(cset, links; kind = "layered", num_layers = 4, std = 0.2)
    @test hasfield(typeof(rec_layered), :num_layers)
    @test hasfield(typeof(rec_layered), :standard_dev)
    @test rec_layered.num_layers == 4
    @test rec_layered.standard_dev == 0.2

    rec_merged = CausalSetZoology.compute_statistics(cset, links; kind = "merged", r = 2.0, order = 5, rel_size_KR = 0.15, link_probability = 0.4)
    @test hasfield(typeof(rec_merged), :r)
    @test hasfield(typeof(rec_merged), :order)
    @test hasfield(typeof(rec_merged), :rel_size_KR)
    @test hasfield(typeof(rec_merged), :link_probability)
end

# Verifies observable filtering computes and returns only the requested statistic groups.
@testitem "generate_statistics: compute_statistics observables filter" setup=[setupGenerateStatistics] begin
    cset, links = _generated_fixture(kind = "grid")

    rec = CausalSetZoology.compute_statistics(
        cset,
        links;
        kind = "grid",
        observables = [:degree, :ev_sym, :communicability],
    )

    @test hasfield(typeof(rec), :n)
    @test hasfield(typeof(rec), :in_degree_hist)
    @test hasfield(typeof(rec), :degree_hist)
    @test hasfield(typeof(rec), :ev_sym_link)
    @test hasfield(typeof(rec), :communicability_link)

    @test !hasfield(typeof(rec), :in_degree_hist_link)
    @test !hasfield(typeof(rec), :max_pathlen_hist)
    @test !hasfield(typeof(rec), :max_pathlen_sources_hist)
    @test !hasfield(typeof(rec), :cardinalities_hist)
    @test !hasfield(typeof(rec), :ev_imag_antisym_in_link)

    @test hasfield(typeof(rec), :segment_ratio)
    @test hasfield(typeof(rec), :segment_angle)
    @test hasfield(typeof(rec), :rotation_angle)
    @test hasfield(typeof(rec), :lattice)
end

# Verifies each supported observable group can be requested individually.
@testitem "generate_statistics: compute_statistics observables one-by-one" setup=[setupGenerateStatistics] begin
    cset, links = _generated_fixture(kind = "grid")
    expected = Dict(
        :degree => (:degree_hist, [:in_degree_hist_link, :ev_sym_link, :cardinalities_hist]),
        :link_degree => (:degree_hist_link, [:degree_hist, :ev_sym_link, :cardinalities_hist]),
        :ev_sym => (:ev_sym_link, [:degree_hist, :degree_hist_link, :ev_imag_antisym_in_link]),
        :ev_antisym => (:ev_imag_antisym_in_link, [:degree_hist, :degree_hist_link, :ev_sym_link]),
        :cardinalities => (:cardinalities_hist, [:degree_hist, :degree_hist_link, :ev_sym_link]),
        :max_pathlen => (:max_pathlen_hist, [:degree_hist, :degree_hist_link, :cardinalities_hist, :max_pathlen_sources_hist]),
        :max_pathlen_sources => (:max_pathlen_sources_hist, [:degree_hist, :degree_hist_link, :cardinalities_hist, :max_pathlen_hist]),
        :communicability => (:communicability_link, [:degree_hist, :degree_hist_link, :cardinalities_hist]),
    )

    for (obs, (present_field, absent_fields)) in expected
        rec = CausalSetZoology.compute_statistics(cset, links; kind = "grid", observables = [obs])
        @test hasfield(typeof(rec), :n)
        @test hasfield(typeof(rec), :connectivity)
        @test hasfield(typeof(rec), present_field)
        for fld in absent_fields
            @test !hasfield(typeof(rec), fld)
        end
    end
end

# Verifies compute_statistics validation guards for kind/domain/shape.
@testitem "generate_statistics: compute_statistics validation" setup=[setupGenerateStatistics] begin
    cset, links = _generated_fixture(kind = "grid")

    @test_throws ArgumentError CausalSetZoology.compute_statistics(cset, links; kind = "bad_kind")
    @test_throws ArgumentError CausalSetZoology.compute_statistics(
        cset,
        links;
        kind = "grid",
        observables = [:degree, :not_real],
    )

    links_bad = CausalSetZoology.SparseLinksCauset(
        Int64(2),
        [Int32[2], Int32[]],
        [Int32[], Int32[1]],
    )
    @test_throws DimensionMismatch CausalSetZoology.compute_statistics(cset, links_bad; kind = "grid")
end

# Verifies links-only overload computes only link-supported observable groups.
@testitem "generate_statistics: compute_statistics links-only" setup=[setupGenerateStatistics] begin
    links = _chain4_sparse_links()

    rec = CausalSetZoology.compute_statistics(
        links;
        kind = "grid",
        observables = [:link_degree, :max_pathlen, :max_pathlen_sources, :ev_sym, :ev_antisym, :communicability],
    )

    @test hasfield(typeof(rec), :n)
    @test hasfield(typeof(rec), :degree_hist_link)
    @test hasfield(typeof(rec), :max_pathlen_hist)
    @test hasfield(typeof(rec), :max_pathlen_sources_hist)
    @test hasfield(typeof(rec), :ev_sym_link)
    @test hasfield(typeof(rec), :ev_imag_antisym_in_link)
    @test hasfield(typeof(rec), :communicability_link)

    @test !hasfield(typeof(rec), :degree_hist)
    @test !hasfield(typeof(rec), :cardinalities_hist)
    @test !hasfield(typeof(rec), :connectivity)

    @test rec.n == 4
    @test rec.num_sources == 1
    @test rec.num_sinks == 1
    @test sum(values(rec.degree_hist_link)) == rec.n
    @test rec.max_pathlen_hist == Dict(3 => 1, 2 => 1, 1 => 1, 0 => 1)
    @test rec.max_pathlen_sources_hist == Dict(3 => 1)
end

# Verifies links-only overload defaults to all supported observables and matches common fields from the two-argument method.
@testitem "generate_statistics: compute_statistics links-only equivalence" setup=[setupGenerateStatistics] begin
    cset, links = _generated_fixture(kind = "grid")
    obs = [:link_degree, :max_pathlen, :max_pathlen_sources, :ev_sym, :ev_antisym, :communicability]

    rec_full = CausalSetZoology.compute_statistics(cset, links; kind = "grid", observables = obs)
    rec_links = CausalSetZoology.compute_statistics(links; kind = "grid")

    common_fields = (
        :n,
        :degree_hist_link,
        :degree_mean_link,
        :num_sources,
        :num_sinks,
        :num_sources_link,
        :num_sinks_link,
        :max_pathlen_hist,
        :max_pathlen_mean,
        :max_pathlen_sources_hist,
        :max_pathlen_sources_mean,
        :ev_sym_link,
        :ev_sym_mean_link,
        :ev_imag_antisym_in_link,
        :ev_imag_antisym_in_mean_link,
        :communicability_link,
        :communicability_mean_link,
        :segment_ratio,
        :segment_angle,
        :rotation_angle,
        :lattice,
    )
    for fld in common_fields
        @test getproperty(rec_links, fld) == getproperty(rec_full, fld)
    end
end

# Verifies links-only overload rejects closure-dependent observables.
@testitem "generate_statistics: compute_statistics links-only validation" setup=[setupGenerateStatistics] begin
    links = _chain4_sparse_links()

    @test_throws ArgumentError CausalSetZoology.compute_statistics(
        links;
        kind = "grid",
        observables = [:degree],
    )
    @test_throws ArgumentError CausalSetZoology.compute_statistics(
        links;
        kind = "grid",
        observables = [:cardinalities],
    )
    @test_throws ArgumentError CausalSetZoology.compute_statistics(
        links;
        kind = "bad_kind",
    )
    empty_links = CausalSetZoology.SparseLinksCauset(Int64(0), Vector{Vector{Int32}}(), Vector{Vector{Int32}}())
    @test_throws DomainError CausalSetZoology.compute_statistics(
        empty_links;
        kind = "grid",
    )
end

# Verifies writer computes and stores one full batch of statistics from a tiny dataset.
@testitem "generate_statistics: create_statistics_dataset_and_save basic" setup=[setupGenerateStatistics] begin
    tmp = mktempdir()
    in_path = joinpath(tmp, "dataset.jld2")
    out_path = joinpath(tmp, "stats.jld2")

    _write_tiny_dataset(in_path; kind = "grid")
    CausalSetZoology.create_statistics_dataset_and_save(in_path, out_path, "grid", 1, 1, 1)

    @test isfile(out_path)

    JLD2.jldopen(out_path, "r") do f
        @test f["meta/batchsize"] == 1
        @test f["meta/nbatches"] == 1
        @test f["meta/N"] == 1
        recs = f["batches/1"]
        @test length(recs) == 1
        rec = recs[1]
        @test hasfield(typeof(rec), :n)
        @test hasfield(typeof(rec), :connectivity)
    end
end

# Verifies kind-dependent metadata loading path is wired for layered records.
@testitem "generate_statistics: create_statistics_dataset_and_save layered metadata" setup=[setupGenerateStatistics] begin
    tmp = mktempdir()
    in_path = joinpath(tmp, "dataset_layered.jld2")
    out_path = joinpath(tmp, "stats_layered.jld2")

    _write_tiny_dataset(in_path; kind = "layered")
    CausalSetZoology.create_statistics_dataset_and_save(in_path, out_path, "layered", 1, 1, 1)

    @test isfile(out_path)
    JLD2.jldopen(out_path, "r") do f
        rec = f["batches/1"][1]
        @test hasfield(typeof(rec), :num_layers)
        @test hasfield(typeof(rec), :standard_dev)
    end
end

# Verifies statistics writer can consume links-only datasets when only link observables are requested.
@testitem "generate_statistics: create_statistics_dataset_and_save links-only dataset" setup=[setupGenerateStatistics] begin
    tmp = mktempdir()
    in_path = joinpath(tmp, "dataset_links_only.jld2")
    out_path = joinpath(tmp, "stats_links_only.jld2")

    config = Dict("kind" => "grid", "num_csets" => 1, "cset_size" => 16, "links_only" => true)
    CausalSetZoology.create_dataset_and_save(
        in_path,
        "grid",
        1,
        1,
        1,
        1,
        config,
        42;
        cset_size = 16,
        lattice_distr = Distributions.DiscreteUniform(1, 1),
        lattices = ["quadratic"],
        segment_ratio_distr = Distributions.Uniform(1.0, 1.01),
        rotate_angle_distr = Distributions.Uniform(0.0, 0.01),
        oblique_angle_distr = Distributions.Uniform(10.0, 10.01),
        links_only = true,
    )

    CausalSetZoology.create_statistics_dataset_and_save(
        in_path,
        out_path,
        "grid",
        1,
        1,
        1;
        observables = [:link_degree],
    )

    @test isfile(out_path)
    JLD2.jldopen(out_path, "r") do f
        rec = f["batches/1"][1]
        @test hasfield(typeof(rec), :degree_hist_link)
        @test !hasfield(typeof(rec), :degree_hist)
        @test !hasfield(typeof(rec), :connectivity)
        @test hasfield(typeof(rec), :segment_ratio)
    end
end

# Verifies writer argument and kind validation.
@testitem "generate_statistics: create_statistics_dataset_and_save validation" setup=[setupGenerateStatistics] begin
    tmp = mktempdir()
    in_path = joinpath(tmp, "dataset_bad.jld2")
    out_path = joinpath(tmp, "stats_bad.jld2")
    _write_tiny_dataset(in_path; kind = "grid")

    @test_throws DomainError CausalSetZoology.create_statistics_dataset_and_save(in_path, out_path, "grid", 0, 1, 1)
    @test_throws DomainError CausalSetZoology.create_statistics_dataset_and_save(in_path, out_path, "grid", 1, 0, 1)
    @test_throws DomainError CausalSetZoology.create_statistics_dataset_and_save(in_path, out_path, "grid", 1, 1, 0)
    @test_throws ArgumentError CausalSetZoology.create_statistics_dataset_and_save(in_path, out_path, "bad_kind", 1, 1, 1)
end

# Verifies full module pipeline: dataset generation then statistics generation.
@testitem "data_generation pipeline: dataset to statistics e2e" setup=[setupGenerateStatistics] begin
    _ensure_worker()

    specs = [
        (
            kind = "grid",
            kwargs = (
                cset_size = 16,
                rdistr = Distributions.Uniform(2.0, 2.01),
                lattice_distr = Distributions.DiscreteUniform(1, 1),
                lattices = ["quadratic"],
                segment_ratio_distr = Distributions.Uniform(1.0, 1.01),
                rotate_angle_distr = Distributions.Uniform(0.0, 0.01),
                oblique_angle_distr = Distributions.Uniform(10.0, 10.01),
            ),
            expected_stats_fields = [:segment_ratio, :segment_angle, :rotation_angle, :lattice],
        ),
        (
            kind = "layered",
            kwargs = (
                cset_size = 16,
                layers_distr = Distributions.DiscreteUniform(2, 2),
                link_probability_distr = Distributions.Uniform(0.5, 0.51),
            ),
            expected_stats_fields = [:num_layers, :standard_dev],
        ),
    ]

    for spec in specs
        tmp = mktempdir()
        dataset_path = joinpath(tmp, "dataset_$(spec.kind).jld2")
        stats_path = joinpath(tmp, "stats_$(spec.kind).jld2")

        N = 2
        batchsize = 1
        nbatches = 2
        config = Dict("kind" => spec.kind, "num_csets" => N, "cset_size" => 16)

        CausalSetZoology.create_dataset_and_save(
            dataset_path,
            spec.kind,
            1,
            batchsize,
            nbatches,
            N,
            config,
            99;
            spec.kwargs...,
        )
        @test isfile(dataset_path)

        CausalSetZoology.create_statistics_dataset_and_save(
            dataset_path,
            stats_path,
            spec.kind,
            batchsize,
            nbatches,
            N,
        )
        @test isfile(stats_path)

        JLD2.jldopen(stats_path, "r") do f
            @test f["meta/N"] == N
            @test f["meta/nbatches"] == nbatches
            @test f["meta/batchsize"] == batchsize

            rec = f["batches/1"][1]
            @test hasfield(typeof(rec), :n)
            @test hasfield(typeof(rec), :connectivity)
            for fld in spec.expected_stats_fields
                @test hasfield(typeof(rec), fld)
            end
        end
    end
end
