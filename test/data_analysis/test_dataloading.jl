@testsnippet setupDataloading begin
    using Test
    using Statistics
    using JLD2

    Base.@kwdef struct _DLRec
        in_degree_hist_link::Dict{Int,Float64}
        out_degree_hist_link::Dict{Int,Float64}
        max_pathlen_hist::Dict{Int,Float64}
        ev_sym_link::Vector{Float64}
        score::Float64
        scalar::Float64
    end

    Base.@kwdef struct _DLRecBadScalar
        in_degree_hist_link::Dict{Int,Float64}
        out_degree_hist_link::Dict{Int,Float64}
        max_pathlen_hist::Dict{Int,Float64}
        ev_sym_link::Vector{Float64}
        score::Float64
        scalar::String
    end

    function _write_dl_file(path::String, batches)
        JLD2.jldopen(path, "w") do f
            f["meta/nbatches"] = length(batches)
            for (i, b) in enumerate(batches)
                f["batches/$i"] = b
            end
        end
    end

    function _make_dl_fixture()
        dir = mktempdir()
        p1 = joinpath(dir, "a.jld2")
        p2 = joinpath(dir, "b.jld2")

        b1 = [[
            _DLRec(Dict(1 => 1.0), Dict(1 => 2.0), Dict(1 => 3.0), [1.0, 2.0], 1.0, 10.0),
            _DLRec(Dict(1 => 2.0), Dict(1 => 1.0), Dict(1 => 1.0), [2.0, 3.0], 2.0, 20.0),
        ]]
        b2 = [[
            _DLRec(Dict(1 => 3.0), Dict(1 => 1.0), Dict(1 => 2.0), [4.0, 5.0], 3.0, 10.0),
            _DLRec(Dict(1 => 4.0), Dict(1 => 2.0), Dict(1 => 2.0), [6.0, 7.0], 4.0, 20.0),
        ]]

        _write_dl_file(p1, b1)
        _write_dl_file(p2, b2)
        return p1, p2
    end

    function _make_dl_bad_scalar_fixture()
        dir = mktempdir()
        p = joinpath(dir, "bad_scalar.jld2")
        b = [[
            _DLRecBadScalar(Dict(1 => 1.0), Dict(1 => 2.0), Dict(1 => 3.0), [1.0, 2.0], 1.0, "bad"),
            _DLRecBadScalar(Dict(1 => 2.0), Dict(1 => 1.0), Dict(1 => 1.0), [2.0, 3.0], 2.0, "bad"),
        ]]
        _write_dl_file(p, b)
        return p
    end

    function _contains_typeerror(e)
        e isa CompositeException || return false
        return any(ex -> ex isa TypeError || (ex isa TaskFailedException && ex.task.exception isa TypeError), e.exceptions)
    end
end

@testitem "dataloading helpers: running moments" setup=[setupDataloading] begin
    # Test intent: validate dataloading helpers: running moments behavior and output contract.
    m = CausalSetZoology._RunningMoments()
    μ0, σ0 = CausalSetZoology._mean_std(m)
    @test isnan(μ0)
    @test isnan(σ0)

    CausalSetZoology._push_moment!(m, 1.0)
    @test CausalSetZoology._mean_std(m) == (1.0, 0.0)

    CausalSetZoology._push_moment!(m, 2.0)
    μ, σ = CausalSetZoology._mean_std(m)
    @test μ ≈ 1.5
    @test σ ≈ sqrt(0.5)
end

@testitem "dataloading helpers: scan config and keep_sample" setup=[setupDataloading] begin
    # Test intent: validate dataloading helpers: scan config and keep_sample behavior and output contract.
    c_int = CausalSetZoology._scan_config(2)
    @test c_int.step == 2
    @test c_int.offset == 0
    @test CausalSetZoology._keep_sample(1, c_int) == false
    @test CausalSetZoology._keep_sample(2, c_int) == true
    @test CausalSetZoology._keep_sample(3, c_int) == false
    @test CausalSetZoology._keep_sample(4, c_int) == true

    c_float = CausalSetZoology._scan_config(0.5)
    @test c_float.step == 2
    @test c_float.offset == 1
    @test CausalSetZoology._keep_sample(1, c_float) == true
    @test CausalSetZoology._keep_sample(2, c_float) == false
    @test CausalSetZoology._keep_sample(3, c_float) == true

    @test_throws DomainError CausalSetZoology._scan_config(0)
    @test_throws DomainError CausalSetZoology._scan_config(0.0)
    @test_throws DomainError CausalSetZoology._scan_config(1.5)
end

@testitem "dataloading helpers: filter normalization" setup=[setupDataloading] begin
    # Test intent: validate dataloading helpers: filter normalization behavior and output contract.
    paths = ["a", "b"]
    fs = CausalSetZoology._normalize_filters(paths, nothing)
    @test fs == [nothing, nothing]

    good_filters = Union{Nothing,Function}[nothing, x -> true]
    @test CausalSetZoology._normalize_filters(paths, good_filters) == good_filters
    @test_throws DimensionMismatch CausalSetZoology._normalize_filters(paths, [nothing])
end

@testitem "dataloading helpers: scan_records" setup=[setupDataloading] begin
    # Test intent: validate dataloading helpers: scan_records behavior and output contract.
    p1, _ = _make_dl_fixture()

    seen_scores = Float64[]
    CausalSetZoology._scan_records(x -> push!(seen_scores, x.score), p1, nothing, CausalSetZoology._scan_config(1))
    @test seen_scores == [1.0, 2.0]

    seen_filtered = Float64[]
    CausalSetZoology._scan_records(x -> push!(seen_filtered, x.score), p1, x -> x.score > 1.5, CausalSetZoology._scan_config(1))
    @test seen_filtered == [2.0]
end

@testitem "dataloading helpers: extraction and column helpers" setup=[setupDataloading] begin
    # Test intent: validate dataloading helpers: extraction and column helpers behavior and output contract.
    x = _DLRec(Dict(1 => 1.0), Dict(1 => 2.0), Dict(1 => 3.0), [1.0, 2.0], 1.0, 10.0)
    @test CausalSetZoology._extract_field_value(x, :score) == 1.0
    @test CausalSetZoology._extract_field_value(x, (:in_degree_hist_link, 1)) == 1.0
    @test CausalSetZoology._extract_field_value(x, (:in_degree_hist_link, 99)) == 0

    vals = Any[nothing, nothing]
    CausalSetZoology._push_auto_typed!(vals, 1, 1.0)
    CausalSetZoology._push_auto_typed!(vals, 1, 2.0)
    CausalSetZoology._push_auto_typed!(vals, 2, (1.0, 10.0))
    out = CausalSetZoology._finalize_any_columns(vals, 2)
    @test out[1] == [1.0, 2.0]
    @test out[2] == [(1.0, 10.0)]
end

@testitem "dataloading: load_and_average_std_scalar grouped" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_and_average_std_scalar grouped behavior and output contract.
    p1, _ = _make_dl_fixture()

    avs = CausalSetZoology.load_and_average_std_scalar([p1], [:score], :scalar)
    @test [s for (s, _) in avs[1]] == [10.0, 20.0]
    @test [stats[1][1] for (_, stats) in avs[1]] == [1.0, 2.0]
    @test [stats[1][2] for (_, stats) in avs[1]] == [0.0, 0.0]

    avs_binned = CausalSetZoology.load_and_average_std_scalar([p1], [:score], :scalar; num_bins = 1)
    @test length(avs_binned[1]) == 1
    @test avs_binned[1][1][2][1] == (1.5, sqrt(0.5))
end

@testitem "dataloading: load_and_average_std_scalar grouped validation" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_and_average_std_scalar grouped validation behavior and output contract.
    p1, _ = _make_dl_fixture()

    @test_throws DomainError CausalSetZoology.load_and_average_std_scalar([p1], [:score], :scalar; num_bins = 0)
    @test_throws ArgumentError CausalSetZoology.load_and_average_std_scalar([p1], Symbol[], :scalar)
end

@testitem "dataloading: load_histograms_from_paths plain" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_histograms_from_paths plain behavior and output contract.
    p1, p2 = _make_dl_fixture()

    h = CausalSetZoology.load_histograms_from_paths([p1, p2], :in_degree_hist_link)
    @test h == [
        [Dict(1 => 1.0), Dict(1 => 2.0)],
        [Dict(1 => 3.0), Dict(1 => 4.0)],
    ]
end

@testitem "dataloading: load_histograms_from_paths scalar" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_histograms_from_paths scalar behavior and output contract.
    p1, _ = _make_dl_fixture()

    hs = CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link, :scalar)
    @test hs == [[
        (Dict(1 => 1.0), 10.0),
        (Dict(1 => 2.0), 20.0),
    ]]
    @test [s for (_, s) in hs[1]] == [10.0, 20.0]
end

@testitem "dataloading: load_histograms filtering and thinning" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_histograms filtering and thinning behavior and output contract.
    p1, _ = _make_dl_fixture()

    filt = Union{Nothing,Function}[x -> x.score > 1.5]
    hf = CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; filters = filt, thinning = 1)
    @test hf == [[Dict(1 => 2.0)]]

    ht = CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; thinning = 2)
    @test ht == [[Dict(1 => 2.0)]]

    hfs = CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link, :scalar; filters = filt, thinning = 1)
    @test hfs == [[(Dict(1 => 2.0), 20.0)]]
end

@testitem "dataloading: load_histograms validation" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_histograms validation behavior and output contract.
    p1, _ = _make_dl_fixture()
    pbad = _make_dl_bad_scalar_fixture()

    @test_throws TypeError CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; filters = [nothing, nothing])
    @test_throws DimensionMismatch CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; filters = Union{Nothing,Function}[nothing, nothing])
    @test_throws DomainError CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; thinning = 0)

    err = try
        CausalSetZoology.load_histograms_from_paths([pbad], :in_degree_hist_link, :scalar)
        nothing
    catch e
        e
    end
    @test _contains_typeerror(err)
end

@testitem "dataloading helpers: field extractor planning" setup=[setupDataloading] begin
    # Test intent: validate dataloading helpers: field extractor planning behavior and output contract.
    fields = Union{Symbol,Tuple{Symbol,Int64}}[:score, (:in_degree_hist_link, 1), :scalar]
    symbol_specs, hist_specs = CausalSetZoology._split_field_specs(fields)
    @test symbol_specs == [(1, :score), (3, :scalar)]
    @test hist_specs == [(2, :in_degree_hist_link, 1)]

    ex_plain = CausalSetZoology._field_extractor(fields)
    @test ex_plain.nfields == 3
    @test ex_plain.symbol_specs == symbol_specs
    @test ex_plain.hist_specs == hist_specs
    @test ex_plain.scalar === nothing

    ex_scalar = CausalSetZoology._field_extractor(fields, :scalar)
    @test ex_scalar.nfields == 3
    @test ex_scalar.symbol_specs == symbol_specs
    @test ex_scalar.hist_specs == hist_specs
    @test ex_scalar.scalar == :scalar
end

@testitem "dataloading helpers: low-level loaders" setup=[setupDataloading] begin
    # Test intent: validate dataloading helpers: low-level loaders behavior and output contract.
    p1, _ = _make_dl_fixture()
    pbad = _make_dl_bad_scalar_fixture()

    fields = Union{Symbol,Tuple{Symbol,Int64}}[:score, (:in_degree_hist_link, 1)]
    ex_plain = CausalSetZoology._field_extractor(fields)
    ex_scalar = CausalSetZoology._field_extractor(fields, :scalar)
    cfg1f = CausalSetZoology._scan_config(1.0)
    @test CausalSetZoology._load_fields_one(p1, nothing, cfg1f, ex_plain) == [[1.0, 2.0], [1.0, 2.0]]
    @test CausalSetZoology._load_fields_one(p1, nothing, cfg1f, ex_scalar) == [[(1.0, 10.0), (2.0, 20.0)], [(1.0, 10.0), (2.0, 20.0)]]
    @test_throws TypeError CausalSetZoology._load_fields_one(pbad, nothing, cfg1f, ex_scalar)

    h_plain = CausalSetZoology._HistogramExtractor(:in_degree_hist_link, nothing)
    h_scalar = CausalSetZoology._HistogramExtractor(:in_degree_hist_link, :scalar)
    cfg1i = CausalSetZoology._scan_config(1)
    @test CausalSetZoology._load_histograms_one(p1, nothing, cfg1i, h_plain) == [Dict(1 => 1.0), Dict(1 => 2.0)]
    @test CausalSetZoology._load_histograms_one(p1, nothing, cfg1i, h_scalar) == [(Dict(1 => 1.0), 10.0), (Dict(1 => 2.0), 20.0)]
    @test_throws TypeError CausalSetZoology._load_histograms_one(pbad, nothing, cfg1i, h_scalar)

    @test CausalSetZoology._load_field_with_scalar_one(p1, nothing, cfg1f, :score, :scalar) == [(1.0, 10.0), (2.0, 20.0)]
    @test CausalSetZoology._load_field_with_scalar_one(p1, nothing, cfg1f, (:in_degree_hist_link, 1), :scalar) == [(1.0, 10.0), (2.0, 20.0)]
    @test_throws TypeError CausalSetZoology._load_field_with_scalar_one(pbad, nothing, cfg1f, :score, :scalar)
end

@testitem "dataloading: load_fields_from_paths plain" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_fields_from_paths plain behavior and output contract.
    p1, _ = _make_dl_fixture()
    fields = Union{Symbol,Tuple{Symbol,Int64}}[:score, (:in_degree_hist_link, 1)]

    f = CausalSetZoology.load_fields_from_paths([p1], fields)
    @test f == [[[1.0, 2.0], [1.0, 2.0]]]
end

@testitem "dataloading: load_fields_from_paths scalar" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_fields_from_paths scalar behavior and output contract.
    p1, _ = _make_dl_fixture()

    fs = CausalSetZoology.load_fields_from_paths([p1], [:score], :scalar)
    @test fs == [[[(1.0, 10.0), (2.0, 20.0)]]]
    @test [s for (_, s) in fs[1][1]] == [10.0, 20.0]
end

@testitem "dataloading: load_fields_from_paths validation" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_fields_from_paths validation behavior and output contract.
    p1, _ = _make_dl_fixture()
    pbad = _make_dl_bad_scalar_fixture()
    fields = Union{Symbol,Tuple{Symbol,Int64}}[:score, (:in_degree_hist_link, 1)]

    @test_throws DimensionMismatch CausalSetZoology.load_fields_from_paths([p1], fields; filters = Union{Nothing,Function}[nothing, nothing])
    @test_throws DomainError CausalSetZoology.load_fields_from_paths([p1], fields; thinning = 0.0)
    @test_throws DimensionMismatch CausalSetZoology.load_fields_from_paths([p1], fields, :scalar; filters = Union{Nothing,Function}[nothing, nothing])

    err = try
        CausalSetZoology.load_fields_from_paths([pbad], fields, :scalar)
        nothing
    catch e
        e
    end
    @test _contains_typeerror(err)
end

@testitem "dataloading: load_field_with_scalar" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_field_with_scalar behavior and output contract.
    p1, _ = _make_dl_fixture()

    one_sym = CausalSetZoology.load_field_with_scalar([p1], :score, :scalar)
    @test one_sym == [[(1.0, 10.0), (2.0, 20.0)]]

    one_hist = CausalSetZoology.load_field_with_scalar([p1], (:in_degree_hist_link, 1), :scalar)
    @test one_hist == [[(1.0, 10.0), (2.0, 20.0)]]
end

@testitem "dataloading: load_field_with_scalar validation" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_field_with_scalar validation behavior and output contract.
    p1, _ = _make_dl_fixture()
    pbad = _make_dl_bad_scalar_fixture()

    @test_throws DimensionMismatch CausalSetZoology.load_field_with_scalar([p1], :score, :scalar; filters = Union{Nothing,Function}[nothing, nothing])
    @test_throws DomainError CausalSetZoology.load_field_with_scalar([p1], :score, :scalar; thinning = 1.5)
    @test_throws TypeError CausalSetZoology.load_field_with_scalar([pbad], :score, :scalar)
end

@testitem "dataloading: load_and_average_std_scalar plain" setup=[setupDataloading] begin
    # Test intent: validate dataloading: load_and_average_std_scalar plain behavior and output contract.
    p1, _ = _make_dl_fixture()

    av = CausalSetZoology.load_and_average_std_scalar([p1], [:score])
    @test av == [[(1.5, sqrt(0.5))]]
end
