@testsnippet setupDataloading begin
    using Test
    using Statistics
    using JLD2
    using Random


    Base.@kwdef struct _DLRec
        in_degree_hist_link::Dict{Int,Float64}
        out_degree_hist_link::Dict{Int,Float64}
        max_pathlen_hist::Dict{Int,Float64}
        ev_sym_link::Vector{Float64}
        score::Float64
        scalar::Float64
    end

    function _write_dl_file(path::String, batches::Vector{Vector{_DLRec}})
        JLD2.jldopen(path, "w") do f
            f["meta/nbatches"] = length(batches)
            for (i, b) in enumerate(batches)
                f["batches/$i"] = b
            end
        end
    end
end

@testitem "dataloading: load, join, densify" setup=[setupDataloading] begin
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

    h = CausalSetZoology.load_histograms_from_paths([p1, p2], :in_degree_hist_link)
    @test length(h) == 2
    @test length(h[1]) == 2

    hs = CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link, :scalar)
    @test hs[1][1][2] == 10.0

    @test size(CausalSetZoology.densify_hists([Dict(0 => 1.0, 2 => 2.0), Dict(1 => 1.0)])) == (2, 3)

    d1::Dict = Dict{Int,Int}(1 => 1)
    d2::Dict = Dict{Int,Int}(1 => 2, 2 => 1)
    jh = CausalSetZoology.join_histograms(Vector{Vector{Vector{Dict}}}([[[d1]], [[d2]]]))
    @test jh[1][1][1] == 3

    t1 = (Dict{Int,Float64}(1 => 1.0)::Dict, 5.0::Real)
    t2 = (Dict{Int,Float64}(1 => 2.0)::Dict, 5.0::Real)
    jhs = CausalSetZoology.join_histograms(Vector{Vector{Vector{Tuple{Dict,Real}}}}([[[t1]], [[t2]]]))
    @test jhs[1][1][1][1] == 3.0
    @test jhs[1][1][2] == 5.0

    fields = Union{Symbol,Tuple{Symbol,Int64}}[:score, (:in_degree_hist_link, 1)]
    f = CausalSetZoology.load_fields_from_paths([p1], fields)
    @test f[1][1] == [1.0, 2.0]
    @test f[1][2] == [1.0, 2.0]

    fs = CausalSetZoology.load_fields_from_paths([p1], [:score], :scalar)
    @test fs[1][1][1] == (1.0, 10.0)

    one = CausalSetZoology.load_field_with_scalar([p1], :score, :scalar)
    @test one[1][2] == (2.0, 20.0)

    av = CausalSetZoology.load_and_average_std_scalar([p1], [:score])
    @test av[1][1][1] ≈ 1.5

    avs = CausalSetZoology.load_and_average_std_scalar([p1], [:score], :scalar)
    @test length(avs[1]) == 2

    # filtering/thinning branches
    filt = Union{Nothing,Function}[x -> x.score > 1.5]
    hf = CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; filters = filt, thinning = 1)
    @test length(hf[1]) == 1
    ht = CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; thinning = 2)
    @test length(ht[1]) == 1

    # validation branches
    @test_throws TypeError CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; filters = [nothing, nothing])
    @test_throws AssertionError CausalSetZoology.load_histograms_from_paths([p1], :in_degree_hist_link; thinning = 0)
    @test_throws AssertionError CausalSetZoology.load_fields_from_paths([p1], fields; thinning = 0.0)
    @test_throws AssertionError CausalSetZoology.load_field_with_scalar([p1], :score, :scalar; thinning = 1.5)
    @test_throws AssertionError CausalSetZoology.join_histograms(Vector{Vector{Vector{Tuple{Dict,Real}}}}([[[t1]], [[(Dict{Int,Float64}(1 => 2.0)::Dict, 6.0::Real)]]]))
end
