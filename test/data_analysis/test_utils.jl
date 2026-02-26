@testsnippet setupUtils begin
    using Test
    using Statistics
    using Random
    using JLD2

    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "dataloading.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "data_analysis", "utils.jl"))
end

@testitem "utils: get_size and histogram/vector helpers" setup=[setupUtils] begin
    mktemp() do path, io
        close(io)
        JLD2.jldopen(path, "w") do f
            f["meta/config"] = Dict("cset_size" => 123)
        end
        @test get_size(path) == 123
    end

    h = [[Dict(1 => 2, 2 => 2), Dict(1 => 1, 2 => 3)]]
    np = normalize_hists(h; normalization = :probability)
    @test sum(values(np[1][1])) ≈ 1.0
    nm = normalize_hists(h; normalization = :max)
    @test maximum(values(nm[1][2])) ≈ 1.0

    hs = [[(Dict(1 => 2, 2 => 2), 1.0), (Dict(1 => 1, 2 => 3), 2.0)]]
    ns = normalize_hists(hs; normalization = :probability, num_bins = 2)
    @test length(ns[1]) == 2

    meanh, stdh = average_histogram_with_std([Dict(1 => 1, 2 => 3), Dict(1 => 3, 2 => 1)])
    @test meanh == [2.0, 2.0]
    @test all(x -> x ≥ 0, stdh)

    meanv, stdv = average_vectors_with_std([[1.0, 2.0], [3.0, 4.0]])
    @test meanv == [2.0, 3.0]
    @test all(x -> x > 0, stdv)
    mean_nested, std_nested = average_vectors_with_std([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    @test length(mean_nested) == 2
    @test all(x -> x >= 0, std_nested)

    grouped = average_vectors_with_std([([1.0, 2.0], 1.0), ([2.0, 3.0], 1.0), ([3.0, 4.0], 2.0)])
    @test length(grouped) == 2

    grouped_hist = average_histogram_with_std([(Dict(1 => 1, 2 => 2), 1.0), (Dict(1 => 2, 2 => 1), 1.0)])
    @test length(grouped_hist) == 1

    rz = replace_zeros([0.0, 2.0, 0.0]; ϵ = 1e-2)
    @test rz[2] == 2.0
    @test rz[1] > 0

    shifted = abundance_shift(Dict(1 => 10, 2 => 11, 4 => 13))
    @test !haskey(shifted, -1)
    @test shifted[0] == 11
    @test shifted[2] == 13

    @test_throws AssertionError normalize_hists([[Dict{Int,Int}()]]; normalization = :probability)
    @test_throws AssertionError average_vectors_with_std([[1.0], [1.0, 2.0]])
    @test_throws AssertionError average_vectors_with_std([([1.0], 1.0)]; num_bins = 0)
    @test_throws AssertionError average_histogram_with_std([(Dict(1 => 1), 1.0)]; num_bins = 0)
end
