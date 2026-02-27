@testsnippet setupLaplacian begin
    using Test
    using Random
    using LinearAlgebra
    using CairoMakie
    using Colors
    using Observables
    using Printf

end

@testitem "laplacian_eigenvalue_example: graph generation and plotting" setup=[setupLaplacian] begin
    inter = [0 1; 1 0]
    A = CausalSetZoology.make_undirected_adjacency_from_subgraphs(6, 2, inter; p_internal = 0.0, rng = Random.Xoshiro(1))
    @test size(A) == (6, 6)
    @test issymmetric(A)
    @test all(!A[i, i] for i in 1:6)

    λ = CausalSetZoology.normalized_laplacian_eigenvalues(A)
    @test length(λ) == 6
    @test all(isfinite, λ)

    set_theme!(Theme())
    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1])
    ax2 = CausalSetZoology._draw_subgraph_colored_node_link!(ax, A, 2)
    @test ax2 === ax

    f1 = CausalSetZoology.plot_subgraph_colored_node_link(A, 2; apply_theme = false, fig_size = (500, 400))
    @test f1 isa Figure

    f2 = CausalSetZoology.plot_subgraph_colored_node_link(A, A, A, 2; apply_theme = false, fig_size = (900, 300))
    @test f2 isa Figure

    @test_throws ArgumentError CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, [1 0; 0 0])
    @test_throws ArgumentError CausalSetZoology.normalized_laplacian_eigenvalues(BitMatrix([0 1; 0 0]))
    @test_throws ArgumentError CausalSetZoology.normalized_laplacian_eigenvalues(A; zero_tol = -1.0)
end
