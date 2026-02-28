@testsnippet setupLaplacian begin
    using Test
    using Random
    using LinearAlgebra
    using CairoMakie

    figure_axes(fig::Figure) = [b for b in fig.content if b isa Axis]
    nplots(ax::Axis) = length(ax.scene.plots)
end

# Verifies deterministic graph construction properties and edge accounting.
@testitem "laplacian_eigenvalue_example: make_undirected adjacency structure" setup=[setupLaplacian] begin
    # Two equal subgraphs of size 3 each; no internal edges; exactly 2 cross edges.
    inter = [0 2; 2 0]
    A = CausalSetZoology.make_undirected_adjacency_from_subgraphs(
        6,
        2,
        inter;
        p_internal = 0.0,
        rng = Random.Xoshiro(1),
    )

    @test size(A) == (6, 6)
    @test issymmetric(A)
    @test all(!A[i, i] for i in 1:6)

    # With p_internal=0, only requested inter-subgraph edges should exist.
    @test count(A[1:3, 1:3]) == 0
    @test count(A[4:6, 4:6]) == 0
    @test count(A[1:3, 4:6]) == 2
    @test count(A[4:6, 1:3]) == 2
end

# Verifies internal-edge construction for p_internal=1.0.
@testitem "laplacian_eigenvalue_example: make_undirected internal density extremes" setup=[setupLaplacian] begin
    # With p_internal=1, each subgraph should be a clique.
    inter = zeros(Int, 2, 2)
    A = CausalSetZoology.make_undirected_adjacency_from_subgraphs(
        6,
        2,
        inter;
        p_internal = 1.0,
        rng = Random.Xoshiro(2),
    )

    # Each 3-node clique contributes 3 undirected edges => 6 true off-diagonal entries per block.
    @test count(A[1:3, 1:3]) == 6
    @test count(A[4:6, 4:6]) == 6
    @test count(A[1:3, 4:6]) == 0
end

# Validates input/domain checks for graph-construction helper.
@testitem "laplacian_eigenvalue_example: make_undirected adjacency validation" setup=[setupLaplacian] begin
    # Domain constraints on node/subgraph counts.
    @test_throws DomainError CausalSetZoology.make_undirected_adjacency_from_subgraphs(0, 2, [0 0; 0 0])
    @test_throws DomainError CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 5, zeros(Int, 5, 5))

    # Matrix shape/symmetry/diagonal/content checks.
    @test_throws DimensionMismatch CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, zeros(Int, 1, 2))
    @test_throws DimensionMismatch CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, zeros(Int, 2, 1))
    @test_throws ArgumentError CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, [0 1; 0 0])
    @test_throws DomainError CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, [0 -1; -1 0])
    @test_throws ArgumentError CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, [1 0; 0 0])

    # Probability and feasibility checks.
    @test_throws DomainError CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, [0 0; 0 0]; p_internal = -0.1)
    @test_throws DomainError CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, [0 0; 0 0]; p_internal = 1.1)
    @test_throws DomainError CausalSetZoology.make_undirected_adjacency_from_subgraphs(4, 2, [0 5; 5 0]; p_internal = 0.0)
end

# Verifies known-spectrum behavior for normalized Laplacian.
@testitem "laplacian_eigenvalue_example: normalized_laplacian eigenvalues basic" setup=[setupLaplacian] begin
    # Single edge graph K2 has normalized-Laplacian spectrum {0, 2}.
    A2 = BitMatrix([0 1; 1 0])
    λ2 = sort(CausalSetZoology.normalized_laplacian_eigenvalues(A2))
    @test λ2 ≈ [0.0, 2.0] atol = 1e-12

    # Isolated vertices produce identity Laplacian => all eigenvalues 1.
    Aiso = falses(3, 3)
    λiso = sort(CausalSetZoology.normalized_laplacian_eigenvalues(Aiso))
    @test λiso ≈ [1.0, 1.0, 1.0] atol = 1e-12
end

# Verifies zero-tolerance snapping behavior.
@testitem "laplacian_eigenvalue_example: normalized_laplacian zero tolerance" setup=[setupLaplacian] begin
    A2 = BitMatrix([0 1; 1 0])

    # Extremely small values near 0 should be snapped to exactly 0 when within zero_tol.
    λ = CausalSetZoology.normalized_laplacian_eigenvalues(A2; zero_tol = 1e-6)
    @test any(==(0.0), λ)
end

# Validates error paths for normalized-Laplacian helper.
@testitem "laplacian_eigenvalue_example: normalized_laplacian validation" setup=[setupLaplacian] begin
    # Non-square adjacency should throw.
    @test_throws DimensionMismatch CausalSetZoology.normalized_laplacian_eigenvalues(BitMatrix([0 1 0; 1 0 1]))

    # Non-symmetric and nonzero-diagonal adjacency should throw.
    @test_throws ArgumentError CausalSetZoology.normalized_laplacian_eigenvalues(BitMatrix([0 1; 0 0]))
    @test_throws ArgumentError CausalSetZoology.normalized_laplacian_eigenvalues(BitMatrix([1 1; 1 0]))

    # zero_tol must be nonnegative.
    @test_throws DomainError CausalSetZoology.normalized_laplacian_eigenvalues(BitMatrix([0 1; 1 0]); zero_tol = -1.0)
end

# Verifies core node-link drawing helper mutates axis with visible plot primitives.
@testitem "laplacian_eigenvalue_example: draw subgraph node-link basic" setup=[setupLaplacian] begin
    # Standalone helper plotting: initialize project theme explicitly.
    CausalSetZoology.apply_paper_theme!(logscale_x = false, logscale_y = false)
    A = BitMatrix([
        0 1 0 0;
        1 0 0 1;
        0 0 0 1;
        0 1 1 0;
    ])
    fig = Figure(size = (400, 300))
    ax = Axis(fig[1, 1])

    ax2 = CausalSetZoology._draw_subgraph_colored_node_link!(
        ax,
        A,
        2;
        cluster_radius = 5.0,
        local_base = 1.0,
        local_scale = 0.4,
        node_size = 8.0,
        edge_alpha = 0.3,
        inter_edge_alpha = 0.1,
    )

    @test ax2 === ax
    @test nplots(ax) >= 2
end

# Validates error paths for node-link drawing helper.
@testitem "laplacian_eigenvalue_example: draw subgraph node-link validation" setup=[setupLaplacian] begin
    # Standalone helper plotting: initialize project theme explicitly.
    CausalSetZoology.apply_paper_theme!(logscale_x = false, logscale_y = false)
    fig = Figure(size = (300, 250))
    ax = Axis(fig[1, 1])

    # Shape/symmetry/subgraph-count checks.
    @test_throws DimensionMismatch CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1 0; 1 0 1]), 1)
    @test_throws ArgumentError CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1; 0 0]), 1)
    @test_throws DomainError CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1; 1 0]), 0)

    # Layout/styling domains.
    @test_throws DomainError CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1; 1 0]), 1; cluster_radius = 0.0)
    @test_throws DomainError CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1; 1 0]), 1; local_base = -1.0)
    @test_throws DomainError CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1; 1 0]), 1; local_scale = -1.0)
    @test_throws DomainError CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1; 1 0]), 1; node_size = 0.0)
    @test_throws DomainError CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1; 1 0]), 1; edge_alpha = -0.1)
    @test_throws DomainError CausalSetZoology._draw_subgraph_colored_node_link!(ax, BitMatrix([0 1; 1 0]), 1; inter_edge_alpha = 1.1)
end

# Verifies single-panel plotting wrapper, including explicit fig_size and save path.
@testitem "laplacian_eigenvalue_example: plot_subgraph_colored_node_link single basic" setup=[setupLaplacian] begin
    A = BitMatrix([
        0 1 0 0;
        1 0 1 0;
        0 1 0 1;
        0 0 1 0;
    ])
    out = joinpath(mktempdir(), "subgraph_single.png")

    fig = CausalSetZoology.plot_subgraph_colored_node_link(
        A,
        2;
        apply_theme = false,
        fig_size = (520, 360),
        fig_path = out,
        cluster_radius = 5.0,
    )

    @test fig isa Figure
    @test length(figure_axes(fig)) == 1
    @test isfile(out)
end

# Validates single-panel plotting wrapper input checks and propagated errors.
@testitem "laplacian_eigenvalue_example: plot_subgraph_colored_node_link single validation" setup=[setupLaplacian] begin
    A = BitMatrix([0 1; 1 0])

    @test_throws DomainError CausalSetZoology.plot_subgraph_colored_node_link(A, 1; theme_magnification = 0.0)
    @test_throws DomainError CausalSetZoology.plot_subgraph_colored_node_link(A, 1; color_transparency = 1.1)
    @test_throws DomainError CausalSetZoology.plot_subgraph_colored_node_link(A, 1; fig_size = (0, 100))

    # Propagated from draw helper.
    @test_throws DomainError CausalSetZoology.plot_subgraph_colored_node_link(A, 0; apply_theme = false)
end

# Verifies three-panel plotting wrapper basic behavior and save path.
@testitem "laplacian_eigenvalue_example: plot_subgraph_colored_node_link triple basic" setup=[setupLaplacian] begin
    A1 = BitMatrix([0 1 0; 1 0 1; 0 1 0])
    A2 = BitMatrix([0 1 1; 1 0 0; 1 0 0])
    A3 = BitMatrix([0 0 1; 0 0 1; 1 1 0])
    out = joinpath(mktempdir(), "subgraph_triple.png")

    fig = CausalSetZoology.plot_subgraph_colored_node_link(
        A1,
        A2,
        A3,
        2;
        apply_theme = false,
        fig_size = (900, 320),
        fig_path = out,
    )

    @test fig isa Figure
    @test length(figure_axes(fig)) == 3
    @test isfile(out)
end

# Validates three-panel plotting wrapper input checks and propagated errors.
@testitem "laplacian_eigenvalue_example: plot_subgraph_colored_node_link triple validation" setup=[setupLaplacian] begin
    A = BitMatrix([0 1; 1 0])

    @test_throws DomainError CausalSetZoology.plot_subgraph_colored_node_link(A, A, A, 1; theme_magnification = 0.0)
    @test_throws DomainError CausalSetZoology.plot_subgraph_colored_node_link(A, A, A, 1; color_transparency = -0.1)
    @test_throws DomainError CausalSetZoology.plot_subgraph_colored_node_link(A, A, A, 1; fig_size = (100, 0))

    # Propagated from draw helper on one panel.
    @test_throws DomainError CausalSetZoology.plot_subgraph_colored_node_link(A, A, A, 0; apply_theme = false)
end
