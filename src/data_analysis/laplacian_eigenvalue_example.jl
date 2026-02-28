"""
    make_undirected_adjacency_from_subgraphs(
        n_nodes::Int,
        n_subgraphs::Int,
        inter_edges::AbstractMatrix{<:Integer};
        p_internal::Float64 = 0.5,
        rng = nothing,
    )::BitMatrix

Create a symmetric adjacency matrix for an undirected simple graph.

- Nodes are split into `n_subgraphs` contiguous, near-equal subgraphs.
- Inside each subgraph, each edge is added independently with probability `p_internal`.
- For each subgraph pair `(a,b)`, exactly `inter_edges[a,b]` edges are added between
  randomly chosen node pairs from those two subgraphs (without duplicates).

# Arguments
- `n_nodes`: Input parameter `n_nodes` used by this method.
- `n_subgraphs`: Input parameter `n_subgraphs` used by this method.
- `inter_edges`: Input parameter `inter_edges` used by this method.

# Keyword Arguments
- `p_internal`: Keyword option `p_internal` controlling this method's behavior.
- `rng`: Random number generator used for stochastic steps.

# Returns
- `result::BitMatrix`: Output of `make_undirected_adjacency_from_subgraphs` with type annotation `BitMatrix`.

# Throws
- `DomainError`: Raised when numeric parameters violate domain constraints.
- `DimensionMismatch`: Raised when `inter_edges` shape does not match `n_subgraphs`.
- `ArgumentError`: Raised when `inter_edges` is not symmetric or has nonzero diagonal."""
function make_undirected_adjacency_from_subgraphs(
    n_nodes::Int,
    n_subgraphs::Int,
    inter_edges::AbstractMatrix{<:Integer};
    p_internal::Float64 = 0.5,
    rng = nothing,
)::BitMatrix
    if !(n_nodes >= 1)
        throw(DomainError(n_nodes, "n_nodes must be >= 1"))
    end
    if !(1 <= n_subgraphs <= n_nodes)
        throw(DomainError(n_subgraphs, "n_subgraphs must satisfy 1 <= n_subgraphs <= n_nodes (n_nodes=$n_nodes)"))
    end
    if !(size(inter_edges, 1) == n_subgraphs)
        throw(DimensionMismatch("inter_edges has $(size(inter_edges, 1)) rows but n_subgraphs=$n_subgraphs"))
    end
    if !(size(inter_edges, 2) == n_subgraphs)
        throw(DimensionMismatch("inter_edges has $(size(inter_edges, 2)) columns but n_subgraphs=$n_subgraphs"))
    end
    if !(LinearAlgebra.issymmetric(inter_edges))
        throw(ArgumentError("inter_edges must be symmetric"))
    end
    if !(all(inter_edges .>= 0))
        throw(DomainError(minimum(inter_edges), "inter_edges must contain only nonnegative counts"))
    end
    if !(all(inter_edges[i, i] == 0 for i in 1:n_subgraphs))
        throw(ArgumentError("inter_edges diagonal must be zero"))
    end
    if !(0.0 <= p_internal <= 1.0)
        throw(DomainError(p_internal, "p_internal must satisfy 0.0 <= p_internal <= 1.0"))
    end
    # near-equal partition sizes: first `r` subgraphs get one extra node
    q, r = divrem(n_nodes, n_subgraphs)
    sizes = fill(q, n_subgraphs)
    for i in 1:r
        sizes[i] += 1
    end

    ranges = Vector{UnitRange{Int}}(undef, n_subgraphs)
    start_idx = 1
    for s in 1:n_subgraphs
        stop_idx = start_idx + sizes[s] - 1
        ranges[s] = start_idx:stop_idx
        start_idx = stop_idx + 1
    end

    rand01() = rng === nothing ? rand() : rand(rng)
    randint(lo::Int, hi::Int) = rng === nothing ? rand(lo:hi) : rand(rng, lo:hi)

    A = falses(n_nodes, n_nodes)

    # random edges inside each subgraph
    for sg in 1:n_subgraphs
        idx = ranges[sg]
        for i in first(idx):(last(idx) - 1)
            for j in (i + 1):last(idx)
                if rand01() < p_internal
                    A[i, j] = true
                    A[j, i] = true
                end
            end
        end
    end

    # requested edges between distinct subgraphs
    for a in 1:(n_subgraphs - 1)
        ra = ranges[a]
        na = length(ra)
        for b in (a + 1):n_subgraphs
            rb = ranges[b]
            nb = length(rb)
            requested = Int(inter_edges[a, b])
            max_possible = na * nb
            if !(requested ≤ max_possible)
                throw(
                    DomainError(
                        requested,
                        "requested inter-subgraph edges for pair ($a,$b) exceed maximum $max_possible",
                    ),
                )
            end
            used = Set{Tuple{Int,Int}}()
            while length(used) < requested
                u = first(ra) + randint(1, na) - 1
                v = first(rb) + randint(1, nb) - 1
                if (u, v) ∉ used
                    push!(used, (u, v))
                    A[u, v] = true
                    A[v, u] = true
                end
            end
        end
    end

    return A
end

"""
    normalized_laplacian_eigenvalues(
        A::BitMatrix;
        zero_tol::Float64 = 1e-12,
    )::Vector{Float64}

Compute eigenvalues of the symmetrically normalized graph Laplacian

`L_sym = I - D^{-1/2} A D^{-1/2}`

for an undirected simple graph adjacency matrix `A`.

For isolated vertices (`degree = 0`), `D^{-1/2}` is set to 0.
Eigenvalues with `abs(λ) < zero_tol` are returned as exact `0.0`.

# Arguments
- `A`: Input parameter `A` used by this method.

# Keyword Arguments
- `zero_tol`: Keyword option `zero_tol` controlling this method's behavior.

# Returns
- `result::Vector{Float64}`: Output of `normalized_laplacian_eigenvalues` with type annotation `Vector{Float64}`.

# Throws
- `DimensionMismatch`: Raised when `A` is not square.
- `DomainError`: Raised when numeric parameters violate domain constraints.
- `ArgumentError`: Raised when `A` is not symmetric or has a nonzero diagonal."""
function normalized_laplacian_eigenvalues(
    A::BitMatrix;
    zero_tol::Float64 = 1e-12,
)::Vector{Float64}
    n, m = size(A)
    if !(n == m)
        throw(DimensionMismatch("adjacency matrix must be square; got size ($n, $m)"))
    end
    if !(LinearAlgebra.issymmetric(A))
        throw(ArgumentError("adjacency matrix must be symmetric"))
    end
    if !(all(!A[i, i] for i in 1:n))
        throw(ArgumentError("adjacency matrix diagonal must be zero"))
    end
    if !(zero_tol ≥ 0.0)
        throw(DomainError(zero_tol, "zero_tol must be >= 0"))
    end
    degrees = vec(sum(A, dims = 2))
    invsqrtdeg = zeros(Float64, n)
    for i in 1:n
        d = degrees[i]
        if d > 0
            invsqrtdeg[i] = inv(sqrt(Float64(d)))
        end
    end

    Dhalf = LinearAlgebra.Diagonal(invsqrtdeg)
    Afloat = Matrix{Float64}(A)
    L = Matrix{Float64}(LinearAlgebra.I, n, n) - Dhalf * Afloat * Dhalf

    λ = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(L))
    for i in eachindex(λ)
        if abs(λ[i]) < zero_tol
            λ[i] = 0.0
        end
    end

    return λ
end

"""
    _draw_subgraph_colored_node_link!(
        ax,
        A::BitMatrix,
        n_subgraphs::Int;
        cluster_radius::Real = 6.0,
        local_base::Real = 1.8,
        local_scale::Real = 0.5,
        node_size::Real = 10,
        edge_alpha::Real = 0.25,
        inter_edge_alpha::Real = 0.08,
    )::CairoMakie.Axis

Draw subgraph-colored node-link geometry onto an existing Makie axis.

Nodes are partitioned into `n_subgraphs` contiguous groups, arranged in separate
clusters, and colored by group membership. Edges are drawn with different alpha
values for intra- vs inter-subgraph connections.

# Returns
- The mutated axis `ax`.

# Arguments
- `ax`: Input parameter `ax` used by this method.
- `A`: Input parameter `A` used by this method.
- `n_subgraphs`: Input parameter `n_subgraphs` used by this method.

# Keyword Arguments
- `cluster_radius`: Keyword option `cluster_radius` controlling this method's behavior.
- `local_base`: Keyword option `local_base` controlling this method's behavior.
- `local_scale`: Keyword option `local_scale` controlling this method's behavior.
- `node_size`: Keyword option `node_size` controlling this method's behavior.
- `edge_alpha`: Keyword option `edge_alpha` controlling this method's behavior.
- `inter_edge_alpha`: Keyword option `inter_edge_alpha` controlling this method's behavior.

# Throws
- `DimensionMismatch`: Raised when `A` is not square.
- `DomainError`: Raised when numeric layout/styling parameters violate domain constraints.
- `ArgumentError`: Raised when `A` is not symmetric."""
function _draw_subgraph_colored_node_link!(
    ax,
    A::BitMatrix,
    n_subgraphs::Int;
    cluster_radius::Real = 6.0,
    local_base::Real = 1.8,
    local_scale::Real = 0.5,
    node_size::Real = 10,
    edge_alpha::Real = 0.25,
    inter_edge_alpha::Real = 0.08,
)
    n, m = size(A)
    if !(n == m)
        throw(DimensionMismatch("adjacency matrix must be square; got size ($n, $m)"))
    end
    if !(LinearAlgebra.issymmetric(A))
        throw(ArgumentError("adjacency matrix must be symmetric"))
    end
    if !(1 <= n_subgraphs <= n)
        throw(DomainError(n_subgraphs, "n_subgraphs must satisfy 1 <= n_subgraphs <= n (n=$n)"))
    end
    if !(cluster_radius > 0)
        throw(DomainError(cluster_radius, "cluster_radius must be > 0"))
    end
    if !(local_base ≥ 0)
        throw(DomainError(local_base, "local_base must be >= 0"))
    end
    if !(local_scale ≥ 0)
        throw(DomainError(local_scale, "local_scale must be >= 0"))
    end
    if !(node_size > 0)
        throw(DomainError(node_size, "node_size must be > 0"))
    end
    if !(0 <= edge_alpha <= 1)
        throw(DomainError(edge_alpha, "edge_alpha must satisfy 0 <= edge_alpha <= 1"))
    end
    if !(0 <= inter_edge_alpha <= 1)
        throw(DomainError(inter_edge_alpha, "inter_edge_alpha must satisfy 0 <= inter_edge_alpha <= 1"))
    end
    # Same near-equal contiguous partition as in graph generation.
    q, r = divrem(n, n_subgraphs)
    sizes = fill(q, n_subgraphs)
    for i in 1:r
        sizes[i] += 1
    end

    ranges = Vector{UnitRange{Int}}(undef, n_subgraphs)
    start_idx = 1
    for s in 1:n_subgraphs
        stop_idx = start_idx + sizes[s] - 1
        ranges[s] = start_idx:stop_idx
        start_idx = stop_idx + 1
    end

    # Cluster centers on a circle; nodes on local circles around each center.
    # Defaults are chosen so medium-sized subgraphs (e.g. 3x10 nodes) are
    # visually larger and closer than before.
    pos = Vector{CairoMakie.Point2f}(undef, n)
    subgraph_of_node = Vector{Int}(undef, n)

    for sg in 1:n_subgraphs
        θc = 2π * (sg - 1) / n_subgraphs
        cx = cluster_radius * cos(θc)
        cy = cluster_radius * sin(θc)
        idx = ranges[sg]
        k = length(idx)
        local_r = local_base + local_scale * sqrt(k)
        for (t, node) in enumerate(idx)
            θ = 2π * (t - 1) / max(k, 1)
            x = cx + local_r * cos(θ)
            y = cy + local_r * sin(θ)
            pos[node] = CairoMakie.Point2f(x, y)
            subgraph_of_node[node] = sg
        end
    end

    colors_obs = CairoMakie.theme(:palette).color
    colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
    node_colors = [colors[mod1(subgraph_of_node[i], length(colors))] for i in 1:n]

    CairoMakie.hidedecorations!(ax)
    CairoMakie.hidespines!(ax)
    ax.aspect = CairoMakie.DataAspect()

    for i in 1:(n - 1)
        pi = pos[i]
        for j in (i + 1):n
            if A[i, j]
                pj = pos[j]
                same = subgraph_of_node[i] == subgraph_of_node[j]
                α = same ? edge_alpha : inter_edge_alpha
                CairoMakie.lines!(ax, [pi[1], pj[1]], [pi[2], pj[2]], color = (:black, α), linewidth = 1)
            end
        end
    end

    CairoMakie.scatter!(ax, first.(pos), last.(pos), color = node_colors, markersize = node_size)
    return ax
end

"""
    plot_subgraph_colored_node_link(
        A::BitMatrix,
        n_subgraphs::Int;
        fig_size::Union{Nothing,Tuple{Int,Int}} = nothing,
        apply_theme::Bool = true,
        theme_double_column::Bool = false,
        theme_magnification::Real = 1.0,
        color_transparency::Float64 = 1.0,
        cluster_radius::Real = 6.0,
        local_base::Real = 1.8,
        local_scale::Real = 0.5,
        node_size::Real = 10,
        edge_alpha::Real = 0.25,
        inter_edge_alpha::Real = 0.08,
        fig_path::Union{Nothing,String} = nothing,
    )::CairoMakie.Figure

Create and return a single-panel node-link figure colored by subgraph.

This is the primary plotting method for this function name.

# Arguments
- `A`: Input parameter `A` used by this method.
- `n_subgraphs`: Input parameter `n_subgraphs` used by this method.

# Keyword Arguments
- `fig_size`: Keyword option `fig_size` controlling this method's behavior.
- `apply_theme`: Keyword option `apply_theme` controlling this method's behavior.
- `theme_double_column`: Keyword option `theme_double_column` controlling this method's behavior.
- `theme_magnification`: Keyword option `theme_magnification` controlling this method's behavior.
- `color_transparency`: Keyword option `color_transparency` controlling this method's behavior.
- `cluster_radius`: Keyword option `cluster_radius` controlling this method's behavior.
- `local_base`: Keyword option `local_base` controlling this method's behavior.
- `local_scale`: Keyword option `local_scale` controlling this method's behavior.
- `node_size`: Keyword option `node_size` controlling this method's behavior.
- `edge_alpha`: Keyword option `edge_alpha` controlling this method's behavior.
- `inter_edge_alpha`: Keyword option `inter_edge_alpha` controlling this method's behavior.
- `fig_path`: Path or collection of paths used for loading/saving data.

# Returns
- `result::CairoMakie.Figure`: Output of `plot_subgraph_colored_node_link` with type annotation `CairoMakie.Figure`.

# Throws
- `DomainError`: Raised when visual parameter domains are invalid.
- `DimensionMismatch`: Propagated from `_draw_subgraph_colored_node_link!` when `A` is not square.
- `ArgumentError`: Propagated from `_draw_subgraph_colored_node_link!` when `A` is not symmetric."""
function plot_subgraph_colored_node_link(
    A::BitMatrix,
    n_subgraphs::Int;
    fig_size::Union{Nothing,Tuple{Int,Int}} = nothing,
    apply_theme::Bool = true,
    theme_double_column::Bool = false,
    theme_magnification::Real = 1.0,
    color_transparency::Float64 = 1.0,
    cluster_radius::Real = 6.0,
    local_base::Real = 1.8,
    local_scale::Real = 0.5,
    node_size::Real = 10,
    edge_alpha::Real = 0.25,
    inter_edge_alpha::Real = 0.08,
    fig_path::Union{Nothing,String} = nothing,
)
    if !(theme_magnification > 0)
        throw(DomainError(theme_magnification, "theme_magnification must be > 0"))
    end
    if !(0 <= color_transparency <= 1)
        throw(DomainError(color_transparency, "color_transparency must satisfy 0 <= color_transparency <= 1"))
    end
    if fig_size !== nothing
        w, h = fig_size
        if !(w > 0 && h > 0)
            throw(DomainError(fig_size, "fig_size entries must be positive"))
        end
    end

    themed_size = if apply_theme
        apply_paper_theme!(
            double_column = theme_double_column,
            magnification = theme_magnification,
            color_transparency = color_transparency,
        )
    else
        (900, 900)
    end
    size_to_use = fig_size === nothing ? themed_size : fig_size
    fig = CairoMakie.Figure(size = size_to_use)
    ax = CairoMakie.Axis(fig[1, 1])
    _draw_subgraph_colored_node_link!(
        ax,
        A,
        n_subgraphs;
        cluster_radius = cluster_radius,
        local_base = local_base,
        local_scale = local_scale,
        node_size = node_size,
        edge_alpha = edge_alpha,
        inter_edge_alpha = inter_edge_alpha,
    )

    if !isnothing(fig_path)
        CairoMakie.save(fig_path, fig)
    end

    return fig
end

"""
    plot_subgraph_colored_node_link(
        A1::BitMatrix,
        A2::BitMatrix,
        A3::BitMatrix,
        n_subgraphs::Int;
        fig_size::Union{Nothing,Tuple{Int,Int}} = nothing,
        apply_theme::Bool = true,
        theme_magnification::Real = 1.0,
        color_transparency::Float64 = 1.0,
        cluster_radius::Real = 6.0,
        local_base::Real = 1.8,
        local_scale::Real = 0.5,
        node_size::Real = 10,
        edge_alpha::Real = 0.25,
        inter_edge_alpha::Real = 0.08,
        fig_path::Union{Nothing,String} = nothing,
    )

See the main method
`plot_subgraph_colored_node_link(A, n_subgraphs; ...)` for core styling and
layout controls.

# Changes in this overload
- Accepts three adjacency matrices (`A1`, `A2`, `A3`) instead of one.
- Produces a 1×3 panel figure with separator bars.
- Uses `double_column = true` as the theme default when `apply_theme=true`.

# Arguments
- `A1`: Input parameter `A1` used by this method.
- `A2`: Input parameter `A2` used by this method.
- `A3`: Input parameter `A3` used by this method.
- `n_subgraphs`: Input parameter `n_subgraphs` used by this method.

# Keyword Arguments
- `fig_size`: Keyword option `fig_size` controlling this method's behavior.
- `apply_theme`: Keyword option `apply_theme` controlling this method's behavior.
- `theme_magnification`: Keyword option `theme_magnification` controlling this method's behavior.
- `color_transparency`: Keyword option `color_transparency` controlling this method's behavior.
- `cluster_radius`: Keyword option `cluster_radius` controlling this method's behavior.
- `local_base`: Keyword option `local_base` controlling this method's behavior.
- `local_scale`: Keyword option `local_scale` controlling this method's behavior.
- `node_size`: Keyword option `node_size` controlling this method's behavior.
- `edge_alpha`: Keyword option `edge_alpha` controlling this method's behavior.
- `inter_edge_alpha`: Keyword option `inter_edge_alpha` controlling this method's behavior.
- `fig_path`: Path or collection of paths used for loading/saving data.

# Returns
- `result`: Output of `plot_subgraph_colored_node_link` as described in the summary above.

# Throws
- `DomainError`: Raised when visual parameter domains are invalid.
- `DimensionMismatch`: Propagated from `_draw_subgraph_colored_node_link!` when an input matrix is not square.
- `ArgumentError`: Propagated from `_draw_subgraph_colored_node_link!` when an input matrix is not symmetric."""
function plot_subgraph_colored_node_link(
    A1::BitMatrix,
    A2::BitMatrix,
    A3::BitMatrix,
    n_subgraphs::Int;
    fig_size::Union{Nothing,Tuple{Int,Int}} = nothing,
    apply_theme::Bool = true,
    theme_magnification::Real = 1.0,
    color_transparency::Float64 = 1.0,
    cluster_radius::Real = 6.0,
    local_base::Real = 1.8,
    local_scale::Real = 0.5,
    node_size::Real = 10,
    edge_alpha::Real = 0.25,
    inter_edge_alpha::Real = 0.08,
    fig_path::Union{Nothing,String} = nothing,
)
    if !(theme_magnification > 0)
        throw(DomainError(theme_magnification, "theme_magnification must be > 0"))
    end
    if !(0 <= color_transparency <= 1)
        throw(DomainError(color_transparency, "color_transparency must satisfy 0 <= color_transparency <= 1"))
    end
    if fig_size !== nothing
        w, h = fig_size
        if !(w > 0 && h > 0)
            throw(DomainError(fig_size, "fig_size entries must be positive"))
        end
    end

    themed_size = if apply_theme
        apply_paper_theme!(
            double_column = true,
            magnification = theme_magnification,
            color_transparency = color_transparency,
        )
    else
        (1600, 450)
    end
    size_to_use = fig_size === nothing ? themed_size : fig_size

    fig = CairoMakie.Figure(size = size_to_use)
    ax1 = CairoMakie.Axis(fig[1, 1])
    ax2 = CairoMakie.Axis(fig[1, 3])
    ax3 = CairoMakie.Axis(fig[1, 5])

    # Thick vertical separator bars between the three panels.
    CairoMakie.Box(fig[1, 2], color = :black, strokewidth = 0)
    CairoMakie.Box(fig[1, 4], color = :black, strokewidth = 0)
    CairoMakie.colsize!(fig.layout, 2, CairoMakie.Fixed(8))
    CairoMakie.colsize!(fig.layout, 4, CairoMakie.Fixed(8))

    _draw_subgraph_colored_node_link!(
        ax1,
        A1,
        n_subgraphs;
        cluster_radius = cluster_radius,
        local_base = local_base,
        local_scale = local_scale,
        node_size = node_size,
        edge_alpha = edge_alpha,
        inter_edge_alpha = inter_edge_alpha,
    )
    _draw_subgraph_colored_node_link!(
        ax2,
        A2,
        n_subgraphs;
        cluster_radius = cluster_radius,
        local_base = local_base,
        local_scale = local_scale,
        node_size = node_size,
        edge_alpha = edge_alpha,
        inter_edge_alpha = inter_edge_alpha,
    )
    _draw_subgraph_colored_node_link!(
        ax3,
        A3,
        n_subgraphs;
        cluster_radius = cluster_radius,
        local_base = local_base,
        local_scale = local_scale,
        node_size = node_size,
        edge_alpha = edge_alpha,
        inter_edge_alpha = inter_edge_alpha,
    )

    if !isnothing(fig_path)
        CairoMakie.save(fig_path, fig)
    end

    return fig
end
