"""
    SparseLinksCauset

Sparse causal set representation that stores only link relations
(transitive reduction), not full transitive closure.
"""
struct SparseLinksCauset <: CausalSets.AbstractCauset
    atom_count::Int64
    future_links::Vector{Vector{Int32}} # future link neighbors
    past_links::Vector{Vector{Int32}}   # past link neighbors
end

"""
    SparseLinksCauset(manifold, sprinkling)

Create a `SparseLinksCauset` from a manifold sprinkling. Conversion computes
the transitive reduction automatically.
"""
function SparseLinksCauset(
    manifold::CausalSets.AbstractManifold{N},
    sprinkling::Vector{CausalSets.Coordinates{N}},
)::SparseLinksCauset where {N}
    return convert(SparseLinksCauset, CausalSets.ManifoldCauset(manifold, sprinkling))
end

function SparseLinksCauset(causet::CausalSets.AbstractCauset)::SparseLinksCauset
    return convert(SparseLinksCauset, causet)
end

"""
    Base.convert(::Type{SparseLinksCauset}, causet)

Convert any `CausalSets.AbstractCauset` to `SparseLinksCauset`, computing
transitive reduction (links) during conversion.

Conversion uses `CausalSets.in_past_of` directly and does not materialize an
intermediate `BitArrayCauset`.
"""
function Base.convert(
    ::Type{SparseLinksCauset},
    causet::CausalSets.AbstractCauset,
)::SparseLinksCauset
    atom_count = length(causet)

    future_links = [Int32[] for _ in 1:atom_count]
    past_links = [Int32[] for _ in 1:atom_count]
    tls_future = [Int32[] for _ in 1:Threads.maxthreadid()]

    # Thread-safe first pass: each iteration only writes to future_links[i].
    # Candidate caching avoids re-checking in_past_of(causet, i, j) for every k.
    Threads.@threads :dynamic for i in 1:atom_count
        future_i = tls_future[Threads.threadid()]
        empty!(future_i)

        @inbounds for k in (i + 1):atom_count
            CausalSets.in_past_of(causet, i, k) || continue
            push!(future_i, Int32(k))
        end

        row = future_links[i]
        @inbounds for k32 in future_i
            k = Int(k32)
            is_link = true
            for j32 in future_i
                j = Int(j32)
                j >= k && break
                if CausalSets.in_past_of(causet, j, k)
                    is_link = false
                    break
                end
            end
            is_link && push!(row, k32)
        end
    end

    # Build past links from future links.
    @inbounds for i in 1:atom_count
        for k in future_links[i]
            push!(past_links[Int(k)], Int32(i))
        end
    end

    return SparseLinksCauset(atom_count, future_links, past_links)
end

function _causet_from_manifold_and_sprinkling(
    manifold::CausalSets.AbstractManifold{N},
    sprinkling::Vector{CausalSets.Coordinates{N}};
    links::Bool = false,
)::Union{CausalSets.BitArrayCauset,SparseLinksCauset} where {N}
    if links
        return SparseLinksCauset(manifold, sprinkling)
    end
    return CausalSets.BitArrayCauset(manifold, sprinkling)
end

"""
    create_Minkowski_quasicrystal_cset(N, center; ..., links=false)

Create a Minkowski quasicrystal causal set.

This mirrors `QuantumGrav.create_Minkowski_quasicrystal_cset`, but switches the
final representation via `links`: `BitArrayCauset` for `links=false` and
`SparseLinksCauset` for `links=true`.
"""
function create_Minkowski_quasicrystal_cset(
    N::Int64,
    center::NTuple{2,Float64};
    ρ::Union{Float64,Nothing} = nothing,
    crystal::Union{Tuple{Vector{Float64},Vector{Float64}},Nothing} = nothing,
    exact_size::Bool = true,
    deviation_from_mean_size::Float64 = 0.1,
    max_iter::Int64 = 100,
    links::Bool = false,
)::Union{CausalSets.BitArrayCauset,SparseLinksCauset}
    if ρ === nothing && crystal === nothing
        error("Either ρ or crystal must be provided")
    end

    point_set = QuantumGrav.translate_sub_spacetime_crystal(
        N,
        center;
        ρ = ρ,
        crystal = crystal,
        exact_size = exact_size,
        deviation_from_mean_size = deviation_from_mean_size,
        max_iter = max_iter,
    )

    cartesian_points = Vector{CausalSets.Coordinates{2}}([
        ((αin + αout) / 2, (αout - αin) / 2) for (αin, αout) in point_set
    ])
    sort!(cartesian_points, by = p -> p[1])

    return _causet_from_manifold_and_sprinkling(
        CausalSets.MinkowskiManifold{2}(),
        cartesian_points;
        links = links,
    )
end

"""
    make_polynomial_manifold_cset(npoints, rng, order, r; ..., links=false)

Create a simply connected polynomial-manifold causal set.

This mirrors `QuantumGrav.make_polynomial_manifold_cset`, but switches the
first return value via `links`: `BitArrayCauset` for `links=false` and
`SparseLinksCauset` for `links=true`. The sprinkling and coefficient outputs
are unchanged.
"""
function make_polynomial_manifold_cset(
    npoints::Int64,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    d::Int64 = 2,
    type::Type{T} = Float32,
    links::Bool = false,
)::Tuple{
    Union{CausalSets.BitArrayCauset,SparseLinksCauset},
    Vector{CausalSets.Coordinates{d}},
    Array{T,d},
} where {T<:Number}
    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end
    if order < 0
        throw(ArgumentError("order must be greater than -1, got $order"))
    end
    if r <= 1
        throw(
            ArgumentError(
                "r must be greater than 1 for exponential convergence of the Chebyshev series, got $r",
            ),
        )
    end
    if d < 1
        throw(ArgumentError("dimension d must be at least 1, got $d"))
    end

    chebyshev_coefs = zeros(Float64, ntuple(_ -> order + 1, d))
    for I in CartesianIndices(chebyshev_coefs)
        chebyshev_coefs[I] = r^(-sum(Tuple(I))) * randn(rng)
    end

    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order)
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)
    polym = CausalSets.PolynomialManifold{d}(squaretaylorcoefs)
    boundary = CausalSets.BoxBoundary{d}((ntuple(_ -> -1.0, d), ntuple(_ -> 1.0, d)))
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints; rng = rng)
    cset = _causet_from_manifold_and_sprinkling(polym, sprinkling; links = links)

    return cset, sprinkling, type.(chebyshev_coefs)
end

"""
    make_polynomial_manifold_cset_with_nontrivial_topology(npoints, n_vertical_cuts, genus, rng, order, r; ..., links=false)

Create a polynomial-manifold causal set with nontrivial topology.

This mirrors `QuantumGrav.make_polynomial_manifold_cset_with_nontrivial_topology`,
but switches the first return value via `links`: `BitArrayCauset` for
`links=false` and `SparseLinksCauset` for `links=true`. The remaining return
values are unchanged.
"""
function make_polynomial_manifold_cset_with_nontrivial_topology(
    npoints::Int64,
    n_vertical_cuts::Int64,
    genus::Int64,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    d::Int64 = 2,
    tolerance::Float64 = 1e-12,
    type::Type{T} = Float32,
    links::Bool = false,
)::Tuple{
    Union{CausalSets.BitArrayCauset,SparseLinksCauset},
    Vector{CausalSets.Coordinates{d}},
    Tuple{
        Vector{CausalSets.Coordinates{d}},
        Vector{Tuple{CausalSets.Coordinates{d},CausalSets.Coordinates{d}}},
    },
    Matrix{T},
} where {T<:Number}
    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end
    if n_vertical_cuts < 0
        throw(ArgumentError("n_vertical_cuts must be larger than 0, is $n_vertical_cuts"))
    end
    if genus < 0
        throw(ArgumentError("n_finite_cuts must be larger than 0, is $genus"))
    end
    if order <= -1
        throw(ArgumentError("order must be greater than -1, got $order"))
    end
    if r <= 1
        throw(
            ArgumentError(
                "r must be greater than 1 for exponential convergence of the Chebyshev series, got $r",
            ),
        )
    end
    if d != 2
        throw(ArgumentError("Currently, only 2D is supported, got $d"))
    end
    if tolerance <= 0
        throw(ArgumentError("tolerance must be > 0, got $tolerance"))
    end

    chebyshev_coefs = zeros(Float64, order + 1, order + 1)
    for i = 1:order
        for j = 1:order
            chebyshev_coefs[i, j] = r^(-i - j) * Random.randn(rng)
        end
    end

    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order)
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)
    polym = CausalSets.PolynomialManifold{d}(squaretaylorcoefs)
    boundary = CausalSets.BoxBoundary{d}(((-1.0, -1.0), (1.0, 1.0)))
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints; rng = rng)
    branch_point_info =
        QuantumGrav.generate_random_branch_points(n_vertical_cuts; genus = genus, tolerance = tolerance)
    branched_sprinkling =
        QuantumGrav.filter_sprinkling_near_cuts(sprinkling, branch_point_info; tolerance = tolerance)
    branched_cset = QuantumGrav.BranchedManifoldCauset(polym, branch_point_info, branched_sprinkling)
    cset = links ? SparseLinksCauset(branched_cset) : CausalSets.BitArrayCauset(branched_cset; tolerance = tolerance)

    return cset, branched_sprinkling, branch_point_info, type.(chebyshev_coefs)
end

"""
    create_grid_causet_in_boundary_2D(size, lattice, boundary, manifold; ..., links=false)

Create a 2D grid causal set inside `boundary`.

This mirrors `QuantumGrav.create_grid_causet_in_boundary_2D`, but switches the
first return value via `links`: `BitArrayCauset` for `links=false` and
`SparseLinksCauset` for `links=true`. The convergence flag and coordinate
matrix are unchanged.
"""
function create_grid_causet_in_boundary_2D(
    size::Int64,
    lattice::AbstractString,
    boundary::CausalSets.AbstractBoundary{2},
    manifold::CausalSets.AbstractManifold{2};
    type::Type{T} = Float32,
    a::Float64 = 1.0,
    b::Float64 = 0.5,
    gamma_deg::Float64 = 60.0,
    rotate_deg = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    shell_thickness::Union{Nothing,Float64} = nothing,
    links::Bool = false,
)::Tuple{Union{CausalSets.BitArrayCauset,SparseLinksCauset},Bool,Matrix{T}} where {T<:Number}
    size ≥ 1 || throw(ArgumentError("size must be ≥ 1"))

    if boundary isa CausalSets.BoxBoundary{2}
        box = boundary.edges
        grid = QuantumGrav.generate_grid_2d_in_box(
            size,
            lattice,
            box;
            a = a,
            b = b,
            gamma_deg = gamma_deg,
            rotate_deg = rotate_deg,
            rng = rng,
            shell_thickness = shell_thickness,
        )
    elseif boundary isa CausalSets.CausalDiamondBoundary{2}
        Tdur = boundary.duration
        box = ((0.0, 0.0), (Tdur, Tdur))
        uv_grid = QuantumGrav.generate_grid_2d_in_box(
            size,
            lattice,
            box;
            a = a,
            b = b,
            gamma_deg = gamma_deg,
            rotate_deg = rotate_deg,
            rng = rng,
            shell_thickness = shell_thickness,
        )
        grid = map(uv_grid) do (u, v)
            ((u + v) / 2, (v - u) / 2)
        end
    else
        throw(ArgumentError("Unsupported boundary type: $(typeof(boundary))"))
    end

    pseudosprinkling = QuantumGrav.sort_grid_by_time_from_manifold(manifold, grid)
    cset = _causet_from_manifold_and_sprinkling(manifold, pseudosprinkling; links = links)

    return cset, true, type.(stack(collect.(pseudosprinkling), dims = 1))
end
