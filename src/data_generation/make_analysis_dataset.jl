################################################################################

args = ARGS
for (i, arg) in enumerate(args)

    if arg == "--out"
        if i + 1 <= length(args)
            global out_path = args[i+1]
        else
            println("Error: --out requires a file path argument.")
            exit(1)
        end
    end

    if arg == "--N"
        if i + 1 <= length(args)
            global N = parse(Int, args[i+1])
        else
            println("Error: --N requires an integer argument.")
            exit(1)
        end
    end

    if arg == "--kind"
        if i + 1 <= length(args)
            global kind = args[i+1]
        else
            println("Error: --kind requires a string argument.")
            exit(1)
        end
    end

    if arg == "--size"
        global cset_size = parse(Int,args[i+1])
    end

    if arg == "--batchsize"
        if i + 1 <= length(args)
            global batchsize = parse(Int, args[i+1])
        else
            println("Error: --batchsize requires an integer argument.")
            exit(1)
        end
    end

    if arg == "--seed"
        if i + 1 <= length(args)
            global seed = parse(Int, args[i+1])
        else
            println("Error: --seed requires an integer argument.")
            exit(1)
        end
    end

    if arg == "--num_processes"
        if i + 1 <= length(args)
            global num_workers = parse(Int, args[i+1])
        else
            println("Error: --num_workers requires a integer argument.")
            exit(1)
        end
    end

    if arg == "--D"
        if i + 1 <= length(args)
            global D = parse(Int, args[i+1])
        else
            println("Error: --D requires an integer argument.")
            exit(1)
        end
    end

    if arg == "--cut_restriction"
        if i + 1 <= length(args)
            global cut_restriction = args[i+1]
        else
            println("Error: --cut_restriction requires a string argument.")
            exit(1)
        end
    end

    if arg == "--link_probability"
        if i + 1 <= length(args)
            global link_probability = parse(Float64, args[i+1])
        else
            println("Error: --link_probability requires a float argument.")
            exit(1)
        end
    end

    if arg == "--help" || arg == "-h"
        println(
            "Usage: julia make_analysis_dataset.jl [--kind <kind>] [--out <output_path>] [--N <number>]",
        )
        println("Options:")
        println(
            "  --kind <kind>                    Kind of dataset to create (random, layered, manifoldlike_simply_connected, manifold_non_simply_connected, destroyed, merged, grid).",
        )
        println("  --out <output_path>              Path to the output file.")
        println("  --N <number>                     Number of samples to generate.")
        println("  --batchsize <number>             Number of causal sets per batch (default: 100).")
        println("  --size <number>                  Causal set size.")
        println("  --seed <number>                  Global RNG seed (default: 123456).")
        println("  --num_workers <number>           Number of workers (default: 1).")
        println("  --D <number>                     Dimensionality of the spacetime (default: 2) -- only supported for Minkowski sprinklings and manifoldlike_simply_connected kinds.")
        println("  --cut_restriction <restriction>  Restricts allowed topological cuts (for kind manifoldlike_non_simply_connected). Can be \"boundary_cuts\" or \"free_cuts\".")
        println("  --link_probability <number>      Fix link probability for merged creation (0.0 to 1.0).")
        println("  --help, -h                       Show this help message.")
        exit(0)
    end
end

# Check for required arguments
if !@isdefined(out_path)
    println("Error: --out is required.")
    exit(1)
end

if !@isdefined(N)
    println("Error: --N is required.")
    exit(1)
end

if !@isdefined(kind)
    println("Error: --kind is required.")
    exit(1)
end

if @isdefined(link_probability)
    if link_probability < 0.0 || link_probability > 1.0
        println("Error: --link_probability must be between 0.0 and 1.0.")
        exit(1)
    end
    if kind != "merged"
        @warn "--link_probability is only used for kind=merged; ignoring for kind=$(kind)"
    end
end

generate_cset_size = !@isdefined(cset_size)

info_parts = String[
    "N=$N",
    generate_cset_size ? "cset size=variable" : "cset size=$(cset_size)",
    "kind=$kind",
]

@isdefined(D) && push!(info_parts, "D=$(D)")
@isdefined(cut_restriction) && push!(info_parts, "cut_restriction=$(cut_restriction)")
@isdefined(link_probability) && push!(info_parts, "link_probability=$(link_probability)")
@isdefined(batchsize) && push!(info_parts, "batchsize=$(batchsize)")
@isdefined(seed) && push!(info_parts, "seed=$(seed)")
@isdefined(out_path) && push!(info_parts, "output path=$(out_path)")
if !@isdefined(num_workers)
    num_workers = 1    
end
push!(info_parts, "num_workers=$num_workers")

@info "Running dataset creation with $(join(info_parts, ", "))"

if !@isdefined(seed)
    seed = 123456
end



################################################################################

import Pkg
#Pkg.update()
Pkg.activate(@__DIR__)
Pkg.instantiate()

using ProgressMeter
import JLD2
using Distributed

if nprocs() - 1 < num_workers
    @info "Starting workers" requested=num_workers existing=(nprocs() - 1)
    addprocs(num_workers - (nprocs() - 1); exeflags="--threads=1")
    @info "Workers started"
end

@everywhere begin
    import QuantumGrav as QG
    import CausalSets
    import Random
    import Distributions
    import LinearAlgebra
    import JLD2
end



################################################################################

@everywhere function transitive_reduction!(mat::AbstractMatrix)
    n = size(mat, 1)
    for i = 1:n
        for j = (i+1):n
            if mat[i, j] == 1
                # If any intermediate node k exists with i → k and k → j, remove i → j
                for k = (i+1):(j-1)
                    if mat[i, k] == 1 && mat[k, j] == 1
                        mat[i, j] = 0 # remove intermediate nodes
                        break
                    end
                end
            end
        end
    end
end

@everywhere const _big_crystal_ref = Ref{Any}(nothing)
@everywhere const _big_crystal_path_ref = Ref{Union{Nothing,String}}(nothing)

@everywhere function _get_big_crystal(path::String)
    if _big_crystal_ref[] === nothing || _big_crystal_path_ref[] != path
        f = JLD2.jldopen(path, "r")
        _big_crystal_ref[] = f["big_set"]
        close(f)
        _big_crystal_path_ref[] = path
    end
    return _big_crystal_ref[]
end

@everywhere function generate_batch(
    b::Int,
    batchsize::Int,
    N::Int,
    kind::AbstractString,
    generate_cset_size::Bool,
    cset_size::Int,
    seed::Int;
    link_probability::Union{Nothing,Float64}=nothing,
    D::Union{Nothing,Int}=nothing,
    cut_restriction::Union{Nothing,String}=nothing,
    big_crystal=nothing,
    big_crystal_path::Union{Nothing,String}=nothing,
)
    rdistr = Distributions.Uniform(2, 8)
    layers_distr = Distributions.DiscreteUniform(2, 25)
    link_probability_distr = Distributions.Uniform(0.0, 1.0)
    connectivity_distr = Distributions.Normal(0.49981532, 0.06963808)
    genus_distr = Distributions.DiscreteUniform(1, 10)
    num_boundary_cuts_distr = Distributions.DiscreteUniform(1, 10)
    lattice_distr = Distributions.DiscreteUniform(1, 5)
    segment_ratio_distr = Distributions.Uniform(.1, 10.)
    rotate_angle_distr = Distributions.Uniform(0., 180.)
    oblique_angle_distr = Distributions.Uniform(1., 59.)
    non_manifoldlikeness_distr = Distributions.Uniform(0.01, .2)
    lattices = ["quadratic", "rectangular", "rhombic", "hexagonal", "oblique"]

    mink = nothing
    causal_diamond_boundary = nothing
    if kind == "minkowski_sprinkling"
        mink = CausalSets.MinkowskiManifold{D === nothing ? 2 : D}()
        causal_diamond_boundary = CausalSets.CausalDiamondBoundary{D === nothing ? 2 : D}(1.)
    end

    start_i = (b - 1) * batchsize + 1
    end_i = min(b * batchsize, N)

    csets_b = CausalSets.BitArrayCauset[]
    adjs_b  = BitMatrix[]
    links_b = BitMatrix[]
    r_b  = Float64[]
    order_b = Int[]
    num_boundary_cuts_b = Int[]
    genus_b = Int[]
    num_layers_b = Int[]
    std_b = Float64[]
    segment_ratio_b = Float64[]
    segment_angle_b = Float64[]
    rotation_angle_b = Float64[]
    rel_num_flips_b = Float64[]
    rel_size_KR_b = Float64[]
    link_probability_b = Float64[]
    lattice_b = String[]
    trans_in_b  = Float64[]
    trans_out_b = Float64[]

    for i in start_i:end_i
        rng = Random.MersenneTwister(seed + i)
        cset_size_i = if generate_cset_size
            ndistr = Distributions.DiscreteUniform(256, 2048)
            rand(rng, ndistr)
        else
            cset_size
        end

        if kind == "minkowski_sprinkling"
            sprinkling = CausalSets.generate_sprinkling(mink, causal_diamond_boundary, cset_size_i)
            cset = CausalSets.BitArrayCauset(mink, sprinkling)

        elseif kind == "minkowski_quasicrystal"
            if big_crystal === nothing
                if big_crystal_path === nothing
                    throw(ArgumentError("big_crystal_path must be provided for kind = \"minkowski_quasicrystal\""))
                end
                big_crystal = _get_big_crystal(big_crystal_path)
            end
            ϵ = sqrt(cset_size_i / length(big_crystal[1]))
            trans_distr = Distributions.Uniform(ϵ, 1 - ϵ)
            αin = rand(rng, trans_distr)
            αout = rand(rng, trans_distr)
            cset = QG.create_Minkowski_quasicrystal_cset(
                cset_size_i,
                (αin, αout);
                crystal = big_crystal,
                exact_size = true,
                deviation_from_mean_size = .1,
                max_iter = 100,
            )
            push!(trans_in_b, αin)
            push!(trans_out_b, αout)

        elseif kind == "manifoldlike_simply_connected"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            cset, _, __ = QG.make_polynomial_manifold_cset(cset_size_i, rng, order, Float64(r); d = D === nothing ? 2 : D)
            push!(r_b, r)
            push!(order_b, order)

        elseif kind == "manifoldlike_non_simply_connected"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            num_boundary_cuts = cut_restriction == "free_cuts" ? 0 : rand(rng, num_boundary_cuts_distr)
            genus = cut_restriction == "boundary_cuts" ? 0 : rand(rng, genus_distr)
            cset, _, _, _  = QG.make_polynomial_manifold_cset_with_nontrivial_topology(cset_size_i, num_boundary_cuts, genus, rng, order, r)
            push!(r_b, r)
            push!(order_b, order)
            push!(num_boundary_cuts_b, num_boundary_cuts)
            push!(genus_b, genus)

        elseif kind == "destroyed"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            num_flips = Int64(ceil(cset_size_i * (cset_size_i - 1) /2 * rand(rng, non_manifoldlikeness_distr)))
            cset, _, _ = QG.destroy_manifold_cset(cset_size_i, num_flips, rng, order, r)
            push!(r_b, r)
            push!(order_b, order)
            push!(rel_num_flips_b, num_flips / (cset_size_i * (cset_size_i - 1) / 2 ))

        elseif kind == "merged"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            link_probability_value = link_probability === nothing ? rand(rng, link_probability_distr) : link_probability
            n2_rel = rand(rng, non_manifoldlikeness_distr)
            cset, _, _ = QG.insert_KR_into_manifoldlike(cset_size_i, order, r, link_probability_value; rng = rng, n2_rel = n2_rel)
            push!(r_b, r)
            push!(order_b, order)
            push!(rel_size_KR_b, n2_rel)
            push!(link_probability_b, link_probability_value)

        elseif kind == "grid"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            lattice = lattices[rand(rng, lattice_distr)]
            segment_ratio = rand(rng, segment_ratio_distr)
            rotate_angle = rand(rng, rotate_angle_distr)
            oblique_angle = rand(rng, oblique_angle_distr)
            cset, _, _ = QG.create_grid_causet_2D_polynomial_manifold(
                cset_size_i,
                lattice,
                rng,
                order,
                r;
                b = segment_ratio,
                gamma_deg = oblique_angle,
                rotate_deg = rotate_angle
            )
            push!(r_b, r)
            push!(order_b, order)
            push!(segment_ratio_b, segment_ratio)
            push!(segment_angle_b, oblique_angle)
            push!(rotation_angle_b, rotate_angle)
            push!(lattice_b, lattice)

        elseif kind == "random"
            connectivity_goal = rand(rng, connectivity_distr)
            abs_tol = 1e-2
            converged = false
            tries = 0
            num_new_goals = 1
            cset_try = nothing
            while num_new_goals < 5
                while !converged && tries ≤ 100
                    cset_try, converged = QG.sample_bitarray_causet_by_connectivity(
                        cset_size_i,
                        connectivity_goal,
                        20,
                        rng;
                        abs_tol = abs_tol,
                    )
                    tries += 1
                end
                if converged
                    break
                else
                    @warn "Skipping causet after 100 failed attempts"
                    continue
                end

                num_new_goals += 1
                if num_new_goals == 5
                    error("Did not converge 5 times after 100 failed attempts. Maybe change connectivity_distr")
                end
            end
            cset = cset_try

        elseif kind == "layered"
            num_layers = rand(rng, layers_distr)
            link_probability = rand(rng, link_probability_distr)
            std_distr = Distributions.Uniform(0., Float64(cset_size_i / (2 * num_layers)))
            std = rand(rng, std_distr)
            cset, n_per_layer = QG.create_random_layered_causet(cset_size_i, num_layers; p = link_probability, rng = rng, standard_deviation = std)
            push!(num_layers_b, num_layers)
            push!(std_b, std)
        end

        adj = transpose(reduce(hcat, cset.future_relations))
        link = deepcopy(adj)
        transitive_reduction!(link)

        push!(csets_b, cset)
        push!(adjs_b, adj)
        push!(links_b, link)
    end

    return (
        csets_b = csets_b,
        adjs_b = adjs_b,
        links_b = links_b,
        r_b = r_b,
        order_b = order_b,
        num_boundary_cuts_b = num_boundary_cuts_b,
        genus_b = genus_b,
        num_layers_b = num_layers_b,
        std_b = std_b,
        segment_ratio_b = segment_ratio_b,
        segment_angle_b = segment_angle_b,
        rotation_angle_b = rotation_angle_b,
        rel_num_flips_b = rel_num_flips_b,
        rel_size_KR_b = rel_size_KR_b,
        link_probability_b = link_probability_b,
        lattice_b = lattice_b,
        trans_in_b = trans_in_b,
        trans_out_b = trans_out_b,
    )
end

################################################################################

big_crystal = nothing
big_crystal_path = nothing
ϵ = nothing
trans_distr = nothing

if kind == "minkowski_quasicrystal"
    big_crystal_path = "/Volumes/Causal Set Silo/causal_sets/crystals/spacetime_quasicrystal_5e8.jld2"
    f = JLD2.jldopen(big_crystal_path, "r")
    big_crystal = f["big_set"]
    close(f)

    ϵ = sqrt(cset_size / length(big_crystal[1]))
    trans_distr = Distributions.Uniform(ϵ, 1 - ϵ)

    @info "Using quasicrystal with $(length(big_crystal[1])) points such that ε = $(ϵ)."
end

rdistr = Distributions.Uniform(2, 8)
orderdistr = Distributions.DiscreteUniform(2, 16)
if generate_cset_size
    ndistr = Distributions.DiscreteUniform(256, 2048)
end
layers_distr = Distributions.DiscreteUniform(2, 25)
link_probability_distr = Distributions.Uniform(0.0, 1.0)
connectivity_distr = Distributions.Normal(0.49981532, 0.06963808)
genus_distr = Distributions.DiscreteUniform(1, 10)
num_boundary_cuts_distr = Distributions.DiscreteUniform(1, 10)
lattice_distr = Distributions.DiscreteUniform(1, 5)
segment_ratio_distr = Distributions.Uniform(.1, 10.)
rotate_angle_distr = Distributions.Uniform(0., 180.)
oblique_angle_distr = Distributions.Uniform(1., 59.)
non_manifoldlikeness_distr = Distributions.Uniform(0.01, .2)

lattices = ["quadratic", "rectangular", "rhombic", "hexagonal", "oblique"]

################################################################################
## Only define Minkowski manifold and boundary for sprinkling, not globally
mink = nothing
causal_diamond_boundary = nothing

if kind == "minkowski_sprinkling"
    mink = CausalSets.MinkowskiManifold{@isdefined(D) ? D : 2}()
    causal_diamond_boundary = CausalSets.CausalDiamondBoundary{@isdefined(D) ? D : 2}(1.)
end

config = Dict(
    "kind" => kind,
    "num_csets" => N,
    "cset_size" => cset_size,
)

if @isdefined(D)
    config["dimension"] = D
end

if @isdefined(cut_restriction)
    config["cut_restriction"] = cut_restriction
end
if @isdefined(link_probability)
    config["link_probability"] = link_probability
end

nbatches = cld(N, batchsize)

JLD2.jldopen(out_path, "w") do fout
    fout["meta/batchsize"] = batchsize
    fout["meta/nbatches"]  = nbatches
    fout["meta/N"]         = N
    fout["meta/config"]    = config

    @info "Assigning batches to workers" nbatches=nbatches num_workers=num_workers
    workers_list = Distributed.workers()
    if num_workers < 1
        error("num_workers must be >= 1")
    end
    if length(workers_list) < num_workers
        error("num_workers=$num_workers but only $(length(workers_list)) workers available")
    end
    workers_list = workers_list[1:num_workers]

    batch_map = [collect(w:num_workers:nbatches) for w in 1:num_workers]
    results = RemoteChannel(() -> Channel{Tuple{Int,Any}}(num_workers))

    @info "Launching worker tasks"
    for (idx, w) in enumerate(workers_list)
        @spawnat w begin
            for b in batch_map[idx]
                data = generate_batch(
                    b,
                    batchsize,
                    N,
                    kind,
                    generate_cset_size,
                    cset_size,
                    seed;
                    link_probability = @isdefined(link_probability) ? link_probability : nothing,
                    D = @isdefined(D) ? D : nothing,
                    cut_restriction = @isdefined(cut_restriction) ? cut_restriction : nothing,
                    big_crystal = big_crystal,
                    big_crystal_path = big_crystal_path,
                )
                put!(results, (b, data))
            end
        end
    end

    p = Progress(N; desc = "Creating causal sets")
    pending = Dict{Int,Any}()
    next_b = 1
    @info "Collecting and writing batches" total_batches=nbatches
    for _ in 1:nbatches
        b, data = take!(results)
        pending[b] = data
        while haskey(pending, next_b)
            data = pending[next_b]
            b = next_b
            delete!(pending, next_b)
            next_b += 1

        fout["batches/$b/csets"] = data.csets_b
        fout["batches/$b/adjs"]  = data.adjs_b
        fout["batches/$b/links"] = data.links_b

        if kind == "minkowski_quasicrystal"
            fout["batches/$b/trans_in"]  = data.trans_in_b
            fout["batches/$b/trans_out"] = data.trans_out_b
        end

        if kind == "manifoldlike_simply_connected"
            fout["batches/$b/r"] = data.r_b
            fout["batches/$b/order"] = data.order_b
        end

        if kind == "manifoldlike_non_simply_connected"
            fout["batches/$b/r"] = data.r_b
            fout["batches/$b/order"] = data.order_b
            fout["batches/$b/num_boundary_cuts"] = data.num_boundary_cuts_b
            fout["batches/$b/genus"] = data.genus_b
        end

        if kind == "destroyed"
            fout["batches/$b/r"] = data.r_b
            fout["batches/$b/order"] = data.order_b
            fout["batches/$b/rel_num_flips"] = data.rel_num_flips_b
        end

        if kind == "merged"
            fout["batches/$b/r"] = data.r_b
            fout["batches/$b/order"] = data.order_b
            fout["batches/$b/rel_size_KR"] = data.rel_size_KR_b
            fout["batches/$b/link_probability"] = data.link_probability_b
        end

        if kind == "grid"
            fout["batches/$b/r"] = data.r_b
            fout["batches/$b/order"] = data.order_b
            fout["batches/$b/segment_ratio"] = data.segment_ratio_b
            fout["batches/$b/segment_angle"] = data.segment_angle_b
            fout["batches/$b/rotation_angle"] = data.rotation_angle_b
            fout["batches/$b/lattice"] = data.lattice_b
        end

        if kind == "layered"
            fout["batches/$b/num_layers"] = data.num_layers_b
            fout["batches/$b/std"] = data.std_b
        end

            next!(p; step=length(data.csets_b))
        end
    end
end
@info "Dataset creation complete. Output written to $(out_path)."

if Distributed.nprocs() > 1
    try
        Distributed.rmprocs(Distributed.workers())
        @info "Workers shut down successfully."
    catch err
        @info "Worker shutdown failed." error=err
    end
end
