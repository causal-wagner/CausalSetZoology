################################################################################

function _parse_bool_flag(value::AbstractString, name::AbstractString)::Bool
    lower = lowercase(strip(value))
    if lower in ("true", "1", "yes", "on")
        return true
    elseif lower in ("false", "0", "no", "off")
        return false
    end
    println("Error: $(name) must be a boolean (true/false).")
    exit(1)
end

function _load_random_connectivity_distribution(path::AbstractString)::Distributions.Distribution
    isfile(path) || error("Connectivity distribution file not found: $(path)")
    return JLD2.jldopen(path, "r") do f
        return f["distribution"]
    end
end

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
            global num_processes = parse(Int, args[i+1])
        else
            println("Error: --num_processes requires a integer argument.")
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

    if arg == "--links_only"
        if i + 1 <= length(args)
            global links_only = _parse_bool_flag(args[i+1], "--links_only")
        else
            println("Error: --links_only requires a boolean argument.")
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
        println("  --num_processes <number>           Number of workers (default: 1).")
        println("  --D <number>                     Dimensionality of the spacetime (default: 2) -- only supported for Minkowski sprinklings and manifoldlike_simply_connected kinds.")
        println("  --cut_restriction <restriction>  Restricts allowed topological cuts (for kind manifoldlike_non_simply_connected). Can be \"boundary_cuts\" or \"free_cuts\".")
        println("  --link_probability <number>      Fix link probability for merged creation (0.0 to 1.0).")
        println("  --links_only <bool>              Generate and store only sparse links for supported kinds (default: false).")
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

if !@isdefined(batchsize)
    batchsize = 100
end

if !@isdefined(links_only)
    links_only = false
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
push!(info_parts, "links_only=$(links_only)")
@isdefined(batchsize) && push!(info_parts, "batchsize=$(batchsize)")
@isdefined(seed) && push!(info_parts, "seed=$(seed)")
@isdefined(out_path) && push!(info_parts, "output path=$(out_path)")
if !@isdefined(num_processes)
    num_processes = 1    
end
push!(info_parts, "num_processes=$num_processes")

@info "Running dataset creation with $(join(info_parts, ", "))"
if lowercase(get(ENV, "CSZ_DEBUG_DATASET", "0")) in ("1", "true", "yes", "on")
    @info "Dataset debug logging enabled via CSZ_DEBUG_DATASET"
end
const _CSZ_DEBUG_DATASET = lowercase(get(ENV, "CSZ_DEBUG_DATASET", "0")) in ("1", "true", "yes", "on")

if !@isdefined(seed)
    seed = 123456
end



################################################################################

import Pkg
#Pkg.update()
t_pkg_start = time_ns()
Pkg.activate(@__DIR__)
Pkg.instantiate()
if _CSZ_DEBUG_DATASET
    @info "Pkg setup done" ms = round((time_ns() - t_pkg_start) / 1e6; digits = 3)
end

using ProgressMeter
import JLD2
using Distributed
import CausalSetZoology
import CausalSets

if nprocs() - 1 < num_processes
    @info "Starting workers" requested=num_processes existing=(nprocs() - 1)
    t_addprocs_start = time_ns()
    active_proj = Base.active_project()
    exeflags = `--project=$(active_proj) --threads=1`
    addprocs(
        num_processes - (nprocs() - 1);
        exeflags = exeflags,
        env = Dict("CSZ_DEBUG_DATASET" => get(ENV, "CSZ_DEBUG_DATASET", "0")),
    )
    @info "Workers started"
    if _CSZ_DEBUG_DATASET
        @info "addprocs done" ms = round((time_ns() - t_addprocs_start) / 1e6; digits = 3)
    end
end

if _CSZ_DEBUG_DATASET
    @info "Initializing worker modules" workers = workers()
    t_workers_init_start = time_ns()
    @everywhere begin
        import Random
        import Distributions
        import LinearAlgebra
        import JLD2
        import CausalSetZoology
    end
    @info "Worker modules initialized" total_ms = round((time_ns() - t_workers_init_start) / 1e6; digits = 3)
    worker_ping_ms = Dict{Int,Float64}()
    for w in workers()
        t0 = time_ns()
        remotecall_fetch(() -> nothing, w)
        worker_ping_ms[w] = round((time_ns() - t0) / 1e6; digits = 3)
    end
    @info "Worker ping done" per_worker_ms = worker_ping_ms
else
    @everywhere begin
        import Random
        import Distributions
        import LinearAlgebra
        import JLD2
        import CausalSetZoology
    end
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

    @info "Using quasicrystal with $(length(big_crystal[1])) points."
end

rdistr = Distributions.Uniform(2, 8)
orderdistr = Distributions.DiscreteUniform(2, 16)
if generate_cset_size
    ndistr = Distributions.DiscreteUniform(256, 2048)
else    
    ndistr = nothing
end
layers_distr = Distributions.DiscreteUniform(2, 25)
link_probability_distr = Distributions.Uniform(0.0, 1.0)
connectivity_dist_path = joinpath(@__DIR__, "connectivity_dist.jld2")
connectivity_distr = kind == "random" ? _load_random_connectivity_distribution(connectivity_dist_path) : nothing
genus_distr = Distributions.DiscreteUniform(1, 10)
num_boundary_cuts_distr = Distributions.DiscreteUniform(1, 10)
lattice_distr = Distributions.DiscreteUniform(1, 1)
segment_ratio_distr = Distributions.Uniform(.1, 10.)
rotate_angle_distr = Distributions.Uniform(0., 180.)
oblique_angle_distr = Distributions.Uniform(1., 59.)
non_manifoldlikeness_distr = Distributions.DiscreteNonParametric(10 .^range(-4,-1,10),fill(1/10,10))
 # Distributions.truncated(
    #Distributions.LocationScale(
    #    0.0001,
    #    1.0,
    #    Distributions.Exponential((0.5 - 0.0001) / log(100)),
    #),
    #0.0001,
    #0.1,
#)

lattices = ["oblique"]

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
)

if generate_cset_size
    config["cset_size"] = "variable"
    cset_size = nothing
else
    config["cset_size"] = cset_size
end

if @isdefined(D)
    config["dimension"] = D
elseif kind == "minkowski_sprinkling" || kind == "manifoldlike_simply_connected" || kind == "manifoldlike_non_simply_connected"
    @warn "Dimension not specified for kind $(kind); defaulting to D=2."
    config["dimension"] = 2
    D = 2
else
    D = nothing
end

if @isdefined(cut_restriction)
    config["cut_restriction"] = cut_restriction
elseif kind == "manifoldlike_non_simply_connected"
    config["cut_restriction"] = "none"
    cut_restriction = nothing
else
    cut_restriction = nothing
end

if @isdefined(link_probability)
    config["link_probability"] = link_probability
elseif kind == "merged"
    config["link_probability"] = "variable"
    link_probability = nothing
else
    link_probability = nothing
end
config["links_only"] = links_only

nbatches = cld(N, batchsize)

CausalSetZoology.create_dataset_and_save(
    out_path,
    kind,
    num_processes,
    batchsize,
    nbatches,
    N,
    config,
    seed;

    cset_size = cset_size,

    link_probability = link_probability,
    D = D,
    cut_restriction = cut_restriction,
    big_crystal = big_crystal,

    ndistr = ndistr,

    rdistr = rdistr,
    genus_distr = genus_distr,
    num_boundary_cuts_distr = num_boundary_cuts_distr,

    lattice_distr = lattice_distr,
    lattices = lattices,
    segment_ratio_distr = segment_ratio_distr,
    rotate_angle_distr = rotate_angle_distr,
    oblique_angle_distr = oblique_angle_distr,

    non_manifoldlikeness_distr = non_manifoldlikeness_distr,

    layers_distr = layers_distr,
    link_probability_distr = link_probability_distr,

    connectivity_distr = connectivity_distr,

    mink = mink,
    causal_diamond_boundary = causal_diamond_boundary,
    links_only = links_only,
)

@info "Dataset creation complete. Output written to $(out_path)."

if Distributed.nprocs() > 1
    try
        Distributed.rmprocs(Distributed.workers())
        @info "Workers shut down successfully."
    catch err
        @info "Worker shutdown failed." error=err
    end
end
