args = ARGS
for (i, arg) in enumerate(args)
    if arg == "--help" || arg == "-h"
        println(
            "Usage: julia make_analysis_statistics.jl [--kind <kind>] [--in <input_path>] [--out <output_path>] [--num_processes <number>] [--batchsize <number>]",
        )
        println("Options:")
        println("  --in <input_path>                Path to the input .jld2 file containing dataset information.")
        println("  --out <output_path>              Path to save the resulting .csv file with computed statistics.")
        println("  --num_processes <number>         Number of parallel processes to use for computation.")
        println("  --help, -h                       Show this help message.")
        exit(0)
    end

    if arg == "--in"
        if i + 1 <= length(args)
            global in_path = args[i+1]
        else
            println("Error: --in requires a file path argument.")
            exit(1)
        end
    end

    if arg == "--out"
        if i + 1 <= length(args)
            global out_path = args[i+1]
        else
            println("Error: --out requires a file path argument.")
            exit(1)
        end
    end

    if arg == "--num_processes"
        if i + 1 <= length(args)
            global num_processes = parse(Int, args[i+1])
        else
            println("Error: --num_processes requires an integer argument.")
            exit(1)
        end
    end

end

################################################################################
import Pkg
Pkg.activate(@__DIR__)

if !@isdefined(num_processes)
    num_processes = 1
end

using Distributed
if nprocs() == 1
    @info "    adding processes $(num_processes)"
    Distributed.addprocs(num_processes; exeflags = "--threads=1")
end

@everywhere import JLD2
import CausalSetZoology

const kind = JLD2.jldopen(in_path, "r") do f
    f["meta/config"]["kind"]
end


@info "Running statistics computation with kind=$(kind), in path=$(in_path), output path=$(out_path), number of processes=$(num_processes)"

@everywhere import CausalSets
@everywhere import LinearAlgebra

@everywhere import QuantumGrav as QG

@everywhere using ProgressMeter
@everywhere using Statistics

@everywhere import CausalSetZoology

LinearAlgebra.BLAS.set_num_threads(1)

################################################################################
@everywhere connectivity(adj, size) = count(x -> x > 0.0, adj) / (size * (size - 1) / 2)

################################################################################
@everywhere function countmap(values)
    counts = Dict{eltype(values),Int}()
    for v in values
        counts[v] = get(counts, v, 0) + 1
    end
    return counts
end

################################################################################
@info "loading metadata..."
JLD2.jldopen(in_path, "r") do f
    global batchsize_in = f["meta/batchsize"]
    global nbatches     = f["meta/nbatches"]
    global N            = f["meta/N"]
end
JLD2.jldopen(in_path, "r") do f
    config = f["meta/config"]
    inferred_kind = config["kind"]
    @info "Inferred dataset kind from config" kind=inferred_kind
end
@info "Input file batches" batchsize=batchsize_in nbatches=nbatches N=N

##############################################################################################
CausalSetZoology.create_statistics_dataset_and_save(
    in_path,
    out_path,
    kind,
    batchsize_in,
    nbatches,
    N,
)

Distributed.rmprocs(workers())
@info "removed all worker processes"
