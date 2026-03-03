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

using Distributed
if nprocs() == 1
    @info "    adding processes $(num_processes)"
    Distributed.addprocs(num_processes; exeflags = "--threads=1")
end

@everywhere import JLD2

const kind = JLD2.jldopen(in_path, "r") do f
    f["meta/config"]["kind"]
end


@info "Running statistics computation with kind=$(kind), in path=$(in_path), output path=$(out_path), number of processes=$(num_processes)"

@everywhere import CausalSets as CS
@everywhere import LinearAlgebra

@everywhere import QuantumGrav as QG

@everywhere using ProgressMeter
@everywhere using Statistics

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

################################################################################
# sparsify histograms
@everywhere function sparse_hist(v)
    d = Dict{Int,Int}()
    for (k, count) in enumerate(v)
        count == 0 && continue
        d[k] = count
    end
    return d
end

@everywhere function ev_summary(ev)
    abs_ev = abs.(ev)
    num_zero_ev = count(abs_ev .<= 1e-10)
    min_abs_nonzero_ev = let nz = abs_ev[abs_ev .> 1e-10]
        isempty(nz) ? NaN : minimum(nz)
    end
    return (
        ev,
        num_zero_ev,
        min_abs_nonzero_ev,
        minimum(ev),
        maximum(ev),
        mean(ev),
        quantile(ev, 0.25),
        quantile(ev, 0.75),
        quantile(ev, 0.5),
    )
end

@everywhere function sym_norm_lap_eigs!(W)
    n = size(W, 1)
    deg = vec(sum(W, dims = 2))
    dinvsqrt = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        di = deg[i]
        dinvsqrt[i] = di > 0.0 ? inv(sqrt(di)) : 0.0
    end

    LinearAlgebra.lmul!(LinearAlgebra.Diagonal(dinvsqrt), W)
    LinearAlgebra.rmul!(W, LinearAlgebra.Diagonal(dinvsqrt))
    W .*= -1.0
    @inbounds for i in 1:n
        W[i, i] += 1.0
    end

    return LinearAlgebra.eigen(LinearAlgebra.Hermitian(W)).values
end

@everywhere function compute(
    i,
    adj,
    link,
    cset;
    r = 0,
    order = 0,
    num_boundary_cuts = 0,
    genus = 0,
    num_layers = 0,
    std = 0,
    segment_ratio = 0,
    segment_angle = 0,
    rotation_angle = 0,
    rel_num_flips = 0,
    rel_size_KR = 0,
    link_probability = 0,
    lattice = 0,
    trans_in = 0,
    trans_out = 0,
    kind = "None",
)
    n = cset.atom_count

    in_deg = sum(adj, dims = 1)[1, :]
    out_deg = sum(adj, dims = 2)[:, 1]

    in_deg_link = sum(link, dims = 1)[1, :]
    out_deg_link = sum(link, dims = 2)[:, 1]

    tdeg = begin
        countmap(in_deg), # Dict{Int,Int}: value → count
        minimum(in_deg),
        maximum(in_deg),
        mean(in_deg),
        quantile(in_deg, 0.25),
        quantile(in_deg, 0.75),
        quantile(in_deg, 0.5),
        countmap(out_deg),
        minimum(out_deg),
        maximum(out_deg),
        mean(out_deg),
        quantile(out_deg, 0.25),
        quantile(out_deg, 0.75),
        quantile(out_deg, 0.5)
    end

    #t_lap = begin
        # Densify once; eigensolvers are dense anyway.
        #adj_f = Float64.(adj)

        # W_sym = A + A^T
        #W_sym = copy(adj_f)
        #W_sym .+= transpose(adj_f)

        # W_aat = A A^T and W_ata = A^T A
        #W_aat = Matrix{Float64}(undef, n, n)
        #W_ata = Matrix{Float64}(undef, n, n)
        #LinearAlgebra.mul!(W_aat, adj_f, transpose(adj_f))
        #LinearAlgebra.mul!(W_ata, transpose(adj_f), adj_f)

        #ev_sym = sym_norm_lap_eigs!(W_sym)
        #ev_aat = sym_norm_lap_eigs!(W_aat)
        #ev_ata = sym_norm_lap_eigs!(W_ata)

        #(
            #ev_summary(ev_sym)...,
            #ev_summary(ev_aat)...,
            #ev_summary(ev_ata)...,
        #)
    #end

    tdeg_link = begin
        countmap(in_deg_link),
        minimum(in_deg_link),
        maximum(in_deg_link),
        mean(in_deg_link),
        quantile(in_deg_link, 0.25),
        quantile(in_deg_link, 0.75),
        quantile(in_deg_link, 0.5),
        countmap(out_deg_link),
        minimum(out_deg_link),
        maximum(out_deg_link),
        mean(out_deg_link),
        quantile(out_deg_link, 0.25),
        quantile(out_deg_link, 0.75),
        quantile(out_deg_link, 0.5)
    end

    t_lap_link = begin
        # Densify once; eigensolvers are dense anyway.
        link_f = Float64.(link)

        # W_sym = A + A^T
        W_sym = copy(link_f)
        W_sym .+= transpose(link_f)

        # W_aat = A A^T and W_ata = A^T A
        #W_aat = Matrix{Float64}(undef, n, n)
        #W_ata = Matrix{Float64}(undef, n, n)
        #LinearAlgebra.mul!(W_aat, link_f, transpose(link_f))
        #LinearAlgebra.mul!(W_ata, transpose(link_f), link_f)

        ev_sym = sym_norm_lap_eigs!(W_sym)
        #ev_aat = sym_norm_lap_eigs!(W_aat)
        #ev_ata = sym_norm_lap_eigs!(W_ata)

        (
            ev_summary(ev_sym)...,
            #ev_summary(ev_aat)...,
            #ev_summary(ev_ata)...,
        )
    end


    t_c = begin
        cardinalities = CS.cardinality_abundances(cset)

        sparse_hist(cardinalities),
        minimum(cardinalities),
        maximum(cardinalities),
        mean(cardinalities),
        quantile(cardinalities, 0.25),
        quantile(cardinalities, 0.75),
        quantile(cardinalities, 0.5)
    end


    t_c2 = begin
        chain_cardinalities_2 = CS.chain_cardinality_abundances(cset, Val(2))
        c2 = reshape(chain_cardinalities_2, :)

        sparse_hist(chain_cardinalities_2),
        minimum(c2),
        maximum(c2),
        mean(c2),
        quantile(c2, 0.25),
        quantile(c2, 0.75),
        quantile(c2, 0.5)
    end


#    t_c3 = begin
#        chain_cardinalities_3 = CS.chain_cardinality_abundances(cset, Val(3))
#        c3 = reshape(chain_cardinalities_3, :)

#        minimum(c3),
#        maximum(c3),
#        mean(c3),
#        quantile(c3, 0.25),
#        quantile(c3, 0.75),
#        quantile(c3, 0.5)
#    end


    t_paths = begin
        sources = findall(in_deg .== 0)
        sinks = findall(out_deg .== 0)

        max_pathlens = [QG.max_pathlen(adj, collect(1:n), s) for s in sources]

        countmap(max_pathlens),
        length(sources),
        length(sinks),
        minimum(max_pathlens),
        maximum(max_pathlens),
        mean(max_pathlens),
        quantile(max_pathlens, 0.25),
        quantile(max_pathlens, 0.75),
        quantile(max_pathlens, 0.5)

    end


    t_paths_link = begin
        sources = findall(in_deg_link .== 0)
        sinks = findall(out_deg_link .== 0)

        max_pathlens = [QG.max_pathlen(link, collect(1:n), s) for s in sources]

        countmap(max_pathlens),
        length(sources),
        length(sinks),
        minimum(max_pathlens),
        maximum(max_pathlens),
        mean(max_pathlens),
        quantile(max_pathlens, 0.25),
        quantile(max_pathlens, 0.75),
        quantile(max_pathlens, 0.5)
    end

    @debug "compute connectivity"
    t_rn = begin
        connectivity = CS.count_relations(cset) / (n * (n - 1) / 2)
    end

    # @debug "compute dimension"
    # t_d = begin
    #     relation_dimension = CS.estimate_relation_dimension(d)
    # end


    in_degree_hist,
    in_degree_min,
    in_degree_max,
    in_degree_mean,
    in_degree_q25,
    in_degree_q75,
    in_degree_median,
    out_degree_hist,
    out_degree_min,
    out_degree_max,
    out_degree_mean,
    out_degree_q25,
    out_degree_q75,
    out_degree_median = tdeg
    
    in_degree_hist_link,
    in_degree_min_link,
    in_degree_max_link,
    in_degree_mean_link,
    in_degree_q25_link,
    in_degree_q75_link,
    in_degree_median_link,
    out_degree_hist_link,
    out_degree_min_link,
    out_degree_max_link,
    out_degree_mean_link,
    out_degree_q25_link,
    out_degree_q75_link,
    out_degree_median_link = tdeg_link

    max_pathlen_hist,
    num_sources,
    num_sinks,
    max_pathlen_min,
    max_pathlen_max,
    max_pathlen_mean,
    max_pathlen_q25,
    max_pathlen_q75,
    max_pathlen_median = t_paths

    max_pathlen_hist_link,
    num_sources_link,
    num_sinks_link,
    max_pathlen_min_link,
    max_pathlen_max_link,
    max_pathlen_mean_link,
    max_pathlen_q25_link,
    max_pathlen_q75_link,
    max_pathlen_median_link = t_paths_link

    #ev_sym,
    #ev_sym_num_zero,
    #ev_sym_min_abs_nonzero,
    #ev_sym_min,
    #ev_sym_max,
    #ev_sym_mean,
    #ev_sym_q25,
    #ev_sym_q75,
    #ev_sym_median,
    #ev_aat,
    #ev_aat_num_zero,
    #ev_aat_min_abs_nonzero,
    #ev_aat_min,
    #ev_aat_max,
    #ev_aat_mean,
    #ev_aat_q25,
    #ev_aat_q75,
    #ev_aat_median,
    #ev_ata,
    #ev_ata_num_zero,
    #ev_ata_min_abs_nonzero,
    #ev_ata_min,
    #ev_ata_max,
    #ev_ata_mean,
    #ev_ata_q25,
    #ev_ata_q75,
    #ev_ata_median = t_lap

    ev_sym_link,
    ev_sym_num_zero_link,
    ev_sym_min_abs_nonzero_link,
    ev_sym_min_link,
    ev_sym_max_link,
    ev_sym_mean_link,
    ev_sym_q25_link,
    ev_sym_q75_link,
    ev_sym_median_link = t_lap_link
    #ev_aat_link,
    #ev_aat_num_zero_link,
    #ev_aat_min_abs_nonzero_link,
    #ev_aat_min_link,
    #ev_aat_max_link,
    #ev_aat_mean_link,
    #ev_aat_q25_link,
    #ev_aat_q75_link,
    #ev_aat_median_link,
    #ev_ata_link,
    #ev_ata_num_zero_link,
    #ev_ata_min_abs_nonzero_link,
    #ev_ata_min_link,
    #ev_ata_max_link,
    #ev_ata_mean_link,
    #ev_ata_q25_link,
    #ev_ata_q75_link,
    #ev_ata_median_link = t_lap_link

    @debug "fetching results rn, cn, d"
    connectivity = t_rn
    # relation_dimension = t_d

    cardinalities_hist,
    cardinalities_min,
    cardinalities_max,
    cardinalities_mean,
    cardinalities_q25,
    cardinalities_q75,
    cardinalities_median = t_c

    chain_cardinalities_2_hist,
    chain_cardinalities_2_min,
    chain_cardinalities_2_max,
    chain_cardinalities_2_mean,
    chain_cardinalities_2_q25,
    chain_cardinalities_2_q75,
    chain_cardinalities_2_median = t_c2


#    chain_cardinalities_3_min,
#    chain_cardinalities_3_max,
#    chain_cardinalities_3_mean,
#    chain_cardinalities_3_q25,
#    chain_cardinalities_3_q75,
#    chain_cardinalities_3_median = t_c3


    d = (
        #
        n = n,
        # indegree
        in_degree_hist = in_degree_hist,
        in_degree_min = in_degree_min,
        in_degree_max = in_degree_max,
        in_degree_mean = in_degree_mean,
        in_degree_q25 = in_degree_q25,
        in_degree_q75 = in_degree_q75,
        in_degree_median = in_degree_median,
        # outdegree
        out_degree_hist = out_degree_hist,
        out_degree_min = out_degree_min,
        out_degree_max = out_degree_max,
        out_degree_mean = out_degree_mean,
        out_degree_q25 = out_degree_q25,
        out_degree_q75 = out_degree_q75,
        out_degree_median = out_degree_median,

        # in degree link
        in_degree_hist_link = in_degree_hist_link,
        in_degree_min_link = in_degree_min_link,
        in_degree_max_link = in_degree_max_link,
        in_degree_mean_link = in_degree_mean_link,
        in_degree_q25_link = in_degree_q25_link,
        in_degree_q75_link = in_degree_q75_link,
        in_degree_median_link = in_degree_median_link,

        # out degree link
        out_degree_hist_link = out_degree_hist_link,
        out_degree_min_link = out_degree_min_link,
        out_degree_max_link = out_degree_max_link,
        out_degree_mean_link = out_degree_mean_link,
        out_degree_q25_link = out_degree_q25_link,
        out_degree_q75_link = out_degree_q75_link,
        out_degree_median_link = out_degree_median_link,

        # pathlens
        max_pathlen_hist = max_pathlen_hist,
        max_pathlen_min = max_pathlen_min,
        max_pathlen_max = max_pathlen_max,
        max_pathlen_mean = max_pathlen_mean,
        max_pathlen_q25 = max_pathlen_q25,
        max_pathlen_q75 = max_pathlen_q75,
        max_pathlen_median = max_pathlen_median,

        # pathlens link
        max_pathlen_hist_link = max_pathlen_hist_link, 
        max_pathlen_min_link = max_pathlen_min_link,
        max_pathlen_max_link = max_pathlen_max_link,
        max_pathlen_mean_link = max_pathlen_mean_link,
        max_pathlen_q25_link = max_pathlen_q25_link,
        max_pathlen_q75_link = max_pathlen_q75_link,
        max_pathlen_median_link = max_pathlen_median_link,

        # sinks/sources
        num_sources = num_sources,
        num_sinks = num_sinks,

        # sinks/sources for link mat
        num_sources_link = num_sources_link,
        num_sinks_link = num_sinks_link,

        # eigenvalues of sym-normalized laplacian for A + A^T
        #ev_sym = ev_sym,
        #ev_sym_num_zero = ev_sym_num_zero,
        #ev_sym_min_abs_nonzero = ev_sym_min_abs_nonzero,
        #ev_sym_min = ev_sym_min,
        #ev_sym_max = ev_sym_max,
        #ev_sym_mean = ev_sym_mean,
        #ev_sym_q25 = ev_sym_q25,
        #ev_sym_q75 = ev_sym_q75,
        #ev_sym_median = ev_sym_median,

        # eigenvalues of sym-normalized laplacian for A A^T
        #ev_aat = ev_aat,
        #ev_aat_num_zero = ev_aat_num_zero,
        #ev_aat_min_abs_nonzero = ev_aat_min_abs_nonzero,
        #ev_aat_min = ev_aat_min,
        #ev_aat_max = ev_aat_max,
        #ev_aat_mean = ev_aat_mean,
        #ev_aat_q25 = ev_aat_q25,
        #ev_aat_q75 = ev_aat_q75,
        #ev_aat_median = ev_aat_median,

        # eigenvalues of sym-normalized laplacian for A^T A
        #ev_ata = ev_ata,
        #ev_ata_num_zero = ev_ata_num_zero,
        #ev_ata_min_abs_nonzero = ev_ata_min_abs_nonzero,
        #ev_ata_min = ev_ata_min,
        #ev_ata_max = ev_ata_max,
        #ev_ata_mean = ev_ata_mean,
        #ev_ata_q25 = ev_ata_q25,
        #ev_ata_q75 = ev_ata_q75,
        #ev_ata_median = ev_ata_median,

        # eigenvalues of sym-normalized laplacian for link + link^T
        ev_sym_link = ev_sym_link,
        ev_sym_num_zero_link = ev_sym_num_zero_link,
        ev_sym_min_abs_nonzero_link = ev_sym_min_abs_nonzero_link,
        ev_sym_min_link = ev_sym_min_link,
        ev_sym_max_link = ev_sym_max_link,
        ev_sym_mean_link = ev_sym_mean_link,
        ev_sym_q25_link = ev_sym_q25_link,
        ev_sym_q75_link = ev_sym_q75_link,
        ev_sym_median_link = ev_sym_median_link,

        # eigenvalues of sym-normalized laplacian for link link^T
        #ev_aat_link = ev_aat_link,
        #ev_aat_num_zero_link = ev_aat_num_zero_link,
        #ev_aat_min_abs_nonzero_link = ev_aat_min_abs_nonzero_link,
        #ev_aat_min_link = ev_aat_min_link,
        #ev_aat_max_link = ev_aat_max_link,
        #ev_aat_mean_link = ev_aat_mean_link,
        #ev_aat_q25_link = ev_aat_q25_link,
        #ev_aat_q75_link = ev_aat_q75_link,
        #ev_aat_median_link = ev_aat_median_link,

        # eigenvalues of sym-normalized laplacian for link^T link
        #ev_ata_link = ev_ata_link,
        #ev_ata_num_zero_link = ev_ata_num_zero_link,
        #ev_ata_min_abs_nonzero_link = ev_ata_min_abs_nonzero_link,
        #ev_ata_min_link = ev_ata_min_link,
        #ev_ata_max_link = ev_ata_max_link,
        #ev_ata_mean_link = ev_ata_mean_link,
        #ev_ata_q25_link = ev_ata_q25_link,
        #ev_ata_q75_link = ev_ata_q75_link,
        #ev_ata_median_link = ev_ata_median_link,

        #
        connectivity = connectivity,
        # relation_dimension = relation_dimension,

        # cardinalities
        cardinalities_hist = cardinalities_hist,
        cardinalities_min = cardinalities_min,
        cardinalities_max = cardinalities_max,
        cardinalities_mean = cardinalities_mean,
        cardinalities_q25 = cardinalities_q25,
        cardinalities_q75 = cardinalities_q75,
        cardinalities_median = cardinalities_median,

        # chain cardinalities 2
        chain_cardinalities_2_hist = chain_cardinalities_2_hist,
        chain_cardinalities_2_min = chain_cardinalities_2_min,
        chain_cardinalities_2_max = chain_cardinalities_2_max,
        chain_cardinalities_2_mean = chain_cardinalities_2_mean,
        chain_cardinalities_2_q25 = chain_cardinalities_2_q25,
        chain_cardinalities_2_q75 = chain_cardinalities_2_q75,
        chain_cardinalities_2_median = chain_cardinalities_2_median,

        # chain_cardinalities_3
        #chain_cardinalities_3_min = chain_cardinalities_3_min,
        #chain_cardinalities_3_max = chain_cardinalities_3_max,
        #chain_cardinalities_3_mean = chain_cardinalities_3_mean,
        #chain_cardinalities_3_q25 = chain_cardinalities_3_q25,
        #chain_cardinalities_3_q75 = chain_cardinalities_3_q75,
        #chain_cardinalities_3_median = chain_cardinalities_3_median,
    )

    if kind == "manifoldlike_simply_connected"
        # @debug "  augmenting manifoldlike data..."
        d2 = (r = r, order = order)
        d = merge(d, d2)
    elseif kind == "manifoldlike_non_simply_connected"
        # @debug "  augmenting manifoldlike data..."
        d2 = (r = r, order = order, num_boundary_cuts = num_boundary_cuts, genus = genus)
        d = merge(d, d2)
    elseif kind == "minkowski_quasicrystal"
        # @debug "  augmenting minkowski quasicrystal data..."
        d2 = (trans_in = trans_in, trans_out = trans_out)
        d = merge(d, d2)
    elseif kind == "destroyed"
        # @debug "  augmenting destroyed data..."
        d2 = (r = r, order = order, rel_num_flips = rel_num_flips)
        d = merge(d, d2)
    elseif kind == "merged"
        # @debug "  augmenting merged data..."
        d2 = (r = r, order = order, rel_size_KR = rel_size_KR, link_probability = link_probability)
        d = merge(d, d2)
    elseif kind=="grid"
        #@debug "  augmenting grid data..."
        d2 = (r = r, order = order, segment_ratio = segment_ratio, segment_angle = segment_angle, rotation_angle = rotation_angle, lattice = lattice)
        d = merge(d, d2)
    elseif kind == "layered"
        # @debug "  augmenting layered data..."
        d2 = (num_layers = num_layers, standard_dev = std)
        d = merge(d, d2)
    end

    return d
end

##############################################################################################
prog = Progress(nbatches; desc = "Computing statistics",)

JLD2.jldopen(out_path, "w") do fout
    fout["meta/batchsize"] = batchsize_in
    fout["meta/nbatches"]  = nbatches
    fout["meta/N"]         = N

    idx = 1
for b = 1:nbatches
        # Eagerly load all batch arrays into memory
        JLD2.jldopen(in_path, "r") do fin
            csets_b = fin["batches/$b/csets"]
            adjs_b  = fin["batches/$b/adjs"]
            links_b = fin["batches/$b/links"]

            # Kind-dependent batch metadata loading
            r_b = order_b = num_boundary_cuts_b = genus_b = 
            rel_num_flips_b = rel_size_KR_b = segment_ratio_b = 
            segment_angle_b = rotation_angle_b = lattice_b = num_layers_b = std_b = 
            trans_in_b = trans_out_b = link_probability_b = nothing

            if kind == "manifoldlike_simply_connected"
                r_b     = fin["batches/$b/r"]
                order_b = fin["batches/$b/order"]

            elseif kind == "manifoldlike_non_simply_connected"
                r_b     = fin["batches/$b/r"]
                order_b = fin["batches/$b/order"]
                num_boundary_cuts_b = fin["batches/$b/num_boundary_cuts"]
                genus_b     = fin["batches/$b/genus"]

            elseif kind == "destroyed"
                r_b             = fin["batches/$b/r"]
                order_b         = fin["batches/$b/order"]
                rel_num_flips_b = fin["batches/$b/rel_num_flips"]

            elseif kind == "merged"
                r_b           = fin["batches/$b/r"]
                order_b       = fin["batches/$b/order"]
                rel_size_KR_b = fin["batches/$b/rel_size_KR"]
                link_probability_b = fin["batches/$b/link_probability"]

            elseif kind == "grid"
                r_b             = fin["batches/$b/r"]
                order_b         = fin["batches/$b/order"]
                segment_ratio_b = fin["batches/$b/segment_ratio"]
                segment_angle_b = fin["batches/$b/segment_angle"]
                rotation_angle_b = fin["batches/$b/rotation_angle"]
                lattice_b       = fin["batches/$b/lattice"]

            elseif kind == "layered"
                num_layers_b = fin["batches/$b/num_layers"]
                std_b        = fin["batches/$b/std"]

            elseif kind == "minkowski_quasicrystal"
                trans_in_b  = fin["batches/$b/trans_in"]
                trans_out_b = fin["batches/$b/trans_out"]
            end


            tmp = pmap(1:length(csets_b)) do i
                if kind == "manifoldlike_simply_connected"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind  = kind,
                        r     = r_b !== nothing         ? r_b[i]         : 0,
                        order = order_b !== nothing     ? order_b[i]     : 0,
                    )
                elseif kind == "manifoldlike_non_simply_connected"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind  = kind,
                        r     = r_b !== nothing         ? r_b[i]         : 0,
                        order = order_b !== nothing     ? order_b[i]     : 0,
                        num_boundary_cuts = num_boundary_cuts_b !== nothing ? num_boundary_cuts_b[i] : 0,
                        genus     = genus_b !== nothing     ? genus_b[i]     : 0,
                    )
                elseif kind == "destroyed"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind          = kind,
                        r             = r_b !== nothing               ? r_b[i]               : 0,
                        order         = order_b !== nothing           ? order_b[i]           : 0,
                        rel_num_flips = rel_num_flips_b !== nothing   ? rel_num_flips_b[i]   : 0,
                    )
                elseif kind == "merged"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind        = kind,
                        r           = r_b !== nothing             ? r_b[i]             : 0,
                        order       = order_b !== nothing         ? order_b[i]         : 0,
                        rel_size_KR = rel_size_KR_b !== nothing   ? rel_size_KR_b[i]   : 0,
                        link_probability = link_probability_b !== nothing ? link_probability_b[i] : 0,
                    )
                elseif kind == "grid"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind           = kind,
                        r              = r_b !== nothing               ? r_b[i]               : 0,
                        order          = order_b !== nothing           ? order_b[i]           : 0,
                        segment_ratio  = segment_ratio_b !== nothing   ? segment_ratio_b[i]   : 0,
                        segment_angle  = segment_angle_b !== nothing   ? segment_angle_b[i]   : 0,
                        rotation_angle = rotation_angle_b !== nothing ? rotation_angle_b[i]  : 0,
                        lattice        = lattice_b !== nothing         ? lattice_b[i]         : 0,
                    )
                elseif kind == "layered"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind       = kind,
                        num_layers = num_layers_b !== nothing ? num_layers_b[i] : 0,
                        std        = std_b !== nothing        ? std_b[i]        : 0,
                    )
                elseif kind == "random"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind       = kind,
                    )
                elseif kind == "minkowski_quasicrystal"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind      = kind,
                        trans_in  = trans_in_b !== nothing  ? trans_in_b[i]  : 0,
                        trans_out = trans_out_b !== nothing ? trans_out_b[i] : 0,
                    )
                elseif kind == "minkowski_sprinkling"
                    compute(
                        i,
                        adjs_b[i],
                        links_b[i],
                        csets_b[i];
                        kind = kind,
                    )
                end
            end
            next!(prog)
            fout["batches/$b"] = tmp
        end
        GC.gc()
    end
end

Distributed.rmprocs(workers())
@info "removed all worker processes"
