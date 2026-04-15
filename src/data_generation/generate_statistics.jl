"""
    compute_statistics(cset, links; kwargs...)

Compute per-causal-set graph/cardinality/spectral summary statistics.

# Arguments
- `cset`: Input causal set.
- `links`: Sparse link graph for `cset`.

# Keyword Arguments
- `kind`: Dataset kind used to attach kind-specific metadata fields.
- `r`, `order`, `num_boundary_cuts`, `genus`, `num_layers`, `std`,
  `segment_ratio`, `segment_angle`, `rotation_angle`, `rel_num_flips`,
  `rel_size_KR`, `link_probability`, `lattice`, `trans_in`, `trans_out`:
  Optional metadata values recorded for matching `kind`.

# Returns
- `stats::NamedTuple`: Statistics record written into the output statistics dataset.

# Throws
- `ArgumentError`: If `kind` is unsupported.
- `DimensionMismatch`: If `cset` and `links` have different atom counts.
- `DomainError`: If `cset` is empty (`atom_count < 1`).
"""
function compute_statistics(
    cset::CausalSets.BitArrayCauset,
    links::SparseLinksCauset;
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
    if n < 1
        throw(DomainError(n, "cset.atom_count must be >= 1"))
    end
    if links.atom_count != n
        throw(
            DimensionMismatch(
                "links.atom_count=$(links.atom_count) must match cset.atom_count=$n",
            ),
        )
    end
    supported_kinds = (
        "minkowski_sprinkling",
        "minkowski_quasicrystal",
        "manifoldlike_simply_connected",
        "manifoldlike_non_simply_connected",
        "destroyed",
        "merged",
        "grid",
        "random",
        "layered",
    )
    if !(kind in supported_kinds)
        throw(ArgumentError("unsupported kind=$kind; expected one of $(collect(supported_kinds))"))
    end

    in_deg, out_deg, deg = CausalSetZoology.degrees(cset)
    in_deg_link, out_deg_link, deg_link = CausalSetZoology.degrees(links)

    tdeg = begin
        StatsBase.countmap(in_deg), # Dict{Int,Int}: value → count
        minimum(in_deg),
        maximum(in_deg),
        Statistics.mean(in_deg),
        Statistics.quantile(in_deg, 0.25),
        Statistics.quantile(in_deg, 0.75),
        Statistics.quantile(in_deg, 0.5),
        StatsBase.countmap(out_deg),
        minimum(out_deg),
        maximum(out_deg),
        Statistics.mean(out_deg),
        Statistics.quantile(out_deg, 0.25),
        Statistics.quantile(out_deg, 0.75),
        Statistics.quantile(out_deg, 0.5),
        StatsBase.countmap(deg),
        minimum(deg),
        maximum(deg),
        Statistics.mean(deg),
        Statistics.quantile(deg, 0.25),
        Statistics.quantile(deg, 0.75),
        Statistics.quantile(deg, 0.5)
    end

    tdeg_link = begin
        StatsBase.countmap(in_deg_link),
        minimum(in_deg_link),
        maximum(in_deg_link),
        Statistics.mean(in_deg_link),
        Statistics.quantile(in_deg_link, 0.25),
        Statistics.quantile(in_deg_link, 0.75),
        Statistics.quantile(in_deg_link, 0.5),
        StatsBase.countmap(out_deg_link),
        minimum(out_deg_link),
        maximum(out_deg_link),
        Statistics.mean(out_deg_link),
        Statistics.quantile(out_deg_link, 0.25),
        Statistics.quantile(out_deg_link, 0.75),
        Statistics.quantile(out_deg_link, 0.5),
        StatsBase.countmap(deg_link),
        minimum(deg_link),
        maximum(deg_link),
        Statistics.mean(deg_link),
        Statistics.quantile(deg_link, 0.25),
        Statistics.quantile(deg_link, 0.75),
        Statistics.quantile(deg_link, 0.5)
    end

    t_lap_link = begin
        # Densify once; eigensolvers are dense anyway.
        link_f = Float64.(CausalSetZoology.dense_future_links(links))

        # W_sym = A + A^T
        W_sym = copy(link_f)
        symmetrize_strictly_upper_triangular!(W_sym)

        ev_sym = sym_norm_lap_eigs!(W_sym)

        (
            ev_summary(ev_sym)...,
        )
    end

    t_imag_antisym_in_lap_link = begin
        ev_imag_antisym_in = CausalSetZoology.imag_antisym_in_lap_eigs(links)
        (
            ev_summary(ev_imag_antisym_in)...,
        )
    end

    t_comm_link = begin
        comm = CausalSetZoology.communicability_row_sums(links)
        (
            comm,
            minimum(comm),
            maximum(comm),
            Statistics.mean(comm),
            Statistics.quantile(comm, 0.25),
            Statistics.quantile(comm, 0.75),
            Statistics.quantile(comm, 0.5),
        )
    end


    t_c = begin
        cardinalities = CausalSets.cardinality_abundances(cset)

        sparse_hist(cardinalities),
        minimum(cardinalities),
        maximum(cardinalities),
        Statistics.mean(cardinalities),
        Statistics.quantile(cardinalities, 0.25),
        Statistics.quantile(cardinalities, 0.75),
        Statistics.quantile(cardinalities, 0.5)
    end


    t_path = begin
        sources = findall(in_deg_link .== 0)
        sinks = findall(out_deg_link .== 0)

        max_pathlens = [CausalSetZoology.height(links, s) for s in sources]

        StatsBase.countmap(max_pathlens),
        length(sources),
        length(sinks),
        minimum(max_pathlens),
        maximum(max_pathlens),
        Statistics.mean(max_pathlens),
        Statistics.quantile(max_pathlens, 0.25),
        Statistics.quantile(max_pathlens, 0.75),
        Statistics.quantile(max_pathlens, 0.5)
    end
    num_sources_link = t_path[2]
    num_sinks_link = t_path[3]

    @debug "compute connectivity"
    t_rn = begin
        CausalSetZoology.connectivity(cset)
    end


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
    out_degree_median,
    degree_hist,
    degree_min,
    degree_max,
    degree_mean,
    degree_q25,
    degree_q75,
    degree_median = tdeg
    
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
    out_degree_median_link,
    degree_hist_link,
    degree_min_link,
    degree_max_link,
    degree_mean_link,
    degree_q25_link,
    degree_q75_link,
    degree_median_link = tdeg_link

    max_pathlen_hist,
    num_sources,
    num_sinks,
    max_pathlen_min,
    max_pathlen_max,
    max_pathlen_mean,
    max_pathlen_q25,
    max_pathlen_q75,
    max_pathlen_median = t_path

    ev_sym_link,
    ev_sym_num_zero_link,
    ev_sym_min_abs_nonzero_link,
    ev_sym_min_link,
    ev_sym_max_link,
    ev_sym_mean_link,
    ev_sym_q25_link,
    ev_sym_q75_link,
    ev_sym_median_link = t_lap_link

    ev_imag_antisym_in_link,
    ev_imag_antisym_in_num_zero_link,
    ev_imag_antisym_in_min_abs_nonzero_link,
    ev_imag_antisym_in_min_link,
    ev_imag_antisym_in_max_link,
    ev_imag_antisym_in_mean_link,
    ev_imag_antisym_in_q25_link,
    ev_imag_antisym_in_q75_link,
    ev_imag_antisym_in_median_link = t_imag_antisym_in_lap_link

    communicability_link,
    communicability_min_link,
    communicability_max_link,
    communicability_mean_link,
    communicability_q25_link,
    communicability_q75_link,
    communicability_median_link = t_comm_link

    @debug "fetching results rn, cn, d"
    connectivity = t_rn

    cardinalities_hist,
    cardinalities_min,
    cardinalities_max,
    cardinalities_mean,
    cardinalities_q25,
    cardinalities_q75,
    cardinalities_median = t_c


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
        # full degree
        degree_hist = degree_hist,
        degree_min = degree_min,
        degree_max = degree_max,
        degree_mean = degree_mean,
        degree_q25 = degree_q25,
        degree_q75 = degree_q75,
        degree_median = degree_median,

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
        # full degree link
        degree_hist_link = degree_hist_link,
        degree_min_link = degree_min_link,
        degree_max_link = degree_max_link,
        degree_mean_link = degree_mean_link,
        degree_q25_link = degree_q25_link,
        degree_q75_link = degree_q75_link,
        degree_median_link = degree_median_link,

        # pathlens
        max_pathlen_hist = max_pathlen_hist,
        max_pathlen_min = max_pathlen_min,
        max_pathlen_max = max_pathlen_max,
        max_pathlen_mean = max_pathlen_mean,
        max_pathlen_q25 = max_pathlen_q25,
        max_pathlen_q75 = max_pathlen_q75,
        max_pathlen_median = max_pathlen_median,

        # sinks/sources
        num_sources = num_sources,
        num_sinks = num_sinks,

        # sinks/sources for link mat
        num_sources_link = num_sources_link,
        num_sinks_link = num_sinks_link,

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

        # eigenvalues of imag times antisymmetric part of normalized in-Laplacian
        ev_imag_antisym_in_link = ev_imag_antisym_in_link,
        ev_imag_antisym_in_num_zero_link = ev_imag_antisym_in_num_zero_link,
        ev_imag_antisym_in_min_abs_nonzero_link = ev_imag_antisym_in_min_abs_nonzero_link,
        ev_imag_antisym_in_min_link = ev_imag_antisym_in_min_link,
        ev_imag_antisym_in_max_link = ev_imag_antisym_in_max_link,
        ev_imag_antisym_in_mean_link = ev_imag_antisym_in_mean_link,
        ev_imag_antisym_in_q25_link = ev_imag_antisym_in_q25_link,
        ev_imag_antisym_in_q75_link = ev_imag_antisym_in_q75_link,
        ev_imag_antisym_in_median_link = ev_imag_antisym_in_median_link,

        # row sums of exp(A_link)
        communicability_link = communicability_link,
        communicability_min_link = communicability_min_link,
        communicability_max_link = communicability_max_link,
        communicability_mean_link = communicability_mean_link,
        communicability_q25_link = communicability_q25_link,
        communicability_q75_link = communicability_q75_link,
        communicability_median_link = communicability_median_link,

        #
        connectivity = connectivity,

        # cardinalities
        cardinalities_hist = cardinalities_hist,
        cardinalities_min = cardinalities_min,
        cardinalities_max = cardinalities_max,
        cardinalities_mean = cardinalities_mean,
        cardinalities_q25 = cardinalities_q25,
        cardinalities_q75 = cardinalities_q75,
        cardinalities_median = cardinalities_median,
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
        d2 = (segment_ratio = segment_ratio, segment_angle = segment_angle, rotation_angle = rotation_angle, lattice = lattice)
        d = merge(d, d2)
    elseif kind == "layered"
        # @debug "  augmenting layered data..."
        d2 = (num_layers = num_layers, standard_dev = std)
        d = merge(d, d2)
    end

    return d
end

"""
    create_statistics_dataset_and_save(in_path, out_path, kind, batchsize_in, nbatches, N)

Read a generated dataset and write a statistics dataset with one record per causal set.

# Arguments
- `in_path`: Input dataset `.jld2` path.
- `out_path`: Output statistics `.jld2` path.
- `kind`: Dataset kind determining optional metadata fields per batch.
- `batchsize_in`: Declared batch size stored in output metadata.
- `nbatches`: Number of batches to read from input.
- `N`: Declared sample count stored in output metadata.

# Returns
- `nothing`

# Throws
- `ArgumentError`: If `kind` is unsupported.
- `DomainError`: If `batchsize_in`, `nbatches`, or `N` are invalid.
- `SystemError`: If input/output files cannot be opened.
- `DimensionMismatch`: If batch arrays/metadata lengths are inconsistent.
"""
function create_statistics_dataset_and_save(
    in_path::String,
    out_path::String,
    kind::String,
    batchsize_in::Int,
    nbatches::Int,
    N::Int,
)
    if !(batchsize_in >= 1)
        throw(DomainError(batchsize_in, "batchsize_in must be >= 1"))
    end
    if !(nbatches >= 1)
        throw(DomainError(nbatches, "nbatches must be >= 1"))
    end
    if !(N >= 1)
        throw(DomainError(N, "N must be >= 1"))
    end
    supported_kinds = (
        "minkowski_sprinkling",
        "minkowski_quasicrystal",
        "manifoldlike_simply_connected",
        "manifoldlike_non_simply_connected",
        "destroyed",
        "merged",
        "grid",
        "random",
        "layered",
    )
    if !(kind in supported_kinds)
        throw(ArgumentError("unsupported kind=$kind; expected one of $(collect(supported_kinds))"))
    end

    prog = ProgressMeter.Progress(nbatches; desc = "Computing statistics")

    JLD2.jldopen(out_path, "w") do fout
        fout["meta/batchsize"] = batchsize_in
        fout["meta/nbatches"]  = nbatches
        fout["meta/N"]         = N

        idx = 1
    for b = 1:nbatches
            # Eagerly load all batch arrays into memory
            JLD2.jldopen(in_path, "r") do fin
                csets_b = fin["batches/$b/csets"]
                links_b = fin["batches/$b/links"]
                if !(length(csets_b) == length(links_b))
                    throw(
                        DimensionMismatch(
                            "batch $b has inconsistent lengths: csets=$(length(csets_b)), links=$(length(links_b))",
                        ),
                    )
                end

                # Kind-dependent batch metadata loading
                r_b = order_b = num_boundary_cuts_b = genus_b = 
                rel_num_flips_b = rel_size_KR_b = segment_ratio_b = 
                segment_angle_b = rotation_angle_b = lattice_b = num_layers_b = std_b = 
                trans_in_b = trans_out_b = link_probability_b = nothing

                if kind == "manifoldlike_simply_connected"
                    r_b     = fin["batches/$b/r"]
                    order_b = fin["batches/$b/order"]
                    if !(length(r_b) == length(csets_b) == length(order_b))
                        throw(DimensionMismatch("batch $b metadata lengths must match csets length $(length(csets_b))"))
                    end

                elseif kind == "manifoldlike_non_simply_connected"
                    r_b     = fin["batches/$b/r"]
                    order_b = fin["batches/$b/order"]
                    num_boundary_cuts_b = fin["batches/$b/num_boundary_cuts"]
                    genus_b     = fin["batches/$b/genus"]
                    if !(length(r_b) == length(order_b) == length(num_boundary_cuts_b) == length(genus_b) == length(csets_b))
                        throw(DimensionMismatch("batch $b metadata lengths must match csets length $(length(csets_b))"))
                    end

                elseif kind == "destroyed"
                    r_b             = fin["batches/$b/r"]
                    order_b         = fin["batches/$b/order"]
                    rel_num_flips_b = fin["batches/$b/rel_num_flips"]
                    if !(length(r_b) == length(order_b) == length(rel_num_flips_b) == length(csets_b))
                        throw(DimensionMismatch("batch $b metadata lengths must match csets length $(length(csets_b))"))
                    end

                elseif kind == "merged"
                    r_b           = fin["batches/$b/r"]
                    order_b       = fin["batches/$b/order"]
                    rel_size_KR_b = fin["batches/$b/rel_size_KR"]
                    link_probability_b = fin["batches/$b/link_probability"]
                    if !(length(r_b) == length(order_b) == length(rel_size_KR_b) == length(link_probability_b) == length(csets_b))
                        throw(DimensionMismatch("batch $b metadata lengths must match csets length $(length(csets_b))"))
                    end

                elseif kind == "grid"
                    segment_ratio_b = fin["batches/$b/segment_ratio"]
                    segment_angle_b = fin["batches/$b/segment_angle"]
                    rotation_angle_b = fin["batches/$b/rotation_angle"]
                    lattice_b       = fin["batches/$b/lattice"]
                    if !(length(segment_ratio_b) == length(segment_angle_b) == length(rotation_angle_b) == length(lattice_b) == length(csets_b))
                        throw(DimensionMismatch("batch $b metadata lengths must match csets length $(length(csets_b))"))
                    end

                elseif kind == "layered"
                    num_layers_b = fin["batches/$b/num_layers"]
                    std_b        = fin["batches/$b/std"]
                    if !(length(num_layers_b) == length(std_b) == length(csets_b))
                        throw(DimensionMismatch("batch $b metadata lengths must match csets length $(length(csets_b))"))
                    end

                elseif kind == "minkowski_quasicrystal"
                    trans_in_b  = fin["batches/$b/trans_in"]
                    trans_out_b = fin["batches/$b/trans_out"]
                    if !(length(trans_in_b) == length(trans_out_b) == length(csets_b))
                        throw(DimensionMismatch("batch $b metadata lengths must match csets length $(length(csets_b))"))
                    end
                end


                tmp = Distributed.pmap(1:length(csets_b)) do i
                    if kind == "manifoldlike_simply_connected"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind  = kind,
                            r     = r_b !== nothing         ? r_b[i]         : 0,
                            order = order_b !== nothing     ? order_b[i]     : 0,
                        )
                    elseif kind == "manifoldlike_non_simply_connected"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind  = kind,
                            r     = r_b !== nothing         ? r_b[i]         : 0,
                            order = order_b !== nothing     ? order_b[i]     : 0,
                            num_boundary_cuts = num_boundary_cuts_b !== nothing ? num_boundary_cuts_b[i] : 0,
                            genus     = genus_b !== nothing     ? genus_b[i]     : 0,
                        )
                    elseif kind == "destroyed"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind          = kind,
                            r             = r_b !== nothing               ? r_b[i]               : 0,
                            order         = order_b !== nothing           ? order_b[i]           : 0,
                            rel_num_flips = rel_num_flips_b !== nothing   ? rel_num_flips_b[i]   : 0,
                        )
                    elseif kind == "merged"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind        = kind,
                            r           = r_b !== nothing             ? r_b[i]             : 0,
                            order       = order_b !== nothing         ? order_b[i]         : 0,
                            rel_size_KR = rel_size_KR_b !== nothing   ? rel_size_KR_b[i]   : 0,
                            link_probability = link_probability_b !== nothing ? link_probability_b[i] : 0,
                        )
                    elseif kind == "grid"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind           = kind,
                            segment_ratio  = segment_ratio_b !== nothing   ? segment_ratio_b[i]   : 0,
                            segment_angle  = segment_angle_b !== nothing   ? segment_angle_b[i]   : 0,
                            rotation_angle = rotation_angle_b !== nothing ? rotation_angle_b[i]  : 0,
                            lattice        = lattice_b !== nothing         ? lattice_b[i]         : 0,
                        )
                    elseif kind == "layered"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind       = kind,
                            num_layers = num_layers_b !== nothing ? num_layers_b[i] : 0,
                            std        = std_b !== nothing        ? std_b[i]        : 0,
                        )
                    elseif kind == "random"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind       = kind,
                        )
                    elseif kind == "minkowski_quasicrystal"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind      = kind,
                            trans_in  = trans_in_b !== nothing  ? trans_in_b[i]  : 0,
                            trans_out = trans_out_b !== nothing ? trans_out_b[i] : 0,
                        )
                    elseif kind == "minkowski_sprinkling"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind = kind,
                        )
                    end
                end
                ProgressMeter.next!(prog)
                fout["batches/$b"] = tmp
            end
            GC.gc()
        end
    end
end
