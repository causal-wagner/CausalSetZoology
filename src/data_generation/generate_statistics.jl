"""
    compute_statistics(cset, links; kwargs...)

Compute per-causal-set graph/cardinality/spectral summary statistics.

# Arguments
- `cset`: Input causal set.
- `links`: Sparse link graph for `cset`.

# Keyword Arguments
- `kind`: Dataset kind used to attach kind-specific metadata fields.
- `observables`: Optional subset of observable groups to compute. Supported
  symbols are `:ev_sym`, `:ev_antisym`, `:cardinalities`, `:link_degree`,
  `:degree`, `:max_pathlen`, `:max_pathlen_sources`, `:height_profile`,
  `:communicability`, `:ev_sym_arpack`, and `:ev_antisym_arpack`. `nothing`
  computes all.
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
    observables::Union{Vector{Symbol},Nothing} = nothing,
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
    supported_observables = (
        :ev_sym,
        :ev_antisym,
        :cardinalities,
        :link_degree,
        :degree,
        :max_pathlen,
        :max_pathlen_sources,
        :height_profile,
        :communicability,
        :ev_sym_arpack,
        :ev_antisym_arpack,
    )
    selected_observables = if isnothing(observables)
        collect(supported_observables)
    else
        invalid = setdiff(observables, collect(supported_observables))
        isempty(invalid) || throw(
            ArgumentError(
                "unsupported observables=$(invalid); expected subset of $(collect(supported_observables))",
            ),
        )
        unique(observables)
    end
    want_degree = :degree in selected_observables
    want_link_degree = :link_degree in selected_observables
    want_max_pathlen = :max_pathlen in selected_observables
    want_max_pathlen_sources = :max_pathlen_sources in selected_observables
    want_height_profile = :height_profile in selected_observables
    want_ev_sym = :ev_sym in selected_observables
    want_ev_antisym = :ev_antisym in selected_observables
    want_ev_sym_arpack = :ev_sym_arpack in selected_observables
    want_ev_antisym_arpack = :ev_antisym_arpack in selected_observables
    want_communicability = :communicability in selected_observables
    want_cardinalities = :cardinalities in selected_observables
    want_any_pathlen = want_max_pathlen || want_max_pathlen_sources || want_height_profile

    in_deg = out_deg = deg = nothing
    if want_degree
        in_deg, out_deg, deg = CausalSetZoology.degrees(cset)
    end
    in_deg_link = out_deg_link = deg_link = nothing
    if want_link_degree || want_any_pathlen
        in_deg_link, out_deg_link, deg_link = CausalSetZoology.degrees(links)
    end

    tdeg = if want_degree
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
    else
        nothing
    end

    tdeg_link = if want_link_degree
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
    else
        nothing
    end

    t_lap_link = if want_ev_sym
        # Densify once; eigensolvers are dense anyway.
        link_f = Float64.(CausalSetZoology.dense_future_links(links))

        # W_sym = A + A^T
        W_sym = copy(link_f)
        symmetrize_strictly_upper_triangular!(W_sym)

        ev_sym = sym_norm_lap_eigs!(W_sym)

        (
            ev_summary(ev_sym)...,
        )
    else
        nothing
    end

    t_imag_antisym_in_lap_link = if want_ev_antisym
        ev_imag_antisym_in = CausalSetZoology.imag_antisym_in_lap_eigs(links)
        (
            ev_summary(ev_imag_antisym_in)...,
        )
    else
        nothing
    end

    t_lap_arpack_link = if want_ev_sym_arpack
        CausalSetZoology.laplacian_extreme_eigenvalues(links)
    else
        nothing
    end

    t_imag_antisym_in_lap_arpack_link = if want_ev_antisym_arpack
        CausalSetZoology.imag_antisym_in_lap_extreme_eigenvalues(links)
    else
        nothing
    end

    t_comm_link = if want_communicability
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
    else
        nothing
    end


    t_c = if want_cardinalities
        cardinalities = CausalSets.cardinality_abundances(cset)

        sparse_hist(cardinalities),
        minimum(cardinalities),
        maximum(cardinalities),
        Statistics.mean(cardinalities),
        Statistics.quantile(cardinalities, 0.25),
        Statistics.quantile(cardinalities, 0.75),
        Statistics.quantile(cardinalities, 0.5)
    else
        nothing
    end


    sources = sinks = nothing
    if want_any_pathlen
        sources = findall(in_deg_link .== 0)
        sinks = findall(out_deg_link .== 0)
    end

    t_path = if want_max_pathlen
        max_pathlens = [CausalSetZoology.height(links, i) for i in 1:n]
        StatsBase.countmap(max_pathlens),
        minimum(max_pathlens),
        maximum(max_pathlens),
        Statistics.mean(max_pathlens),
        Statistics.quantile(max_pathlens, 0.25),
        Statistics.quantile(max_pathlens, 0.75),
        Statistics.quantile(max_pathlens, 0.5)
    else
        nothing
    end

    t_path_sources = if want_max_pathlen_sources
        max_pathlens_sources = [CausalSetZoology.height(links, s) for s in sources]
        StatsBase.countmap(max_pathlens_sources),
        minimum(max_pathlens_sources),
        maximum(max_pathlens_sources),
        Statistics.mean(max_pathlens_sources),
        Statistics.quantile(max_pathlens_sources, 0.25),
        Statistics.quantile(max_pathlens_sources, 0.75),
        Statistics.quantile(max_pathlens_sources, 0.5)
    else
        nothing
    end

    t_height_profile = if want_height_profile
        heights = CausalSetZoology.height_profile(links)
        StatsBase.countmap(heights),
        minimum(heights),
        maximum(heights),
        Statistics.mean(heights),
        Statistics.quantile(heights, 0.25),
        Statistics.quantile(heights, 0.75),
        Statistics.quantile(heights, 0.5)
    else
        nothing
    end

    @debug "compute connectivity"
    connectivity = CausalSetZoology.connectivity(cset)

    d = (n = n,)
    d = merge(d, (connectivity = connectivity,))

    if want_degree
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
        d = merge(d, (
            in_degree_hist = in_degree_hist,
            in_degree_min = in_degree_min,
            in_degree_max = in_degree_max,
            in_degree_mean = in_degree_mean,
            in_degree_q25 = in_degree_q25,
            in_degree_q75 = in_degree_q75,
            in_degree_median = in_degree_median,
            out_degree_hist = out_degree_hist,
            out_degree_min = out_degree_min,
            out_degree_max = out_degree_max,
            out_degree_mean = out_degree_mean,
            out_degree_q25 = out_degree_q25,
            out_degree_q75 = out_degree_q75,
            out_degree_median = out_degree_median,
            degree_hist = degree_hist,
            degree_min = degree_min,
            degree_max = degree_max,
            degree_mean = degree_mean,
            degree_q25 = degree_q25,
            degree_q75 = degree_q75,
            degree_median = degree_median,
        ))
    end

    if want_link_degree
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
        d = merge(d, (
            in_degree_hist_link = in_degree_hist_link,
            in_degree_min_link = in_degree_min_link,
            in_degree_max_link = in_degree_max_link,
            in_degree_mean_link = in_degree_mean_link,
            in_degree_q25_link = in_degree_q25_link,
            in_degree_q75_link = in_degree_q75_link,
            in_degree_median_link = in_degree_median_link,
            out_degree_hist_link = out_degree_hist_link,
            out_degree_min_link = out_degree_min_link,
            out_degree_max_link = out_degree_max_link,
            out_degree_mean_link = out_degree_mean_link,
            out_degree_q25_link = out_degree_q25_link,
            out_degree_q75_link = out_degree_q75_link,
            out_degree_median_link = out_degree_median_link,
            degree_hist_link = degree_hist_link,
            degree_min_link = degree_min_link,
            degree_max_link = degree_max_link,
            degree_mean_link = degree_mean_link,
            degree_q25_link = degree_q25_link,
            degree_q75_link = degree_q75_link,
            degree_median_link = degree_median_link,
        ))
    end

    if want_max_pathlen
        max_pathlen_hist,
        max_pathlen_min,
        max_pathlen_max,
        max_pathlen_mean,
        max_pathlen_q25,
        max_pathlen_q75,
        max_pathlen_median = t_path
        d = merge(d, (
            max_pathlen_hist = max_pathlen_hist,
            max_pathlen_min = max_pathlen_min,
            max_pathlen_max = max_pathlen_max,
            max_pathlen_mean = max_pathlen_mean,
            max_pathlen_q25 = max_pathlen_q25,
            max_pathlen_q75 = max_pathlen_q75,
            max_pathlen_median = max_pathlen_median,
        ))
    end

    if want_max_pathlen_sources
        max_pathlen_sources_hist,
        max_pathlen_sources_min,
        max_pathlen_sources_max,
        max_pathlen_sources_mean,
        max_pathlen_sources_q25,
        max_pathlen_sources_q75,
        max_pathlen_sources_median = t_path_sources
        d = merge(d, (
            max_pathlen_sources_hist = max_pathlen_sources_hist,
            max_pathlen_sources_min = max_pathlen_sources_min,
            max_pathlen_sources_max = max_pathlen_sources_max,
            max_pathlen_sources_mean = max_pathlen_sources_mean,
            max_pathlen_sources_q25 = max_pathlen_sources_q25,
            max_pathlen_sources_q75 = max_pathlen_sources_q75,
            max_pathlen_sources_median = max_pathlen_sources_median,
        ))
    end

    if want_height_profile
        height_profile_hist,
        height_profile_min,
        height_profile_max,
        height_profile_mean,
        height_profile_q25,
        height_profile_q75,
        height_profile_median = t_height_profile
        d = merge(d, (
            height_profile_hist = height_profile_hist,
            height_profile_min = height_profile_min,
            height_profile_max = height_profile_max,
            height_profile_mean = height_profile_mean,
            height_profile_q25 = height_profile_q25,
            height_profile_q75 = height_profile_q75,
            height_profile_median = height_profile_median,
        ))
    end

    if want_any_pathlen
        d = merge(d, (
            num_sources = length(sources),
            num_sinks = length(sinks),
            num_sources_link = length(sources),
            num_sinks_link = length(sinks),
        ))
    end

    if want_ev_sym
        ev_sym_link,
        ev_sym_num_zero_link,
        ev_sym_min_abs_nonzero_link,
        ev_sym_min_link,
        ev_sym_max_link,
        ev_sym_mean_link,
        ev_sym_q25_link,
        ev_sym_q75_link,
        ev_sym_median_link = t_lap_link
        d = merge(d, (
            ev_sym_link = ev_sym_link,
            ev_sym_num_zero_link = ev_sym_num_zero_link,
            ev_sym_min_abs_nonzero_link = ev_sym_min_abs_nonzero_link,
            ev_sym_min_link = ev_sym_min_link,
            ev_sym_max_link = ev_sym_max_link,
            ev_sym_mean_link = ev_sym_mean_link,
            ev_sym_q25_link = ev_sym_q25_link,
            ev_sym_q75_link = ev_sym_q75_link,
            ev_sym_median_link = ev_sym_median_link,
        ))
    end

    if want_ev_antisym
        ev_imag_antisym_in_link,
        ev_imag_antisym_in_num_zero_link,
        ev_imag_antisym_in_min_abs_nonzero_link,
        ev_imag_antisym_in_min_link,
        ev_imag_antisym_in_max_link,
        ev_imag_antisym_in_mean_link,
        ev_imag_antisym_in_q25_link,
        ev_imag_antisym_in_q75_link,
        ev_imag_antisym_in_median_link = t_imag_antisym_in_lap_link
        d = merge(d, (
            ev_imag_antisym_in_link = ev_imag_antisym_in_link,
            ev_imag_antisym_in_num_zero_link = ev_imag_antisym_in_num_zero_link,
            ev_imag_antisym_in_min_abs_nonzero_link = ev_imag_antisym_in_min_abs_nonzero_link,
            ev_imag_antisym_in_min_link = ev_imag_antisym_in_min_link,
            ev_imag_antisym_in_max_link = ev_imag_antisym_in_max_link,
            ev_imag_antisym_in_mean_link = ev_imag_antisym_in_mean_link,
            ev_imag_antisym_in_q25_link = ev_imag_antisym_in_q25_link,
            ev_imag_antisym_in_q75_link = ev_imag_antisym_in_q75_link,
            ev_imag_antisym_in_median_link = ev_imag_antisym_in_median_link,
        ))
    end

    if want_ev_sym_arpack
        ev_sym_arpack_first_nonzero_link, ev_sym_arpack_last_link = t_lap_arpack_link
        d = merge(d, (
            ev_sym_arpack_first_nonzero_link = ev_sym_arpack_first_nonzero_link,
            ev_sym_arpack_last_link = ev_sym_arpack_last_link,
        ))
    end

    if want_ev_antisym_arpack
        ev_antisym_arpack_first_link, ev_antisym_arpack_min_abs_nonzero_link =
            t_imag_antisym_in_lap_arpack_link
        d = merge(d, (
            ev_antisym_arpack_first_link = ev_antisym_arpack_first_link,
            ev_antisym_arpack_min_abs_nonzero_link = ev_antisym_arpack_min_abs_nonzero_link,
        ))
    end

    if want_communicability
        communicability_link,
        communicability_min_link,
        communicability_max_link,
        communicability_mean_link,
        communicability_q25_link,
        communicability_q75_link,
        communicability_median_link = t_comm_link
        d = merge(d, (
            communicability_link = communicability_link,
            communicability_min_link = communicability_min_link,
            communicability_max_link = communicability_max_link,
            communicability_mean_link = communicability_mean_link,
            communicability_q25_link = communicability_q25_link,
            communicability_q75_link = communicability_q75_link,
            communicability_median_link = communicability_median_link,
        ))
    end

    if want_cardinalities
        cardinalities_hist,
        cardinalities_min,
        cardinalities_max,
        cardinalities_mean,
        cardinalities_q25,
        cardinalities_q75,
        cardinalities_median = t_c
        d = merge(d, (
            cardinalities_hist = cardinalities_hist,
            cardinalities_min = cardinalities_min,
            cardinalities_max = cardinalities_max,
            cardinalities_mean = cardinalities_mean,
            cardinalities_q25 = cardinalities_q25,
            cardinalities_q75 = cardinalities_q75,
            cardinalities_median = cardinalities_median,
        ))
    end

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
    compute_statistics(links; kwargs...)

Compute per-causal-set statistics from a `SparseLinksCauset` alone.

# Arguments
- `links`: Sparse link graph representation of a causal set.

# Keyword Arguments
- `kind`: Dataset kind used to attach kind-specific metadata fields.
- `observables`: Optional subset of observable groups to compute. Supported
  symbols are `:ev_sym`, `:ev_antisym`, `:link_degree`, `:max_pathlen`,
  `:max_pathlen_sources`, `:height_profile`, `:communicability`,
  `:ev_sym_arpack`, and `:ev_antisym_arpack`. `nothing` computes all.
- `r`, `order`, `num_boundary_cuts`, `genus`, `num_layers`, `std`,
  `segment_ratio`, `segment_angle`, `rotation_angle`, `rel_num_flips`,
  `rel_size_KR`, `link_probability`, `lattice`, `trans_in`, `trans_out`:
  Optional metadata values recorded for matching `kind`.

# Returns
- `stats::NamedTuple`: Statistics record containing the requested
  link-computable observables plus `n` and any applicable kind metadata.

# Throws
- `ArgumentError`: If `kind` is unsupported or `observables` contains a
  closure-dependent statistic.
- `DomainError`: If `links` is empty (`atom_count < 1`).
"""
function compute_statistics(
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
    observables::Union{Vector{Symbol},Nothing} = nothing,
)
    n = links.atom_count
    if n < 1
        throw(DomainError(n, "links.atom_count must be >= 1"))
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

    supported_observables = (
        :ev_sym,
        :ev_antisym,
        :link_degree,
        :max_pathlen,
        :max_pathlen_sources,
        :height_profile,
        :communicability,
        :ev_sym_arpack,
        :ev_antisym_arpack,
    )
    selected_observables = if isnothing(observables)
        collect(supported_observables)
    else
        invalid = setdiff(observables, collect(supported_observables))
        isempty(invalid) || throw(
            ArgumentError(
                "unsupported observables=$(invalid); expected subset of $(collect(supported_observables))",
            ),
        )
        unique(observables)
    end
    want_link_degree = :link_degree in selected_observables
    want_max_pathlen = :max_pathlen in selected_observables
    want_max_pathlen_sources = :max_pathlen_sources in selected_observables
    want_height_profile = :height_profile in selected_observables
    want_ev_sym = :ev_sym in selected_observables
    want_ev_antisym = :ev_antisym in selected_observables
    want_ev_sym_arpack = :ev_sym_arpack in selected_observables
    want_ev_antisym_arpack = :ev_antisym_arpack in selected_observables
    want_communicability = :communicability in selected_observables
    want_any_pathlen = want_max_pathlen || want_max_pathlen_sources || want_height_profile

    in_deg_link = out_deg_link = deg_link = nothing
    if want_link_degree || want_any_pathlen
        in_deg_link, out_deg_link, deg_link = CausalSetZoology.degrees(links)
    end

    tdeg_link = if want_link_degree
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
    else
        nothing
    end

    t_lap_link = if want_ev_sym
        link_f = Float64.(CausalSetZoology.dense_future_links(links))
        W_sym = copy(link_f)
        symmetrize_strictly_upper_triangular!(W_sym)
        ev_sym = sym_norm_lap_eigs!(W_sym)
        (ev_summary(ev_sym)...,)
    else
        nothing
    end

    t_imag_antisym_in_lap_link = if want_ev_antisym
        ev_imag_antisym_in = CausalSetZoology.imag_antisym_in_lap_eigs(links)
        (ev_summary(ev_imag_antisym_in)...,)
    else
        nothing
    end

    t_lap_arpack_link = if want_ev_sym_arpack
        CausalSetZoology.laplacian_extreme_eigenvalues(links)
    else
        nothing
    end

    t_imag_antisym_in_lap_arpack_link = if want_ev_antisym_arpack
        CausalSetZoology.imag_antisym_in_lap_extreme_eigenvalues(links)
    else
        nothing
    end

    t_comm_link = if want_communicability
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
    else
        nothing
    end

    sources = sinks = nothing
    if want_any_pathlen
        sources = findall(in_deg_link .== 0)
        sinks = findall(out_deg_link .== 0)
    end

    t_path = if want_max_pathlen
        max_pathlens = [CausalSetZoology.height(links, i) for i in 1:n]
        (
            StatsBase.countmap(max_pathlens),
            minimum(max_pathlens),
            maximum(max_pathlens),
            Statistics.mean(max_pathlens),
            Statistics.quantile(max_pathlens, 0.25),
            Statistics.quantile(max_pathlens, 0.75),
            Statistics.quantile(max_pathlens, 0.5),
        )
    else
        nothing
    end

    t_path_sources = if want_max_pathlen_sources
        max_pathlens_sources = [CausalSetZoology.height(links, s) for s in sources]
        (
            StatsBase.countmap(max_pathlens_sources),
            minimum(max_pathlens_sources),
            maximum(max_pathlens_sources),
            Statistics.mean(max_pathlens_sources),
            Statistics.quantile(max_pathlens_sources, 0.25),
            Statistics.quantile(max_pathlens_sources, 0.75),
            Statistics.quantile(max_pathlens_sources, 0.5),
        )
    else
        nothing
    end

    t_height_profile = if want_height_profile
        heights = CausalSetZoology.height_profile(links)
        (
            StatsBase.countmap(heights),
            minimum(heights),
            maximum(heights),
            Statistics.mean(heights),
            Statistics.quantile(heights, 0.25),
            Statistics.quantile(heights, 0.75),
            Statistics.quantile(heights, 0.5),
        )
    else
        nothing
    end

    d = (n = n,)

    if want_link_degree
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
        d = merge(d, (
            in_degree_hist_link = in_degree_hist_link,
            in_degree_min_link = in_degree_min_link,
            in_degree_max_link = in_degree_max_link,
            in_degree_mean_link = in_degree_mean_link,
            in_degree_q25_link = in_degree_q25_link,
            in_degree_q75_link = in_degree_q75_link,
            in_degree_median_link = in_degree_median_link,
            out_degree_hist_link = out_degree_hist_link,
            out_degree_min_link = out_degree_min_link,
            out_degree_max_link = out_degree_max_link,
            out_degree_mean_link = out_degree_mean_link,
            out_degree_q25_link = out_degree_q25_link,
            out_degree_q75_link = out_degree_q75_link,
            out_degree_median_link = out_degree_median_link,
            degree_hist_link = degree_hist_link,
            degree_min_link = degree_min_link,
            degree_max_link = degree_max_link,
            degree_mean_link = degree_mean_link,
            degree_q25_link = degree_q25_link,
            degree_q75_link = degree_q75_link,
            degree_median_link = degree_median_link,
        ))
    end

    if want_max_pathlen
        max_pathlen_hist,
        max_pathlen_min,
        max_pathlen_max,
        max_pathlen_mean,
        max_pathlen_q25,
        max_pathlen_q75,
        max_pathlen_median = t_path
        d = merge(d, (
            max_pathlen_hist = max_pathlen_hist,
            max_pathlen_min = max_pathlen_min,
            max_pathlen_max = max_pathlen_max,
            max_pathlen_mean = max_pathlen_mean,
            max_pathlen_q25 = max_pathlen_q25,
            max_pathlen_q75 = max_pathlen_q75,
            max_pathlen_median = max_pathlen_median,
        ))
    end

    if want_max_pathlen_sources
        max_pathlen_sources_hist,
        max_pathlen_sources_min,
        max_pathlen_sources_max,
        max_pathlen_sources_mean,
        max_pathlen_sources_q25,
        max_pathlen_sources_q75,
        max_pathlen_sources_median = t_path_sources
        d = merge(d, (
            max_pathlen_sources_hist = max_pathlen_sources_hist,
            max_pathlen_sources_min = max_pathlen_sources_min,
            max_pathlen_sources_max = max_pathlen_sources_max,
            max_pathlen_sources_mean = max_pathlen_sources_mean,
            max_pathlen_sources_q25 = max_pathlen_sources_q25,
            max_pathlen_sources_q75 = max_pathlen_sources_q75,
            max_pathlen_sources_median = max_pathlen_sources_median,
        ))
    end

    if want_height_profile
        height_profile_hist,
        height_profile_min,
        height_profile_max,
        height_profile_mean,
        height_profile_q25,
        height_profile_q75,
        height_profile_median = t_height_profile
        d = merge(d, (
            height_profile_hist = height_profile_hist,
            height_profile_min = height_profile_min,
            height_profile_max = height_profile_max,
            height_profile_mean = height_profile_mean,
            height_profile_q25 = height_profile_q25,
            height_profile_q75 = height_profile_q75,
            height_profile_median = height_profile_median,
        ))
    end

    if want_any_pathlen
        d = merge(d, (
            num_sources = length(sources),
            num_sinks = length(sinks),
            num_sources_link = length(sources),
            num_sinks_link = length(sinks),
        ))
    end

    if want_ev_sym
        ev_sym_link,
        ev_sym_num_zero_link,
        ev_sym_min_abs_nonzero_link,
        ev_sym_min_link,
        ev_sym_max_link,
        ev_sym_mean_link,
        ev_sym_q25_link,
        ev_sym_q75_link,
        ev_sym_median_link = t_lap_link
        d = merge(d, (
            ev_sym_link = ev_sym_link,
            ev_sym_num_zero_link = ev_sym_num_zero_link,
            ev_sym_min_abs_nonzero_link = ev_sym_min_abs_nonzero_link,
            ev_sym_min_link = ev_sym_min_link,
            ev_sym_max_link = ev_sym_max_link,
            ev_sym_mean_link = ev_sym_mean_link,
            ev_sym_q25_link = ev_sym_q25_link,
            ev_sym_q75_link = ev_sym_q75_link,
            ev_sym_median_link = ev_sym_median_link,
        ))
    end

    if want_ev_antisym
        ev_imag_antisym_in_link,
        ev_imag_antisym_in_num_zero_link,
        ev_imag_antisym_in_min_abs_nonzero_link,
        ev_imag_antisym_in_min_link,
        ev_imag_antisym_in_max_link,
        ev_imag_antisym_in_mean_link,
        ev_imag_antisym_in_q25_link,
        ev_imag_antisym_in_q75_link,
        ev_imag_antisym_in_median_link = t_imag_antisym_in_lap_link
        d = merge(d, (
            ev_imag_antisym_in_link = ev_imag_antisym_in_link,
            ev_imag_antisym_in_num_zero_link = ev_imag_antisym_in_num_zero_link,
            ev_imag_antisym_in_min_abs_nonzero_link = ev_imag_antisym_in_min_abs_nonzero_link,
            ev_imag_antisym_in_min_link = ev_imag_antisym_in_min_link,
            ev_imag_antisym_in_max_link = ev_imag_antisym_in_max_link,
            ev_imag_antisym_in_mean_link = ev_imag_antisym_in_mean_link,
            ev_imag_antisym_in_q25_link = ev_imag_antisym_in_q25_link,
            ev_imag_antisym_in_q75_link = ev_imag_antisym_in_q75_link,
            ev_imag_antisym_in_median_link = ev_imag_antisym_in_median_link,
        ))
    end

    if want_ev_sym_arpack
        ev_sym_arpack_first_nonzero_link, ev_sym_arpack_last_link = t_lap_arpack_link
        d = merge(d, (
            ev_sym_arpack_first_nonzero_link = ev_sym_arpack_first_nonzero_link,
            ev_sym_arpack_last_link = ev_sym_arpack_last_link,
        ))
    end

    if want_ev_antisym_arpack
        ev_antisym_arpack_first_link, ev_antisym_arpack_min_abs_nonzero_link =
            t_imag_antisym_in_lap_arpack_link
        d = merge(d, (
            ev_antisym_arpack_first_link = ev_antisym_arpack_first_link,
            ev_antisym_arpack_min_abs_nonzero_link = ev_antisym_arpack_min_abs_nonzero_link,
        ))
    end

    if want_communicability
        communicability_link,
        communicability_min_link,
        communicability_max_link,
        communicability_mean_link,
        communicability_q25_link,
        communicability_q75_link,
        communicability_median_link = t_comm_link
        d = merge(d, (
            communicability_link = communicability_link,
            communicability_min_link = communicability_min_link,
            communicability_max_link = communicability_max_link,
            communicability_mean_link = communicability_mean_link,
            communicability_q25_link = communicability_q25_link,
            communicability_q75_link = communicability_q75_link,
            communicability_median_link = communicability_median_link,
        ))
    end

    if kind == "manifoldlike_simply_connected"
        d = merge(d, (r = r, order = order))
    elseif kind == "manifoldlike_non_simply_connected"
        d = merge(d, (r = r, order = order, num_boundary_cuts = num_boundary_cuts, genus = genus))
    elseif kind == "minkowski_quasicrystal"
        d = merge(d, (trans_in = trans_in, trans_out = trans_out))
    elseif kind == "destroyed"
        d = merge(d, (r = r, order = order, rel_num_flips = rel_num_flips))
    elseif kind == "merged"
        d = merge(d, (r = r, order = order, rel_size_KR = rel_size_KR, link_probability = link_probability))
    elseif kind == "grid"
        d = merge(d, (segment_ratio = segment_ratio, segment_angle = segment_angle, rotation_angle = rotation_angle, lattice = lattice))
    elseif kind == "layered"
        d = merge(d, (num_layers = num_layers, standard_dev = std))
    end

    return d
end

"""
    create_statistics_dataset_and_save(in_path, out_path, kind, batchsize_in, nbatches, N; observables=nothing)

Read a generated dataset and write a statistics dataset with one record per causal set.

# Arguments
- `in_path`: Input dataset `.jld2` path.
- `out_path`: Output statistics `.jld2` path.
- `kind`: Dataset kind determining optional metadata fields per batch.
- `batchsize_in`: Declared batch size stored in output metadata.
- `nbatches`: Number of batches to read from input.
- `N`: Declared sample count stored in output metadata.

# Keyword Arguments
- `observables`: Optional subset of observable groups forwarded to
  `compute_statistics`. `nothing` computes all supported observables.

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
    ;
    observables::Union{Vector{Symbol},Nothing} = nothing,
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
    links_only_dataset = JLD2.jldopen(in_path, "r") do fin
        config = fin["meta/config"]
        get(config, "links_only", false)
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
                links_b = fin["batches/$b/links"]
                batch_group = fin["batches/$b"]
                has_csets = haskey(batch_group, "csets")
                if links_only_dataset && has_csets
                    @warn "Input dataset metadata marks links_only=true, but batch $b contains csets. Using stored csets."
                end
                csets_b = has_csets ? fin["batches/$b/csets"] : nothing
                batch_len = length(links_b)
                if !isnothing(csets_b) && !(length(csets_b) == batch_len)
                    throw(
                        DimensionMismatch(
                            "batch $b has inconsistent lengths: csets=$(length(csets_b)), links=$(batch_len)",
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
                    if !(length(r_b) == batch_len == length(order_b))
                        throw(DimensionMismatch("batch $b metadata lengths must match batch length $(batch_len)"))
                    end

                elseif kind == "manifoldlike_non_simply_connected"
                    r_b     = fin["batches/$b/r"]
                    order_b = fin["batches/$b/order"]
                    num_boundary_cuts_b = fin["batches/$b/num_boundary_cuts"]
                    genus_b     = fin["batches/$b/genus"]
                    if !(length(r_b) == length(order_b) == length(num_boundary_cuts_b) == length(genus_b) == batch_len)
                        throw(DimensionMismatch("batch $b metadata lengths must match batch length $(batch_len)"))
                    end

                elseif kind == "destroyed"
                    r_b             = fin["batches/$b/r"]
                    order_b         = fin["batches/$b/order"]
                    rel_num_flips_b = fin["batches/$b/rel_num_flips"]
                    if !(length(r_b) == length(order_b) == length(rel_num_flips_b) == batch_len)
                        throw(DimensionMismatch("batch $b metadata lengths must match batch length $(batch_len)"))
                    end

                elseif kind == "merged"
                    r_b           = fin["batches/$b/r"]
                    order_b       = fin["batches/$b/order"]
                    rel_size_KR_b = fin["batches/$b/rel_size_KR"]
                    link_probability_b = fin["batches/$b/link_probability"]
                    if !(length(r_b) == length(order_b) == length(rel_size_KR_b) == length(link_probability_b) == batch_len)
                        throw(DimensionMismatch("batch $b metadata lengths must match batch length $(batch_len)"))
                    end

                elseif kind == "grid"
                    segment_ratio_b = fin["batches/$b/segment_ratio"]
                    segment_angle_b = fin["batches/$b/segment_angle"]
                    rotation_angle_b = fin["batches/$b/rotation_angle"]
                    lattice_b       = fin["batches/$b/lattice"]
                    if !(length(segment_ratio_b) == length(segment_angle_b) == length(rotation_angle_b) == length(lattice_b) == batch_len)
                        throw(DimensionMismatch("batch $b metadata lengths must match batch length $(batch_len)"))
                    end

                elseif kind == "layered"
                    num_layers_b = fin["batches/$b/num_layers"]
                    std_b        = fin["batches/$b/std"]
                    if !(length(num_layers_b) == length(std_b) == batch_len)
                        throw(DimensionMismatch("batch $b metadata lengths must match batch length $(batch_len)"))
                    end

                elseif kind == "minkowski_quasicrystal"
                    trans_in_b  = fin["batches/$b/trans_in"]
                    trans_out_b = fin["batches/$b/trans_out"]
                    if !(length(trans_in_b) == length(trans_out_b) == batch_len)
                        throw(DimensionMismatch("batch $b metadata lengths must match batch length $(batch_len)"))
                    end
                end


                tmp = Distributed.pmap(1:batch_len) do i
                    if isnothing(csets_b)
                        if kind == "manifoldlike_simply_connected"
                            compute_statistics(
                                links_b[i];
                                kind  = kind,
                                r     = r_b !== nothing         ? r_b[i]         : 0,
                                order = order_b !== nothing     ? order_b[i]     : 0,
                                observables = observables,
                            )
                        elseif kind == "manifoldlike_non_simply_connected"
                            compute_statistics(
                                links_b[i];
                                kind  = kind,
                                r     = r_b !== nothing         ? r_b[i]         : 0,
                                order = order_b !== nothing     ? order_b[i]     : 0,
                                num_boundary_cuts = num_boundary_cuts_b !== nothing ? num_boundary_cuts_b[i] : 0,
                                genus = genus_b !== nothing     ? genus_b[i]     : 0,
                                observables = observables,
                            )
                        elseif kind == "destroyed"
                            compute_statistics(
                                links_b[i];
                                kind          = kind,
                                r             = r_b !== nothing               ? r_b[i]               : 0,
                                order         = order_b !== nothing           ? order_b[i]           : 0,
                                rel_num_flips = rel_num_flips_b !== nothing   ? rel_num_flips_b[i]   : 0,
                                observables = observables,
                            )
                        elseif kind == "merged"
                            compute_statistics(
                                links_b[i];
                                kind        = kind,
                                r           = r_b !== nothing             ? r_b[i]             : 0,
                                order       = order_b !== nothing         ? order_b[i]         : 0,
                                rel_size_KR = rel_size_KR_b !== nothing   ? rel_size_KR_b[i]   : 0,
                                link_probability = link_probability_b !== nothing ? link_probability_b[i] : 0,
                                observables = observables,
                            )
                        elseif kind == "grid"
                            compute_statistics(
                                links_b[i];
                                kind           = kind,
                                segment_ratio  = segment_ratio_b !== nothing   ? segment_ratio_b[i]   : 0,
                                segment_angle  = segment_angle_b !== nothing   ? segment_angle_b[i]   : 0,
                                rotation_angle = rotation_angle_b !== nothing ? rotation_angle_b[i]  : 0,
                                lattice        = lattice_b !== nothing         ? lattice_b[i]         : 0,
                                observables = observables,
                            )
                        elseif kind == "layered"
                            compute_statistics(
                                links_b[i];
                                kind       = kind,
                                num_layers = num_layers_b !== nothing ? num_layers_b[i] : 0,
                                std        = std_b !== nothing        ? std_b[i]        : 0,
                                observables = observables,
                            )
                        elseif kind == "random"
                            compute_statistics(
                                links_b[i];
                                kind = kind,
                                observables = observables,
                            )
                        elseif kind == "minkowski_quasicrystal"
                            compute_statistics(
                                links_b[i];
                                kind      = kind,
                                trans_in  = trans_in_b !== nothing  ? trans_in_b[i]  : 0,
                                trans_out = trans_out_b !== nothing ? trans_out_b[i] : 0,
                                observables = observables,
                            )
                        elseif kind == "minkowski_sprinkling"
                            compute_statistics(
                                links_b[i];
                                kind = kind,
                                observables = observables,
                            )
                        end
                    elseif kind == "manifoldlike_simply_connected"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind  = kind,
                            r     = r_b !== nothing         ? r_b[i]         : 0,
                            order = order_b !== nothing     ? order_b[i]     : 0,
                            observables = observables,
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
                            observables = observables,
                        )
                    elseif kind == "destroyed"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind          = kind,
                            r             = r_b !== nothing               ? r_b[i]               : 0,
                            order         = order_b !== nothing           ? order_b[i]           : 0,
                            rel_num_flips = rel_num_flips_b !== nothing   ? rel_num_flips_b[i]   : 0,
                            observables = observables,
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
                            observables = observables,
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
                            observables = observables,
                        )
                    elseif kind == "layered"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind       = kind,
                            num_layers = num_layers_b !== nothing ? num_layers_b[i] : 0,
                            std        = std_b !== nothing        ? std_b[i]        : 0,
                            observables = observables,
                        )
                    elseif kind == "random"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind       = kind,
                            observables = observables,
                        )
                    elseif kind == "minkowski_quasicrystal"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind      = kind,
                            trans_in  = trans_in_b !== nothing  ? trans_in_b[i]  : 0,
                            trans_out = trans_out_b !== nothing ? trans_out_b[i] : 0,
                            observables = observables,
                        )
                    elseif kind == "minkowski_sprinkling"
                        compute_statistics(
                            csets_b[i],
                            links_b[i];
                            kind = kind,
                            observables = observables,
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
