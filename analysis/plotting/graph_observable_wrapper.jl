candidates = [
    "/Volumes/Causal Set Silo/causal_sets/analysis/",
    "/Volumes/Causal Set Silochen/causal_sets/analysis/",
]

root_path = let p = findfirst(isdir, candidates)
    p === nothing && error("No analysis path found")
    candidates[p]
end

function data_paths(file_names::Vector{String})::Vector{String}
    return [joinpath(root_path, file_name) for file_name in file_names]
end

function write_distinguishability_csv(path::AbstractString, rows::Vector{NamedTuple})
    mkpath(dirname(path))
    open(path, "w") do io
        if isempty(rows)
            println(io, "observable")
            return
        end
        cols = collect(keys(first(rows)))
        println(io, join(string.(cols), ","))
        for row in rows
            println(io, join((string(getproperty(row, col)) for col in cols), ","))
        end
    end
    return path
end

function compute_all_observables(
    kind::String;
    size::Int = 2048,

    xlim_cardinalities = (1.0, 300.0),
    ylim_cardinalities = (1e-3, 0.04),
    legendpos_cardinalities = :lb,

    xlim_in_degree = (1.0, 150.0),
    ylim_in_degree = (1e-3, 0.2),
    legendpos_in_degree = :rt,

    xlim_out_degree = (1.0, 250.0),
    ylim_out_degree = (1e-4, 0.1),
    legendpos_out_degree = :rt,

    xlim_connectivity = (1.0, 150.0),
    ylim_connectivity = (1e-3, 0.1),
    legendpos_connectivity = :rt,

    xlim_in_degree_link = (1.0, 18.0),
    ylim_in_degree_link = (1e-3, 0.5),
    legendpos_in_degree_link = :lb,

    xlim_out_degree_link = (1.0, 18.0),
    ylim_out_degree_link = (1e-3, 0.5),
    legendpos_out_degree_link = :lb,

    xlim_connectivity_link = (1.0, 18.0),
    ylim_connectivity_link = (1e-3, 0.5),
    legendpos_connectivity_link = :lb,

    xlim_ev_sym_link = (2.0, 500.0),
    ylim_ev_sym_link = (7e-2, 1.0),
    legendpos_ev_sym_link = :rb,

    xlim_max_pathlen = nothing,
    ylim_max_pathlen = (1e-3, 12.0),
    legendpos_max_pathlen = :lt,

    energy::Bool = true,
    mutual_information::Bool = true,
    mahalanobis::Bool = false,
    total_distinguishability::Bool = false,
    mahalanobis_alpha = 0.05,
    mahalanobis_q = 0.05,
    mahalanobis_symmetric = true,
    mahalanobis_R = 1000,
    mahalanobis_progress = false,
    
    mutual_information_k::Int = 5,
    mutual_information_pca_mode::Symbol = :cutoff,
    mutual_information_pca_dim::Int = 32,
    mutual_information_explained_variance::Real = 0.99,
    mutual_information_eigenvalue_rtol::Real = 1e-6,
    mutual_information_max_per_class::Union{Nothing,Int} = nothing,
    energy_distance::Symbol = :Hellinger,
    verbose::Bool = false,
)   

    if kind == "minkowski_quasicrystal"
        comp_name = "minkowski_sprinkling"
    else
        comp_name = "manifoldlike_simply_connected"
    end
    paths = data_paths(["$(kind)_$(size)_10000/statistics.jld2", "$(comp_name)_$(size)_10000/statistics.jld2"])
    fields = [
        :cardinalities_hist,
        :in_degree_hist,
        :out_degree_hist,
        :degree_hist,
        :in_degree_hist_link,
        :out_degree_hist_link,
        :degree_hist_link,
        :ev_sym_link,
        :max_pathlen_hist,
    ]
    loaded = load_fields_from_paths(paths, fields)

    @info "Loaded $(fields) data for $(kind) and manifoldlike datasets of size $(size)."

    cardinalities_hists = [loaded[i][1] for i in eachindex(paths)]
    in_degree_hists      = [loaded[i][2] for i in eachindex(paths)]
    out_degree_hists     = [loaded[i][3] for i in eachindex(paths)]
    connectivity_hists      = [loaded[i][4] for i in eachindex(paths)]
    in_degree_link_hists    = [loaded[i][5] for i in eachindex(paths)]
    out_degree_link_hists   = [loaded[i][6] for i in eachindex(paths)]
    connectivity_link_hists = [loaded[i][7] for i in eachindex(paths)]
    ev_sym_link             = [loaded[i][8] for i in eachindex(paths)]
    max_pathlen_hists       = [loaded[i][9] for i in eachindex(paths)]

    normalized_cardinalities_hists     = normalize_hists(cardinalities_hists)
    normalized_in_degree_hists         = normalize_hists(in_degree_hists)
    normalized_out_degree_hists        = normalize_hists(out_degree_hists)
    normalized_connectivity_hists      = normalize_hists(connectivity_hists)
    normalized_in_degree_link_hists    = normalize_hists(in_degree_link_hists)
    normalized_out_degree_link_hists   = normalize_hists(out_degree_link_hists)
    normalized_connectivity_link_hists = normalize_hists(connectivity_link_hists)
    normalized_max_pathlen_hists       = normalize_hists(max_pathlen_hists; normalization = 1)

    foreach(
        display,
        [

        plot_and_save_hists(
            normalized_cardinalities_hists,
            fig_path("graph_observables/$(kind)/cardinalities.pdf");
            xlim = xlim_cardinalities,
            ylim = ylim_cardinalities,
            xlabel = L"j", 
            ylabel = L"\mathcal{S}_{j}",
            hist_labels = ["$(kind)", "manifoldlike"], 
            legendpos = legendpos_cardinalities,
        ),

        plot_and_save_hists(
            normalized_in_degree_hists,
            fig_path("graph_observables/$(kind)/in_degree.pdf");
            xlim = xlim_in_degree,
            ylim = ylim_in_degree,
            xlabel = L"j", 
            ylabel = L"P_{\mathrm{in}, j}",
            hist_labels = ["$(kind)", "manifoldlike"], 
            legendpos = legendpos_in_degree,
        ),

        plot_and_save_hists(
            normalized_out_degree_hists,
            fig_path("graph_observables/$(kind)/out_degree.pdf");
            xlim = xlim_out_degree,
            ylim = ylim_out_degree,
            xlabel = L"j", 
            ylabel = L"P_{\mathrm{out}, j}",
            hist_labels = ["$(kind)", "manifoldlike"], 
            legendpos = legendpos_out_degree,
        ),

        plot_and_save_hists(
            normalized_connectivity_hists,
            fig_path("graph_observables/$(kind)/connectivity.pdf");
            xlim = xlim_connectivity,
            ylim = ylim_connectivity,
            xlabel = L"j", 
            ylabel = L"P_j",
            hist_labels = ["$(kind)", "manifoldlike"], 
            legendpos = legendpos_connectivity,
        ),


        plot_and_save_hists(
            normalized_in_degree_link_hists,
            fig_path("graph_observables/$(kind)/in_degree_link.pdf");
            xlim = xlim_in_degree_link,
            ylim = ylim_in_degree_link,
            xlabel = L"j", 
            ylabel = L"P^{\mathrm{link}}_{\mathrm{in}, j}",
            hist_labels = ["$(kind)", "manifoldlike"], 
            legendpos = legendpos_in_degree_link,
            #logscale_x = false,
        ),

        plot_and_save_hists(
            normalized_out_degree_link_hists,
            fig_path("graph_observables/$(kind)/out_degree_link.pdf");
            xlim = xlim_out_degree_link,
            ylim = ylim_out_degree_link,
            xlabel = L"j", 
            ylabel = L"P^{\mathrm{link}}_{\mathrm{out}, j}",
            hist_labels = ["$(kind)", "manifoldlike"], 
            legendpos = legendpos_out_degree_link,
            #logscale_x = false,
        ),

        plot_and_save_hists(
            normalized_connectivity_link_hists,
            fig_path("graph_observables/$(kind)/connectivity_link.pdf");
            xlim = xlim_connectivity_link,
            ylim = ylim_connectivity_link,
            xlabel = L"j", 
            ylabel = L"P^{\mathrm{link}}_j",
            hist_labels = ["$(kind)", "manifoldlike"], 
            legendpos = legendpos_connectivity_link,
            #logscale_x = false,
        ),

        plot_and_save_vectors(
            ev_sym_link, 
            fig_path("graph_observables/$(kind)/ev_sym_link.pdf");
            xlim = xlim_ev_sym_link,
            ylim = ylim_ev_sym_link,
            xlabel = L"j", 
            ylabel = L"\lambda_{j}",
            hist_labels = ["$(kind)", "manifoldlike"],
            legendpos = legendpos_ev_sym_link,
            #logscale_x = false,
        ),

        plot_and_save_hists(
            normalized_max_pathlen_hists,
            fig_path("graph_observables/$(kind)/max_pathlen.pdf");
            xlim = xlim_max_pathlen,
            ylim = ylim_max_pathlen,
            xlabel = L"j", 
            ylabel = L"\mathcal{H}_{j}",
            hist_labels = ["$(kind)", "manifoldlike"], 
            legendpos = legendpos_max_pathlen,
        ),

        ]
    )

    observables = [
        ("cardinalities", normalized_cardinalities_hists),
        ("in_degree", normalized_in_degree_hists),
        ("out_degree", normalized_out_degree_hists),
        ("connectivity", normalized_connectivity_hists),
        ("in_degree_link", normalized_in_degree_link_hists),
        ("out_degree_link", normalized_out_degree_link_hists),
        ("connectivity_link", normalized_connectivity_link_hists),
        ("ev_sym_link", ev_sym_link),
        ("max_pathlen", normalized_max_pathlen_hists),
    ]

    rows = NamedTuple[]
    for (name, data) in observables
        if energy
            D_res = energy_based_histogram_distinguishability(data[1], data[2]; distance = energy_distance, verbose = verbose)
            println("For observable $(name), D = $(D_res.D).")
        end
        if mutual_information
            D_mi_res = distinguishability_mutual_information(
                data[1],
                data[2];
                k = mutual_information_k,
                pca_mode = mutual_information_pca_mode,
                pca_dim = mutual_information_pca_dim,
                explained_variance = mutual_information_explained_variance,
                eigenvalue_rtol = mutual_information_eigenvalue_rtol,
                max_per_class = mutual_information_max_per_class,
                verbose = verbose,
            )
            println("For observable $(name), D_mi = $(D_mi_res.D_mi).")
        end

        if mahalanobis
            m_res = mahalanobis_gap_distinguishability(
                data[1],
                data[2];
                alpha = mahalanobis_alpha,
                q = mahalanobis_q,
                symmetric = mahalanobis_symmetric,
                R = mahalanobis_R,
                progress = mahalanobis_progress,
                verbose = verbose,
            )
            println("For observable $(name), D_mahalanobis = $(m_res.D), $(mahalanobis_symmetric ? "D_mahalanobis_sym = $(m_res.D_sym)," : "") M_obs = $(m_res.M_obs), threshold = $(m_res.threshold).")
        end
        row_entries = Pair{Symbol,Any}[:observable => name]
        energy && push!(row_entries, :D => D_res.D)
        mutual_information && push!(row_entries, :D_mutual_information => D_mi_res.D_mi)
        if mahalanobis
            append!(
                row_entries,
                [
                    :D_mahalanobis => m_res.D,
                    :D_mahalanobis_sym => m_res.D_sym,
                    :M_obs => m_res.M_obs,
                    :distinguishable => m_res.distinguishable,
                    :threshold => m_res.threshold,
                    :z_emp => m_res.z_emp,
                    :M_obs_sym => m_res.M_obs_sym,
                    :M_obs_min => m_res.M_obs_min,
                    :threshold_sym => m_res.threshold_sym,
                    :threshold_max => m_res.threshold_max,
                ],
            )
        end
        push!(rows, (; row_entries...))
    end

    if total_distinguishability
        total_selected = (
            normalized_cardinalities_hists,
            normalized_connectivity_link_hists,
            normalized_max_pathlen_hists,
            ev_sym_link,
        )
        total_row_entries = Pair{Symbol,Any}[:observable => "total_selected"]
        if energy
            total_D_res = total_histogram_distinguishability(total_selected...; distance = energy_distance, verbose = verbose)
            println("For observable total_selected, D = $(total_D_res.D).")
            push!(total_row_entries, :D => total_D_res.D)
        end
        if mutual_information
            total_D_mi_res = total_histogram_mutual_information_distinguishability(
                total_selected...;
                k = mutual_information_k,
                pca_mode = mutual_information_pca_mode,
                pca_dim = mutual_information_pca_dim,
                explained_variance = mutual_information_explained_variance,
                eigenvalue_rtol = mutual_information_eigenvalue_rtol,
                max_per_class = mutual_information_max_per_class,
                verbose = verbose,
            )
            println("For observable total_selected, D_mi = $(total_D_mi_res.D_mi).")
            push!(total_row_entries, :D_mutual_information => total_D_mi_res.D_mi)
        end
        if mahalanobis
            total_vecs_a, total_vecs_b = concatenate_hists(total_selected...)
            total_m_res = mahalanobis_gap_distinguishability(
                total_vecs_a,
                total_vecs_b;
                alpha = mahalanobis_alpha,
                q = mahalanobis_q,
                symmetric = mahalanobis_symmetric,
                R = mahalanobis_R,
                progress = mahalanobis_progress,
                verbose = verbose,
            )
            append!(
                total_row_entries,
                [
                    :D_mahalanobis => total_m_res.D,
                    :D_mahalanobis_sym => total_m_res.D_sym,
                    :M_obs => total_m_res.M_obs,
                    :distinguishable => total_m_res.distinguishable,
                    :threshold => total_m_res.threshold,
                    :z_emp => total_m_res.z_emp,
                    :M_obs_sym => total_m_res.M_obs_sym,
                    :M_obs_min => total_m_res.M_obs_min,
                    :threshold_sym => total_m_res.threshold_sym,
                    :threshold_max => total_m_res.threshold_max,
                ],
            )
        end
        push!(rows, (; total_row_entries...))
    end

    csv_path = fig_path("graph_observables/$(kind)/distinguishability.csv")
    
    write_distinguishability_csv(csv_path, rows)
    return rows
end
