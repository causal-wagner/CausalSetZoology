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

function compute_all_observables(
    kind::String;
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
)
    data_paths = data_paths(["$(kind)_2048_10000/statistics.jld2","manifoldlike_simply_connected_2048_10000/statistics.jld2"])
    fields = [
        :in_degree_hist,
        :out_degree_hist,
        :in_degree_hist_link,
        :out_degree_hist_link,
        :ev_sym_link,
        :max_pathlen_hist,
    ]
    loaded = load_fields_from_paths(data_paths, fields)

    in_degree_hists      = [loaded[i][1] for i in eachindex(data_paths)]
    out_degree_hists     = [loaded[i][2] for i in eachindex(data_paths)]
    connectivity_hists      = join_histograms([in_degree_hists,out_degree_hists])
    in_degree_link_hists    = [loaded[i][3] for i in eachindex(data_paths)]
    out_degree_link_hists   = [loaded[i][4] for i in eachindex(data_paths)]
    connectivity_link_hists = join_histograms([in_degree_link_hists,out_degree_link_hists])
    ev_sym_link             = [loaded[i][5] for i in eachindex(data_paths)]
    max_pathlen_hists       = [loaded[i][6] for i in eachindex(data_paths)]

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

    D = histogram_distinguishability(normalized_in_degree_hists[1], normalized_in_degree_link_hists[2])
    m_res = mahalanobis_gap_distinguishability(normalized_in_degree_hists[1], normalized_in_degree_link_hists[2]; stabilization_method=:projection, alpha = 0.05, q = 0.05, symmetric = true)
end
