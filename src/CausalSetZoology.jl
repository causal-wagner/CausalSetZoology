module CausalSetZoology

import JLD2
import Statistics
import Optim
import Random
import SpecialFunctions
import QuantumGrav
import CausalSets
import FFTW
import Distributions
import LinearAlgebra
import StatsBase
import ProgressMeter
import Distributed

export average_histogram_with_std,
    average_vectors_with_std,
    concatenate_hists,
    compute_fourier_grid_deviation,
    compute_mu_evolution,
    compute_sigma_evolution,
    distance_distinguishability_probability,
    null_distance_percentile,
    densify_hists,
    densify_hists_vectors,
    fit_curve,
    fit_histogram_bins,
    fit_mu_convergence,
    fit_mu_infty_beta,
    fit_sigma_convergence,
    fit_sigma_infty_alpha,
    hellinger_distance,
    distinguishability_mutual_information,
    distinguishability_total_variation,
    energy_based_histogram_distinguishability,
    histogram_distinguishability_permutation,
    total_histogram_distinguishability,
    total_histogram_total_variation_distinguishability,
    total_histogram_mutual_information_distinguishability,
    join_histograms,
    load_and_average_std_scalar,
    load_field_with_scalar,
    load_fields_from_paths,
    load_histograms_from_paths,
    mahalanobis_gap_distinguishability,
    minkowski_cardinality_abundance,
    minkowski_cardinality_abundance_2D,
    minkowski_cardinality_abundances_2D_asymptotic,
    normalize_hists,
    mc_pairwise_apply,
    histogram_to_dense_pair,
    weighted_hist_mean,
    weighted_hist_std,
    weighted_hist_skew,
    weighted_hist_exkurt,
    aggregate_hist_moment,
    relative_change,
    replace_zeros,
    scalar_bin_distinguishability,
    scalar_bin_distance_distinguishability_probability,
    scalar_bin_distinguishability_permutation,
    scalar_bin_mahalanobis_gap_distinguishability,
    ev_summary,
    symmetrize_strictly_upper_triangular!,
    sym_norm_lap_eigs!,
    normalized_lap_eigs_symmetrized_links,
    imag_antisym_out_lap,
    imag_antisym_out_lap_eigs,
    imag_antisym_in_lap,
    imag_antisym_in_lap_eigs,
    communicability_row_sums,
    degrees,
    height,
    connectivity,
    sparse_hist,
    dense_future_links,
    SparseLinksCauset,
    compute_statistics,
    create_statistics_dataset_and_save,
    generate_batch,
    create_dataset_and_save



# Base utilities and shared helpers
include("./data_analysis/minkowski_abundance_analytical.jl")
include("./data_analysis/dataloading.jl")
include("./data_analysis/utils.jl")
include("./data_analysis/moments_of_distributions.jl")
include("./data_generation/SparseLinksCauset.jl")
include("./data_generation/utils.jl")

# Fitting and derived analyses
include("./data_analysis/histogram_fitting.jl")
include("./data_analysis/convergence_fitting.jl")
include("./data_analysis/distinguishability.jl")
include("./data_analysis/grid_fourier_analysis.jl")
include("./data_generation/generate_statistics.jl")
include("./data_generation/generate_dataset.jl")
include("./data_generation/graph_observables.jl")

end # module CausalSetZoology
