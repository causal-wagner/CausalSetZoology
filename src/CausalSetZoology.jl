module CausalSetZoology

import JLD2
import CairoMakie
import LaTeXStrings
import Colors
import Printf
import Statistics
import Observables
import ColorSchemes
import Optim
import Random
import SpecialFunctions
import HypergeometricFunctions
import Polylogarithms
import ADTypes
import ForwardDiff
import QuantumGrav
import CausalSets
import FFTW
import AlgebraOfGraphics
import DataFrames
import StatsBase
import CategoricalArrays
import PlotUtils
import Distributions
import LinearAlgebra
import Distributed

export apply_paper_theme!,
    average_histogram_with_std,
    average_vectors_with_std,
    compute_fourier_grid_deviation,
    compute_mu_evolution,
    compute_sigma_evolution,
    convergence_plots_std_change,
    create_parallel_plot,
    densify_hists,
    fit_curve,
    fit_histogram_bins,
    fit_mu_convergence,
    fit_mu_infty_beta,
    fit_sigma_convergence,
    fit_sigma_infty_alpha,
    fourier_transform_grid_deviation,
    hellinger_distance,
    hist_hist_vec_distinguishability_plot_matrix,
    hist_hist_vec_hist_plot_matrix,
    histogram_distinguishability,
    histogram_distinguishability_permutation,
    join_histograms,
    load_and_average_std_scalar,
    load_field_with_scalar,
    load_fields_from_paths,
    load_histograms_from_paths,
    mahalanobis_gap_distinguishability,
    make_undirected_adjacency_from_subgraphs,
    minkowski_cardinality_abundance,
    minkowski_cardinality_abundance_2D,
    minkowski_cardinality_abundances_2D_asymptotic,
    normalize_hists,
    normalized_laplacian_eigenvalues,
    parallel_plot_df,
    plot_alpha_bins,
    plot_and_save_hists,
    plot_and_save_vectors,
    plot_beta_bins,
    plot_fourier_grid_deviation,
    plot_mean_histograms_with_std,
    plot_subgraph_colored_node_link,
    relative_change,
    replace_zeros,
    scalar_bin_distinguishability,
    scalar_bin_distinguishability_permutation,
    scalar_bin_mahalanobis_gap_distinguishability

# Base utilities and shared helpers
include("./data_analysis/plot_theme.jl")
include("./data_analysis/minkowski_abundance_analytical.jl")
include("./data_analysis/dataloading.jl")
include("./data_analysis/utils.jl")

# Fitting and derived analyses
include("./data_analysis/histogram_fitting.jl")
include("./data_analysis/convergence_fitting.jl")
include("./data_analysis/distinguishability.jl")
include("./data_analysis/grid_fourier_analysis.jl")

# Plotting and composite visualizations
include("./data_analysis/histogram_plotting.jl")
include("./data_analysis/convergence_plotting.jl")
include("./data_analysis/parallel_coordinate_plots.jl")
include("./data_analysis/grid_fourier_analysis_plotting.jl")
include("./data_analysis/laplacian_eigenvalue_example.jl")
include("./data_analysis/plot_matrix.jl")

end # module CausalSetZoology
