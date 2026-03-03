module CausalSetZoology

import JLD2
import Printf
import Statistics
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
import Distributions
import LinearAlgebra
import ProgressMeter

export average_histogram_with_std,
    average_vectors_with_std,
    compute_fourier_grid_deviation,
    compute_mu_evolution,
    compute_sigma_evolution,
    densify_hists,
    fit_curve,
    fit_histogram_bins,
    fit_mu_convergence,
    fit_mu_infty_beta,
    fit_sigma_convergence,
    fit_sigma_infty_alpha,
    hellinger_distance,
    histogram_distinguishability,
    histogram_distinguishability_permutation,
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
    relative_change,
    replace_zeros,
    scalar_bin_distinguishability,
    scalar_bin_distinguishability_permutation,
    scalar_bin_mahalanobis_gap_distinguishability

# Base utilities and shared helpers
include("./data_analysis/minkowski_abundance_analytical.jl")
include("./data_analysis/dataloading.jl")
include("./data_analysis/utils.jl")

# Fitting and derived analyses
include("./data_analysis/histogram_fitting.jl")
include("./data_analysis/convergence_fitting.jl")
include("./data_analysis/distinguishability.jl")
include("./data_analysis/grid_fourier_analysis.jl")

end # module CausalSetZoology
