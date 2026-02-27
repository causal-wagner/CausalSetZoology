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
