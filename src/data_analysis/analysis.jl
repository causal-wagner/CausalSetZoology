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
include("plot_theme.jl")
include("minkowski_abundance_analytical.jl")
include("dataloading.jl")
include("utils.jl")

# Fitting and derived analyses
include("histogram_fitting.jl")
include("convergence_fitting.jl")
include("distinguishability.jl")

# Plotting and composite visualizations
include("histogram_plotting.jl")
include("convergence_plotting.jl")
include("parallel_coordinate_plots.jl")
include("grid_fourier_analysis_and_plots.jl")
include("laplacian_eigenvalue_example.jl")
include("plot_matrix.jl")
