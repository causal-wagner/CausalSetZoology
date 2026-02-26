using TestItemRunner

using Test

# data_analysis
include("./data_analysis/test_utils.jl")
include("./data_analysis/test_dataloading.jl")
include("./data_analysis/test_distinguishability.jl")
include("./data_analysis/test_convergence_fitting.jl")
include("./data_analysis/test_convergence_plotting.jl")
include("./data_analysis/test_histogram_fitting.jl")
include("./data_analysis/test_histogram_plotting.jl")
include("./data_analysis/test_grid_fourier_analysis_and_plots.jl")
include("./data_analysis/test_laplacian_eigenvalue_example.jl")
include("./data_analysis/test_minkowski_abundance_analytical.jl")
include("./data_analysis/test_parallel_coordinate_plots.jl")
include("./data_analysis/test_plot_theme.jl")
include("./data_analysis/test_plot_matrix.jl")

# data_creation
include("./data_creation/test_make_analysis_dataset.jl")
include("./data_creation/test_make_analysis_statistics.jl")
include("./data_creation/test_make_analysis_dataset_and_statistics.jl")
include("./data_creation/test_data_generation_integration.jl")

@run_package_tests
