using TestItemRunner

using Test

# data_analysis
include("./data_analysis/test_utils.jl")
include("./data_analysis/test_dataloading.jl")
include("./data_analysis/test_distinguishability.jl")
include("./data_analysis/test_convergence_fitting.jl")
include("./data_analysis/test_histogram_fitting.jl")
include("./data_analysis/test_grid_fourier_analysis.jl")
include("./data_analysis/test_minkowski_abundance_analytical.jl")

# data_generation
include("./data_generation/test_utils.jl")
include("./data_generation/test_sparse_links_causet.jl")
include("./data_generation/test_graph_observables.jl")
include("./data_generation/test_generate_dataset.jl")
include("./data_generation/test_generate_statistics.jl")

@run_package_tests
