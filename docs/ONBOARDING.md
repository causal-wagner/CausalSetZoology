# CausalSetZoology Onboarding

## 1. What This Repo Contains

- `src/CausalSetZoology.jl`: package entrypoint; includes all analysis modules.
- `src/data_generation/`: CLI scripts to generate datasets and statistics.
- `src/data_analysis/`: reusable analysis and plotting utilities.
- `test/`: package tests (`test/runtests.jl`).
- `analysis/`: notebook-oriented environment.

## 2. Prerequisites

- Julia version compatible with `[compat]` in `Project.toml`.
- Git and standard shell tools.

## 3. Environment Setup

From the repo root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Optional: instantiate the dedicated test environment as well:

```bash
julia --project=test -e 'using Pkg; Pkg.instantiate()'
```

## 4. Run Tests

Canonical package test entrypoint:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

If you want to run the test driver directly:

```bash
julia --project=test test/runtests.jl
```

## 5. First End-to-End Run

Use the pipeline driver in `src/data_generation` to generate both dataset and statistics:

```bash
julia --project=. src/data_generation/make_analysis_dataset_and_statistics.jl \
  --kind random \
  --size 8 \
  --num_csets 4 \
  --seed 1 \
  --num_processes 1 \
  --dataset_multiprocessing false \
  --batchsize 1 \
  --outdir runs/random_smoke
```

Expected outputs:

- `runs/random_smoke/dataset.jld2`
- `runs/random_smoke/statistics.jld2`
- `runs/random_smoke/config.yaml`
- reproducibility snapshots (`Project.toml`, `Manifest.toml`, copied scripts).

## 6. Core Script Entry Points

- `src/data_generation/make_analysis_dataset_and_statistics.jl`
  - orchestrates dataset + statistics generation.
- `src/data_generation/make_analysis_dataset.jl`
  - multiprocessing dataset generation.
- `src/data_generation/make_analysis_dataset_sequential.jl`
  - sequential dataset generation.
- `src/data_generation/make_analysis_statistics.jl`
  - statistics generation from an existing dataset.

For options/help:

```bash
julia src/data_generation/make_analysis_dataset_and_statistics.jl --help
julia src/data_generation/make_analysis_dataset.jl --help
julia src/data_generation/make_analysis_statistics.jl --help
```

## 7. Programmatic Usage (REPL / Scripts)

```julia
using CausalSetZoology

# Analytical abundance helper
a = CausalSetZoology.minkowski_cardinality_abundance_2D(100, 2)

# Load one histogram field from a statistics file
paths = ["runs/random_smoke/statistics.jld2"]
hists = CausalSetZoology.load_histograms_from_paths(paths, :cardinalities_hist)

# Normalize and densify for downstream analysis
hn = CausalSetZoology.normalize_hists(hists; normalization=:probability)
X = CausalSetZoology.densify_hists(hn[1])
```

## 8. Analysis Modules Overview

- Data loading: `dataloading.jl`
- Histogram/vector helpers: `utils.jl`
- Curve fitting: `histogram_fitting.jl`
- Convergence fitting/plots: `convergence_fitting.jl`, `convergence_plotting.jl`
- Distinguishability metrics: `distinguishability.jl`
- Fourier grid analysis: `grid_fourier_analysis.jl`, `grid_fourier_analysis_plotting.jl`
- Composite plotting: `plot_matrix.jl`, `parallel_coordinate_plots.jl`

## 9. Typical Contributor Workflow

1. Instantiate environment (`--project=.`).
2. Make changes in `src/`.
3. Run `Pkg.test()` from repo root.
4. If touching data-generation scripts, run a tiny smoke pipeline as in Section 5.
5. Open/update tests in `test/data_analysis` or `test/data_creation`.

## 10. Troubleshooting

- `Invalid key subdir in source section`:
  - your Julia version is older than what this repo requires for `[sources]` usage.
- `Package ... not found` while running ad-hoc scripts:
  - ensure you used `--project=.` or activated the intended environment.
- `QuantumGrav` function missing / API mismatch:
  - dependency revision drift; refresh/resolve environment and ensure both root and test environments are consistent.
- Long test duration:
  - integration tests in `test/data_creation` can be expensive; run targeted subsets while iterating.
