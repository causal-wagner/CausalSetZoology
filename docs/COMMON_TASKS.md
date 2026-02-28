# Common Tasks

## Install / Refresh Environment

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run All Tests (CI Equivalent)

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Run Test Driver Directly

```bash
julia --project=test test/runtests.jl
```

## Run Only Data-Creation Tests

```bash
julia --project=test -e 'using TestItemRunner; TestItemRunner.run_tests("test/data_creation")'
```

## Run Only Data-Analysis Tests

```bash
julia --project=test -e 'using TestItemRunner; TestItemRunner.run_tests("test/data_analysis")'
```

## Tiny End-to-End Pipeline Run

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

## Dataset Generation Only (Sequential)

```bash
julia --project=. src/data_generation/make_analysis_dataset_sequential.jl \
  --kind random \
  --size 8 \
  --N 4 \
  --seed 1 \
  --batchsize 1 \
  --out runs/random_only_dataset.jld2
```

## Statistics Only (From Existing Dataset)

```bash
julia --project=. src/data_generation/make_analysis_statistics.jl \
  --in runs/random_only_dataset.jld2 \
  --out runs/random_only_statistics.jld2 \
  --num_processes 1
```

## Show CLI Help

```bash
julia src/data_generation/make_analysis_dataset_and_statistics.jl --help
julia src/data_generation/make_analysis_dataset.jl --help
julia src/data_generation/make_analysis_statistics.jl --help
```

## Sanity Check Package Import

```bash
julia --project=. -e 'import CausalSetZoology; println("ok")'
```

## Troubleshooting

- `Invalid key subdir in source section`:
  use a Julia version compatible with this repo's `Project.toml`.
- `Package ... not found`:
  ensure `--project=.` (or the intended project) is active.
- `QuantumGrav` symbol missing:
  resolve/instantiate dependencies again and verify pinned source revisions.
