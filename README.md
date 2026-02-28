# CausalSetZoology

CausalSetZoology is a Julia package for:

- generating large sets of causal sets (`src/data_generation`),
- computing derived statistics on those datasets,
- and running analysis/plotting workflows (`src/data_analysis`).

It builds on [CausalSets.jl](https://codeberg.org/cyclopentane/CausalSets.jl/) and [QuantumGrav](https://github.com/ssciwr/QuantumGrav/).

## Quick Start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## First Dataset Run

Create a tiny end-to-end dataset + statistics artifact:

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

This creates:

- `runs/random_smoke/dataset.jld2`
- `runs/random_smoke/statistics.jld2`
- `runs/random_smoke/config.yaml`

## Documentation

- User onboarding: [docs/ONBOARDING.md](docs/ONBOARDING.md)
- Contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Common task cheat sheet: [docs/COMMON_TASKS.md](docs/COMMON_TASKS.md)
