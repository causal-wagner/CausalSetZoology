# Contributing to CausalSetZoology

## Development Setup

From repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Optional test env setup:

```bash
julia --project=test -e 'using Pkg; Pkg.instantiate()'
```

## Run Tests

Primary (CI-aligned) test entrypoint:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Alternative test driver:

```bash
julia --project=test test/runtests.jl
```

## Scope and Structure

- Package entrypoint: `src/CausalSetZoology.jl`
- Data generation scripts: `src/data_generation/`
- Analysis utilities and plotting: `src/data_analysis/`
- Tests: `test/data_analysis/`, `test/data_creation/`

When adding functionality:

1. Implement in `src/`.
2. Add/extend tests under the matching `test/` subtree.
3. Run tests locally before opening a PR.

## Style Guidelines

- Keep functions small and typed where practical.
- Validate user-facing inputs with clear `ArgumentError`/`DomainError`.
- Prefer docstrings for public or widely used helpers.
- Avoid adding notebook-only logic into package modules.

## Data-Generation Changes

If you modify scripts in `src/data_generation/`, run at least one small end-to-end smoke run:

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

## Dependency and Version Notes

- Keep Julia compatibility in `Project.toml` aligned with any `[sources]` features used.
- If upstream `QuantumGrav`/`CausalSets` APIs change, update tests and lockfiles coherently.

## Pull Requests

Please include:

- What changed and why.
- Any breaking behavior changes.
- Test evidence (`Pkg.test()` and/or focused test runs).
- For data-generation changes: example command used and expected outputs.
