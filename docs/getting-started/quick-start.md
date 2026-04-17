# Quick Start

## Running Tests

```bash
# Standard tests (excluding manual and CUDA)
uv run pytest -m "not manual and not cuda"

# Parallel execution
uv run pytest -m "not manual and not cuda" -n auto

# CUDA tests (requires GPU)
uv run pytest -m cuda

# Benchmarks
uv run pytest --benchmark-only --benchmark-autosave
```

## Code Quality

```bash
# Run all pre-commit checks (linting, formatting, type checking)
uvx pre-commit run --all-files
```

## Training

Training is configured via YAML files. See `configs/` for examples:

- `configs/samudra_om4/` — Samudra model configs
- `configs/fomo_om4/` — FOMO model configs

```bash
uv run python -m ocean_emulators.train --config configs/samudra_om4/train.yaml
```

## Evaluation

```bash
uv run python -m ocean_emulators.eval --config configs/samudra_om4/eval.yaml
```
