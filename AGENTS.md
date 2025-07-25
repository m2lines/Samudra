# AGENTS.md

This file provides guidance to automated agents when working with code in this repository.

## Overview

Ocean Emulator is a PyTorch-based machine learning project for training and evaluating ocean model emulators. It implements a ConvNeXt U-Net neural network architecture.

## Development Commands

### Environment Setup

```bash
# Install dependencies with uv
uv sync --dev
source .venv/bin/activate
```

### Running Tests

```bash
# Run standard tests (excluding manual and CUDA tests)
uv run pytest -m "not manual and not cuda"

# Run tests with multiple CPU cores
uv run pytest -m "not manual and not cuda" -n auto

# Run CUDA tests (requires GPU)
uv run pytest -m cuda

# Run manual tests
uv run pytest -m manual -k test_whatever

# Run specific test file
uv run pytest tests/test_train.py

# Run benchmarks
uv run pytest --benchmark-only --benchmark-autosave
```

### Code Quality Checks

```bash
# Run all pre-commit checks (linting, formatting, type checking)
uvx pre-commit run --all-files
```

### Training, Evaluation, and Visualization

Please ask if you need to train or evaluate a model.

For vizualization or other long-running tasks:

* Run with `PYTHONUNBUFFERED=1 uv run ... > /tmp/logfile.txt 2>&1`
* Wait for a while with `timeout $time tail --pid=$pid -f /tmp/logfile.txt`
* You may need to run this repeatedly.

## High-Level Architecture

### Core Components

1. **Model Architecture** (`src/ocean_emulators/models/`)
   * `convnext_unet.py`: Main neural network implementing ocean predictions
   * `modules/`: Reusable network blocks (ConvNext, etc.)
   * Models predict ocean variables from ocean model data

2. **Data Pipeline** (`src/ocean_emulators/datasets.py`)
   * Handles OM4 and CM4 ocean model data via Zarr format
   * Supports time-based train/validation splits
   * Variables include temperature, salinity, u/v velocities, sea surface height, and heat
   * Data normalization and preprocessing

3. **Training Loop** (`src/ocean_emulators/train.py`)
   * Distributed training support via PyTorch DDP
   * Checkpointing with model state and optimizer
   * Weights & Biases integration for experiment tracking
   * Learning rate scheduling and warmup

4. **Evaluation System** (`src/ocean_emulators/eval.py`, `aggregator/`)
   * Comprehensive metrics including RMSE, bias, correlations
   * Ocean heat content (OHC) analysis
   * ENSO metrics and basin-specific analysis
   * Visualization tools for maps, time series, PDFs
   * Aggregator pattern for metric collection

5. **Configuration System** (`src/ocean_emulators/config.py`)
   * YAML-based configuration with JSON schema validation
   * Hierarchical configs with `!include` directives
   * Pydantic models for type safety
   * Command-line overrides supported

### Key Design Patterns

1. **Multiton Pattern**: Used for managing global state in tests via `MultitonScope`
2. **Factory Pattern**: For creating network blocks dynamically
3. **Configuration-Driven**: All major components configured via YAML
4. **Aggregator Pattern**: For collecting distributed metrics during evaluation

### Project Structure

```text
src/ocean_emulators/
├── train.py              # Training entry point
├── eval.py               # Evaluation entry point
├── config.py             # Configuration classes
├── config_schema.py      # JSON schema generation
├── datasets.py           # Data loading
├── models/               # Neural network architectures
├── aggregator/           # Metric aggregation
└── utils/                # Utilities for distributed training, logging

configs/                  # YAML configuration files
tests/                    # Comprehensive test suite
scripts/                  # Data download and preprocessing
```

### Important Considerations

1. **Data Format**: Uses Zarr format for efficient array storage
2. **Distributed Training**: Supports multi-GPU via PyTorch DDP
3. **Performance Mindset**: We include profiling tools (memray, py-spy, scalene) and aim to keep the code performant core train and eval loops.
4. **Testing Philosophy**: Tests marked as `manual` or `cuda` for selective execution
5. **Pre-commit Hooks**: Enforces code quality (ruff, mypy, detect-secrets)
6. **Cloud Training**: Supports SkyPilot for remote job execution
