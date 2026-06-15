<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# AGENTS.md

This file provides guidance to automated agents when working with code in this repository.

## Overview

Samudra is a PyTorch-based machine learning project for training and evaluating models that emulate ocean physics.
We currently support a few ML models, all of which attempt to auto-regressively predict future ocean states. The Samudra
model implements a ConvNeXt U-Net neural network architecture. We have made significant efforts to scale training to
support quarter degree (0.25 x 0.25 lat/lng) data emulation. The samudra-multi model has an encoder, processor, and
decoder structure and aims to emulate ocean physics by first translating data from a physical space to a latent space.
The samudra-multi model supports training on multiple scales of data all at once (e.g. one
degree, half degree and quarter degree), either on a "mix" or "match" schedule (i.e. the cross product of each scale for
input and label, or one input/label scale at a time per batch).

## Data

When the proper S3 style credentials are passed into the local environment, you will be able to open each dataset like so:

<details>

<summary>Example of opening 1, 1/2, and 1/4 degree Zarr data with Xarray.</summary>

```
>>> import xarray as xr
>>> # One degree data with _no_ gaussian filtering (no blur).
>>> ds = xr.open_zarr('s3://emulators/am16581/data/2025-11/om4_onedeg_v3/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 98GB
Dimensions:    (time: 4745, y: 180, x: 360)
Coordinates:
  * time       (time) object 38kB 1958-01-03 12:00:00 ... 2022-12-29 12:00:00
  * x          (x) float64 3kB 0.5 1.5 2.5 3.5 4.5 ... 356.5 357.5 358.5 359.5
  * y          (y) float64 1kB -89.24 -88.25 -87.25 -86.26 ... 87.25 88.25 89.24
Data variables: (12/99)
    hfds       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    mask_0     (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_1     (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_10    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_11    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_12    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    ...         ...
    vo_5       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_6       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_7       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_8       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_9       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    zos        (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
Attributes:
    m2lines/cli_args:                  /Users/alxmrs/git/ocean_emulators/ocea...
    m2lines/date_created:              2025-11-26T12:51:52.411906
    m2lines/ocean_emulators_git_hash:  https://github.com/m2lines/ocean_emula...
    regrid_method:                     conservative
>>> # Half degree data with no blur
>>> ds = xr.open_zarr('s3://emulators/am16581/data/2025-11/om4_halfdeg_v4/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 394GB
Dimensions:    (time: 4745, y: 360, x: 720)
Coordinates:
  * time       (time) object 38kB 1958-01-03 12:00:00 ... 2022-12-29 12:00:00
  * x          (x) float64 6kB 0.25 0.75 1.25 1.75 ... 358.2 358.8 359.2 359.8
  * y          (y) float64 3kB -89.62 -89.12 -88.62 -88.13 ... 88.62 89.12 89.62
Data variables: (12/99)
    hfds       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    mask_0     (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    mask_1     (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    mask_10    (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    mask_11    (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    mask_12    (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    ...         ...
    vo_5       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    vo_6       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    vo_7       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    vo_8       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    vo_9       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    zos        (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
Attributes:
    m2lines/cli_args:                  /Users/alxmrs/git/ocean_emulators/ocea...
    m2lines/date_created:              2025-11-26T11:46:51.855769
    m2lines/ocean_emulators_git_hash:  https://github.com/m2lines/ocean_emula...
    regrid_method:                     conservative
>>> # Quarter degree data with no blur.
>>> ds = xr.open_zarr('s3://emulators/am16581/data/2025-11/om4_quarterdeg_v2/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 2TB
Dimensions:    (time: 4745, y: 720, x: 1440)
Coordinates:
  * time       (time) object 38kB 1958-01-03 12:00:00 ... 2022-12-29 12:00:00
  * x          (x) float64 12kB 0.125 0.375 0.625 0.875 ... 359.4 359.6 359.9
  * y          (y) float64 6kB -89.81 -89.56 -89.31 -89.06 ... 89.31 89.56 89.81
Data variables: (12/99)
    hfds       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    mask_0     (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    mask_1     (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    mask_10    (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    mask_11    (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    mask_12    (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    ...         ...
    vo_5       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    vo_6       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    vo_7       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    vo_8       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    vo_9       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    zos        (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
Attributes:
    hfds:                              {'cell_measures': 'area: areacello', '...
    m2lines/cli_args:                  /Users/alxmrs/git/ocean_emulators/ocea...
    m2lines/date_created:              2025-12-01T15:34:44.338655
    m2lines/ocean_emulators_git_hash:  https://github.com/m2lines/ocean_emula...
    regrid_method:                     conservative
    so:                                {'cell_measures': 'area: areacello', '...
    tauuo:                             {'cell_methods': 'yh:mean xq:point tim...
    tauvo:                             {'cell_methods': 'yq:point xh:mean tim...
    thetao:                            {'cell_measures': 'area: areacello', '...
    uo:                                {'cell_methods': 'z_l:mean yh:mean xq:...
    vo:                                {'cell_methods': 'z_l:mean yq:point xh...
    zos:                               {'cell_measures': 'area: areacello', '...
```

</details>

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

1. **Model Architecture** (`src/samudra/models/`)
   * `samudra.py`: Samudra model using ConvNeXt U-Net backbone for single-scale ocean emulation
   * `samudra_multi.py`: samudra-multi encoder → processor → decoder architecture supporting multi-scale training
   * `samudra_mini.py`: SamudraMini single PerceiverIO model for lightweight training-shape experiments
   * `base.py`: Abstract base model class with common functionality (residual predictions, masking, gradient detaching)
   * `corrector.py`: Optional corrector network for error correction (will likely be deprecated soon)
   * `modules/`: Reusable building blocks including `unet_backbone.py`, `encoder.py` (PerceiverEncoder), `blocks.py` (ConvNext blocks), `activations.py`, and `augment_input.py`

2. **Time Stepping** (`src/samudra/stepper.py`)
   * `Stepper` class with static methods: `train_batch`, `validate_batch`, and `inference`
   * Handles single-step forward passes vs. multi-step autoregressive rollouts
   * Coordinates model execution with loss computation and output writing

3. **Data Pipeline** (`src/samudra/datasets.py`)
   * Handles OM4 ocean model data via Zarr format
   * `TrainData` and `InferenceDataset` classes for training/eval
   * Supports time-based train/validation splits
   * Variables include temperature (`thetao`), salinity (`so`), u/v velocities, sea surface height (`zos`), and surface heat flux (`hfds`)
   * Data normalization via the `Normalize` multiton (only used in the Corrector and Aggregator, should be deprecated)

4. **Training Loop** (`src/samudra/train.py`)
   * Distributed training support via PyTorch DDP
   * Checkpointing with model state and optimizer
   * Weights & Biases integration for experiment tracking
   * Learning rate scheduling with warmup (`utils/schedule.py`)
   * EMA (Exponential Moving Average) support (`utils/ema.py`)

5. **Evaluation System** (`src/samudra/eval.py`, `aggregator/`)
   * `aggregator/main.py`: Base `Aggregator` class for metric collection
   * `aggregator/inference/`: `InferenceEvaluatorAggregator` for rollout evaluation
   * `aggregator/validate/`: `ValidateAggregator` with sub-aggregators for map metrics, reduced metrics, and snapshots
   * `aggregator/loss.py`: Loss utilities (channel, depth, variable breakdowns)
   * `aggregator/metrics.py`: Metric computation
   * `aggregator/plotting.py`: Visualization during training/eval

6. **Visualization** (`src/samudra/viz/`)
   * `core.py`: Core visualization logic for maps, time series, PDFs
   * `config.py`: Visualization configuration
   * `__main__.py`: Entry point (`python -m samudra.viz`)

7. **Configuration System** (`src/samudra/config.py`, `config_base.py`, `config_schema.py`)
   * YAML-based configuration with JSON schema validation
   * Hierarchical configs with `!include` directives
   * Pydantic models for type safety
   * Command-line overrides supported (see `--help`)
   * Schemas in `configs/schemas/` for IDE autocomplete

8. **Utilities** (`src/samudra/utils/`)
   * `data.py`: Data utilities and preprocessing (largest utility module)
   * `distributed.py`: DDP utilities for multi-GPU training
   * `wandb.py`: Weights & Biases integration
   * `samplers.py`: Custom data samplers (including distributed equivalence)
   * `loss.py`: Loss function utilities
   * `schedule.py`: Learning rate scheduling
   * `ema.py`: Exponential Moving Average for model weights
   * `multiton.py`: Multiton pattern for test isolation
   * `profiler.py`: CUDA memory profiling integration
   * `device.py`: Device management and autocast
   * `logging.py`: Logging configuration
   * `writer.py`: Zarr output writing
   * `output.py`: Output dataclasses (`TrainBatchOutput`, `ValBatchOutput`, `ModelInferenceOutput`)
   * `location.py`: Geospatial location utilities
   * `compare.py`: Model/output comparison utilities

### Key Design Patterns

1. **Multiton Pattern**: Used for managing global state (e.g., `Normalize`) with test isolation via `MultitonScope`
2. **Factory Pattern**: Network blocks created dynamically from config
3. **Configuration-Driven**: All major components configured via YAML with Pydantic validation
4. **Aggregator Pattern**: Hierarchical metric collection during training and evaluation (We want to move away from this)
5. **Stepper Abstraction**: Separates model forward logic from training/eval orchestration

### Project Structure

```text
src/samudra/
├── train.py              # Training entry point
├── eval.py               # Evaluation entry point
├── stepper.py            # Time-stepping and inference logic
├── datasets.py           # Data loading (TrainData, InferenceDataset)
├── config.py             # Configuration classes
├── config_base.py        # Base configuration classes
├── config_schema.py      # JSON schema generation
├── constants.py          # Project-wide constants (Grid, etc.)
├── derived_variables.py  # Calculation of derived ocean variables
├── models/
│   ├── base.py           # Abstract base model
│   ├── samudra.py        # Samudra (ConvNeXt U-Net)
│   ├── samudra_multi.py  # samudra-multi (encoder-processor-decoder)
│   ├── samudra_mini.py   # SamudraMini (single PerceiverIO)
│   ├── corrector.py      # Corrector network
│   └── modules/          # Reusable blocks (unet_backbone, encoder, blocks, activations)
├── aggregator/
│   ├── main.py           # Base Aggregator class
│   ├── train.py          # Training aggregation
│   ├── inference/        # Inference evaluation (InferenceEvaluatorAggregator)
│   ├── validate/         # Validation aggregation (map, reduced, snapshot)
│   ├── loss.py           # Loss utilities
│   ├── metrics.py        # Metric computation
│   └── plotting.py       # Visualization utilities
├── viz/                  # Visualization module
│   ├── core.py           # Core viz logic
│   ├── config.py         # Viz configuration
│   └── __main__.py       # Entry point
└── utils/                # 16 utility modules (see above)

configs/
├── samudra_om4/          # Samudra model configs (train, eval, viz, model)
├── samudra_multi_om4/    # samudra-multi model configs (incl. train_multiscale.yaml)
├── samudra_mini_om4/     # SamudraMini model configs
├── data/                 # Data configuration (om4.yaml)
├── test/                 # Minimal test configs
└── schemas/              # JSON schemas for validation

tests/                    # Test suite (16 test files)
scripts/                  # Data cloning, preprocessing, job scripts
skypilot/                 # SkyPilot cloud training configs (train, eval, viz)
notebooks/                # Analysis and preprocessing notebooks
```

### Important Considerations

1. **Data Format**: Uses Zarr format for efficient array storage; supports multiple resolutions (1°, 0.5°, 0.25°)
2. **Distributed Training**: Supports multi-GPU via PyTorch DDP; also supports SLURM and torchrun
3. **Performance Mindset**: We include profiling tools (memray, py-spy, scalene) and aim to keep the code performant in core train and eval loops
4. **Testing Philosophy**: Tests marked as `manual` or `cuda` for selective execution; use `MultitonScope` for test isolation
5. **Cloud Training**: Supports SkyPilot for remote job execution on AWS & Lambda Labs
6. **Noisy Failure**: Do not swallow errors. If something goes wrong, let it fail loudly.
7. **Avoid Hacks**: Don't accommodate bad designs by adding more cruft -- refactor separately first then make the nice change.
8. **Multi-Scale Support**: samudra-multi supports training on multiple data resolutions simultaneously with "mix" or "match" scheduling
