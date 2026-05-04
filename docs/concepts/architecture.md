<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Architecture

## Overview

Ocean Emulators is organized around a few core components that work together to train and evaluate neural ocean emulators.

```
                        Training Pipeline
 ┌─────────────────────────────────────────────────────────┐
 │                                                         │
 │  ┌──────────┐    ┌─────────┐    ┌──────────────────┐    │
 │  │ DataSet  │───▶│ Stepper │───▶│ Model            │    │
 │  │ (Zarr)   │    │         │    │ (Samudra / FOMO) │    │
 │  └──────────┘    │         │◀───│                  │    │
 │                  │         │    └──────────────────┘    │
 │                  │         │                            │
 │                  │         │───▶ Loss ───▶ Optimizer    │
 │                  └─────────┘                            │
 │                       │                                 │
 │                       ▼                                 │
 │                  Aggregator ───▶ W&B / Metrics          │
 └─────────────────────────────────────────────────────────┘
```

The emulator autoregressively predicts future ocean states. During training, short rollouts (K=4 steps) are used. During inference, the model runs freely for hundreds of steps without ground-truth feedback.

## Core Components

```
 ┌───────────────────────────────────────────────────────────┐
 │                     Configuration                         │
 │               (YAML + Pydantic validation)                │
 └──────────┬──────────────┬──────────────┬──────────────────┘
            │              │              │
            ▼              ▼              ▼
 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │   Datasets   │  │    Models    │  │   Training   │
 │  TrainData   │  │  Base Model  │  │    train.py  │
 │  Inference   │  │   Samudra    │  │   Stepper    │
 │  Dataset     │  │    FOMO      │  │   Scheduler  │
 └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                 │                 │
        └────────┬────────┘                 │
                 ▼                          ▼
          ┌─────────────┐          ┌──────────────┐
          │   Stepper   │◀────────▶│  Aggregator  │
          │ train_batch │          │  Metrics     │
          │ validate    │          │  Plotting    │
          │ inference   │          └──────────────┘
          └─────────────┘
```

### Models

All models inherit from a common base class (`ocean_emulators.models.base`) that provides configuration for:

- Residual prediction (predict the change, not the absolute state)
- Ocean masking (land vs. ocean)
- Gradient detaching for multi-step rollouts

**Samudra** (`ocean_emulators.models.samudra`) uses a ConvNeXt U-Net backbone for ocean emulation at 1° resolution.

**Samudra 2** uses the same `Samudra` class with a wider U-Net (`[280,380,480,520]` vs `[200,250,300,400]`), reduced ConvNeXt expansion factor (2 vs 4), zonally-periodic upsampling, and a dynamic variance-weighted loss. Scales to 1°, 1/2°, and 1/4° resolution.

**FOMO** (`ocean_emulators.models.fomo`) uses an encoder → processor → decoder architecture, supporting multi-scale training on different resolutions simultaneously.

### Stepper

The `Stepper` class (`ocean_emulators.stepper`) coordinates model execution:

- `train_batch` — single training step with loss computation
- `validate_batch` — validation without gradient updates
- `inference` — long autoregressive rollouts

### Data Pipeline

`ocean_emulators.datasets` handles OM4 ocean model data in Zarr format:

- `TrainData` — training dataset with time-based splits. Supports single or multiscale training.
- `InferenceDataset` — evaluation dataset for long rollouts. Only supports a single scale of data.
- Supports 1°, 1/2°, and 1/4° resolutions

### Configuration

YAML-based configuration with Pydantic validation (`ocean_emulators.config`). Supports `!include` directives and command-line overrides. See the [Contributing Guide](../contributing.md) for details on working with the configuration system.

### Training Loop

`ocean_emulators.train` orchestrates the full training process:

- Initializes the model, optimizer, and learning rate scheduler
- Runs the training loop: for each epoch, iterates over batches via `Stepper.train_batch`
- Performs multi-step autoregressive rollouts (K steps) with gradient detaching
- Runs validation at configured intervals via `Stepper.validate_batch`
- Supports distributed training via PyTorch DDP and SLURM
- Saves checkpoints (model state, optimizer, epoch) for resumption
- Applies EMA (Exponential Moving Average) to model weights
- Logs metrics and visualizations to Weights & Biases

### Evaluation

`ocean_emulators.eval` runs long autoregressive rollouts for model evaluation:

- Loads a trained checkpoint and runs free-running inference (hundreds of steps, no ground-truth feedback)
- Computes metrics against ground-truth data: RMSE, bias, anomaly correlation
- Produces per-variable, per-depth, and spatial metric breakdowns
- Writes predicted fields to Zarr output for downstream analysis and visualization

### Aggregator

The aggregator system (`ocean_emulators.aggregator`) is a separate component that collects and organizes metrics during both training and evaluation:

- `ValidateAggregator` — computes map metrics, reduced metrics, and snapshot visualizations during training validation
- `InferenceEvaluatorAggregator` — collects metrics during long inference rollouts
- `TrainAggregator` — tracks training loss breakdowns by channel, depth, and variable
