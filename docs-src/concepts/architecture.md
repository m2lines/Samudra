# Architecture

## Overview

Ocean Emulators is organized around a few core components that work together to train and evaluate neural ocean emulators.

```
Input (two ocean states + forcing) → Model → Output (next two predicted states)
```

The emulator autoregressively predicts future ocean states. During training, short rollouts (K=4 steps) are used. During inference, the model runs freely for hundreds of steps without ground-truth feedback.

## Core Components

### Models

All models inherit from a common base class (`ocean_emulators.models.base`) that provides:

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

- `TrainData` — training dataset with time-based splits
- `InferenceDataset` — evaluation dataset for long rollouts
- Supports 1°, 1/2°, and 1/4° resolutions

### Configuration

YAML-based configuration with Pydantic validation (`ocean_emulators.config`). Supports `!include` directives and command-line overrides.

### Training Loop

`ocean_emulators.train` provides:

- Distributed training via PyTorch DDP
- Checkpointing
- Learning rate scheduling with warmup
- EMA (Exponential Moving Average) support
- Weights & Biases integration

### Evaluation

The aggregator system (`ocean_emulators.aggregator`) collects metrics during training and evaluation, including map metrics, reduced metrics, and snapshot visualizations.
