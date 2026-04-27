# Ocean Emulators

**A PyTorch-based machine learning framework for training and evaluating neural ocean emulators.**

Ocean Emulators provides tools to build models that learn to predict future ocean states autoregressively, achieving orders-of-magnitude speedups over traditional ocean general circulation models.

## Models

- **Samudra** — ConvNeXt U-Net architecture for single-scale ocean emulation at 1° resolution.
- **Samudra 2** — Wider ConvNeXt U-Net with dynamic variance-weighted loss, scaling to 1°, 1/2°, and 1/4° resolution with stable ~8-year rollouts.
- **FOMO** — A "Foundation Ocean Model + Observations" uses a encoder-processor-decoder architecture to support multi-scale training.

## Key Features

- Autoregressive prediction of temperature, salinity, velocities, and sea surface height across 19 depth levels.
- Multi-resolution support: 1°, 1/2°, and 1/4°.
- Distributed training via PyTorch DDP.
- Dynamic variance-weighted loss for balanced learning across variables.
- Weights & Biases integration for experiment tracking.

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quick-start.md)
- [Architecture](concepts/architecture.md)
- [API Reference](models/base.md)
