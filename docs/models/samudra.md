<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Samudra

The Samudra model uses a ConvNeXt U-Net backbone for autoregressive ocean emulation. Both Samudra (v1) and Samudra 2 (v2) share the same `Samudra` class — the architectural differences are driven by configuration.

## Samudra v1 vs Samudra 2

| | Samudra (v1) | Samudra 2 (v2) |
|---|---|---|
| **Channel widths** | `[200, 250, 300, 400]` | `[280, 380, 480, 520]` |
| **ConvNeXt expansion factor** | 4 | 2 |
| **Upsampling** | Bilinear | Zonally periodic |
| **Loss** | MSE | Dynamic variance-weighted MSE (limit: 20) |
| **Resolutions** | 1° | 1°, 1/2°, 1/4° |

Samudra 2 widens the U-Net stages by ~40% while reducing the block-internal expansion factor, shifting capacity toward inter-stage features. The dynamic loss reweights per-channel MSE inversely by each channel's running prediction error, amplifying the gradient signal from slow-evolving deep-ocean fields.

**Configs:**

- Samudra v1: `configs/samudra_om4_v1/`
- Samudra 2: `configs/samudra_om4_v2/`
- Samudra 2 (high-res): `configs/samudra_om4_v2_highres/`

## API Reference

::: ocean_emulators.models.samudra
