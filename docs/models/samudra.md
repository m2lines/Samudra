<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

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

- Samudra v1: `src/samudra/configs/samudra_om4_v1/`
- Samudra 2: `src/samudra/configs/samudra_om4_v2/`
- Samudra 2 (high-res): `src/samudra/configs/samudra_om4_v2_highres/`

## Land: `mask` or `omit`

By default (`land: mask`), the U-Net computes over every grid cell, land included, and zeroes land only in its output.
Land values therefore still mix into ocean cells near coasts through every convolution, batch norm, pool and upsample.

With `land: omit` in the model config, land is absent from every layer
([#1001](https://github.com/m2lines/Samudra/issues/1001)):

| Operator | With `land: omit` |
| --- | --- |
| Convolutions (blocks and decoder) | Partial convolution ([Liu et al., 2018](https://arxiv.org/abs/1804.07723)): `conv(x·m) · k²/Σm + b`, then `·m` |
| Pointwise (1×1) layers | output `·m` |
| Batch norm | mean and variance over ocean cells only; under DDP, over the ocean cells of the global batch |
| Average / max pooling | over ocean cells; a coarse cell is ocean if any of its fine cells is |
| Bilinear upsampling | `up(x·m) / up(m)`, then `·m` at the finer level |

Here `m` is the ocean mask: each prognostic input channel uses its own depth's mask, and the boundary forcing and every
hidden layer use the column mask (ocean at any depth, the surface mask since masks nest). The forecast is then exactly
independent of the values on land. Padding is unchanged; the zero padding past the poles reads as absent.

Both settings have the same parameters and buffers, so a checkpoint trained with either loads with either. `omit`
supports `norm: batch` and the bilinear upsampling blocks, not `transposed_conv`. The extra cost is one single-channel
convolution per layer and a few element-wise products.

## API Reference

::: samudra.models.samudra
