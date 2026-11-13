<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Loss

The loss module provides standard and dynamic loss functions for training.

## MSE and compatibility

`loss: mse` is the default. It computes squared error in the model's normalized
units, averages over wet cells and batch entries separately for each channel,
then the training step averages the channels equally. It does not use cell-area
weights. Channels with fewer wet cells receive the same coefficient as other
channels. An all-dry channel returns zero loss and zero gradients. Nonfinite
land values are ignored; nonfinite wet values remain visible in the loss.

`loss: legacy_mse` preserves the previous reduction: zero land values, then
average over the full rectangular grid. Its channel loss equals the wet-cell
MSE multiplied by the channel's wet-cell fraction. Use this option to reproduce
old training objectives or resume a run without changing its loss reduction.
Existing configurations containing `mse` now select the wet-cell reduction.
When resuming, set `loss: legacy_mse` in the training configuration passed to
the new invocation; the checkpoint does not select the loss implementation.
For a dynamic loss, set `loss.metric: legacy_mse` in that configuration to
preserve the old base reduction.
Loss values from the two reductions are not directly comparable.

## Dynamic Loss

The `DynamicLoss` class implements the variance-weighted loss used in Samudra 2. It maintains per-channel scaling weights updated via an exponential moving average of inverse prediction error:

- Channels with lower running error receive larger inverse-error weights, balancing channel contributions
- A configurable `limit` parameter clamps the ratio between channel weights; the shipped presets use 20, while the class default is no limit
- Uses a rolling window of 25 steps to smooth scale estimates

**Configuration (choose one alternative):**

```yaml
# Wet-cell MSE (default)
loss: mse
```

```yaml
# Historical full-grid MSE
loss: legacy_mse
```

```yaml
# Dynamic variance-weighted loss (Samudra 2)
loss:
  type: dynamic
  metric: mse
  limit: 20
```

## API Reference

::: samudra.utils.loss
