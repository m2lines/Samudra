<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Loss

The loss module provides standard and dynamic loss functions for training.

## Dynamic Loss

The `DynamicLoss` class implements the variance-weighted loss used in Samudra 2. It maintains per-channel scaling weights updated via an exponential moving average of inverse prediction error:

- Channels with lower running error receive higher weight, amplifying signals from slow-evolving fields
- A configurable `limit` parameter (default: 20) clamps the max ratio between channel weights to prevent extreme imbalance
- Uses a rolling window of 25 steps to smooth scale estimates

**Configuration:**

```yaml
# Standard MSE (Samudra v1)
loss: mse

# Dynamic variance-weighted loss (Samudra 2)
loss:
  type: dynamic
  metric: mse
  limit: 20
```

## Tendency loss and scaled residual prediction

The tendency loss uses fixed inverse-variance weights derived from each
channel's native one-step change, rather than the model's current error.
Output scaling makes the decoder predict changes in those same units:

```yaml
data:
  input_steps: 2
  output_steps: 1
model:
  pred_residuals: true
  residual_scale_location: OM4_persistence_rms.zarr
loss:
  type: tendency
  metric: mse
  persistence_rms_location: OM4_persistence_rms.zarr
  max_weight_ratio: 1000
```

These fields extend the existing data/model configuration. Both locations
resolve against `experiment.data_root`. The statistics store contains one scalar
RMS per prognostic channel in physical units, computed from training data only,
with land counted as zero change and averaging over the full grid.

For each channel, the output scale is `persistence_rms / state_std`. The model
multiplies decoder outputs by that scale before adding the latest input state.
With two inputs and one output this is a native five-day residual. Scales repeat
in time-major order for multiple outputs, which retain their corresponding-input
residual anchors. Loss weights use the inverse square of the output scales,
optionally capped, and normalized to mean one across channels. The cap does not
alter the output scales.

Output scales are fixed model buffers saved in raw and EMA checkpoints. Training
and evaluation initialize them using the same configuration and normalization;
checkpoint loading restores their saved values. Omit `residual_scale_location`
to retain existing model behavior and checkpoint keys. Scaling requires residual
prediction and a single training data source.

## API Reference

::: samudra.utils.loss
