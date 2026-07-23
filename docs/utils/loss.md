<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Loss

The loss module provides standard and dynamic loss functions for training.

## Dynamic Loss

The `DynamicLoss` class implements the variance-weighted loss used in Samudra 2. It maintains per-channel scaling weights updated via an exponential moving average of inverse prediction error:

- Channels with **higher error** receive **higher weight**, preventing the model from neglecting slow-evolving deep-ocean fields
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

## API Reference

::: samudra.utils.loss
