<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Normalized-delta loss, 2 inputs / 1 output, 1° OM4

**Experiment record, not intended to merge.**

## What this is

Yuan's `normalized_delta` model for the Samudra 2 revision
(`YuanYuan98/Ocean_Emulator`, `revision_experiments/normalized_delta`) is the
1° Samudra U-Net with two changes:

1. **Scaled residual head.** The network emits a unit-scale increment `d`, and
   the next state is `x[t+1] = x[t] + q * d`, with a fixed per-channel
   `q = tau / sigma`. Here `tau` is the RMS five-day change over wet cells
   (physical units, 1975-01-03 to 2013-09-30), and `sigma` is the state's
   normalization std.
2. **Loss on normalized increments.** Unweighted masked MSE between `d` and
   `(label - x[t]) / q`. Since `q` is fixed, this equals a state-space MSE with
   fixed weights `1 / q**2`. That is how it is implemented here, reusing
   `TendencyLoss`.

Together these are GraphCast's residual normalization. Her model has two
output heads per call (`z2 = z1 + q * d2`). This variant predicts **one**
five-day step per call from two input states (`input_steps: 2`,
`output_steps: 1`), with `steps: 8` so that training still covers her 40-day
horizon. All other settings follow her `train.yaml`: BatchNorm, fp32, 4 ranks
x batch 8, 70 epochs, lr 6e-4 cosine, EMA 0.999, seed 15.

## Code

* `model.scale_residuals` (`src/samudra/models/base.py`): a persistent
  `residual_scale` buffer that is NaN until training fills it, and that
  evaluation restores from the checkpoint.
* `loss.type: normalized_delta` (`src/samudra/config.py`).
* `scripts/compute_tendency_stats.py --over wet` writes `tau`.
* `tests/test_normalized_delta.py` checks loss values and gradients against
  her formulas.

## Data

Official `v2026-09` 1° release (`docs/data.md`). Her `sigma` equals that
release's `OM4_stds` for all 77 channels.

## Known differences from her run

* **Loss magnitude is 2× hers.** Each training sample sums 8 one-output steps,
  each averaged over 77 channels. Hers summed 4 two-output steps averaged over
  154 head-channels. Adam cancels a constant factor, but gradient clipping at
  norm 1.0 does not, so some batches are clipped here that would not be in
  hers.
* **No extra BatchNorm forward.** Her trainer ran one extra train-mode forward
  per batch (for its no-op loss `update`), which also updates BatchNorm running
  statistics. The fixed-weight loss here skips that pass, so the eval-mode
  BatchNorm statistics differ slightly.
* Training ends 2013-09-30 (the `om4` preset); hers ended 2013-10-05.
