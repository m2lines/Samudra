<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: Apache-2.0
-->

# Δ-norm 2→2 (Yuan's recipe, rerun on v2026-09)

This is an experiment record, not intended to merge.

**Δ-norm** is GraphCast/Keisler-style residual normalization. The network
predicts a unit-scale increment `d`, and the next state is `x[t] + q·d`, with
`q = τ/σ` per channel. Here `τ` is the RMS five-day change over ocean cells and
`σ` the state's normalization std. The loss is plain MSE on `d`, which is the
same as a state-space MSE with fixed weights `1/q²`, uncapped.

Yuan's run (her "residual model", Ocean_Emulator
`revision_experiments/normalized_delta`) used **2 inputs and 2 chained output
heads** per call: `z1 = x[t] + q·d1` and `z2 = z1 + q·d2`, with gradients
flowing through `z1`. Her slides (2026-11-13 lab update) show it tracking
OM4's 2015–2022 ocean heat content (OHC) trend far better than Samudra 2.

Our 2-input, 1-output version, **Δ-norm 2→1**
([#984](https://github.com/m2lines/Samudra/pull/984)), gains only about
1.4 ZJ/yr of heat against OM4's 9.8 ZJ/yr. This run tests whether going from two
outputs to one is what lost the trend.

## What differs from Δ-norm 2→1

Only three things change, all in this folder:

| Setting | Δ-norm 2→1 | Δ-norm 2→2 |
|---|---|---|
| `data.output_steps` | 1 | 2 |
| `model.residual_anchor` | `same_slot` (identical for one output) | `chained` |
| `steps` (model calls per training sample) | 8 | 4 (still 40 days) |

Everything else is identical:

- data: v2026-09 1°, training 1975–2013-09, Rust loader;
- τ: `OM4_increment_rms_wet.zarr`;
- network: the 1° U-Net with BatchNorm, in fp32;
- optimization: 8 ranks × batch 4, 70 epochs, lr 6e-4 cosine, EMA 0.999, seed 15.

`tests/test_normalized_delta.py` checks that the chained loss and its
gradients equal Yuan's two-head objective.

Yuan's run also differed from ours in four ways this rerun does not copy:

- 4 ranks × batch 8, so a BatchNorm batch of 8;
- an extra BatchNorm forward pass in her trainer;
- evaluating the best-validation checkpoint;
- the v3 data.

The primary evaluation uses the final-epoch EMA checkpoint, as for the other
Δ-norm runs. `best_validation_ckpt.pt` is evaluated too, so the checkpoint
choice can be separated out.

## Pre-registered predictions

These are scored against OM4 on the 2014-10 – 2022-12 rollout.

1. **The OHC trend comes back.**
   - Full-depth OHC trend is within ±30% of OM4's 9.8 ZJ/yr.
   - The 0–700 m trend is at least 5 ZJ/yr (Δ-norm 2→1: 0.3).

   If this fails, the 2→1 change is not what lost the trend. The next
   suspects are the data version and the checkpoint choice.
2. **One-step skill matches Δ-norm 2→1.** The one-step RMSE relative to
   persistence is within 5% of Δ-norm 2→1's at every depth for `thetao` and
   `so`, and both beat persistence for `zos`.
3. **It is stable for 8 years.** It has no NaNs, and its hottest SST stays
   below 40 °C.
4. **Deep variability is not damped more than in 2→1.** The detrended
   temporal-std ratio to OM4 at 1000–5000 m is at least 0.8.
