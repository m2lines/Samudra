<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Autoregressive rollout curriculum for normalized-delta (2 in / 1 out, 1°)

**Experiment record, not intended to merge.** It is stacked on PR #984
(`experiments/normalized_delta_2in1out`). It fine-tunes #984's final checkpoint
on longer rollouts. The model, loss and data don't change.

## From first principles

### What the training loss can and can't see

Samudra trains on short autoregressive rollouts. In #984, each sample runs the
model 8 times, feeding each prediction back as input, and sums the loss over
those 8 five-day steps (40 days). Gradients flow through the whole chain. So
the loss penalizes errors that grow within 40 days, but nothing beyond that.

The failures we care about appear much later. In #969's eight-year rollout,
which uses #984's setup apart from the loss weighting:

- mean SST drifts by −0.25 °C;
- surface salinity drifts by −0.07 psu;
- one region's maximum SST climbs from 32.1 °C to 38.7 °C and is still rising
  at the end.

### How a residual model drifts

A residual model's state after `n` steps is the true state plus the sum of
its per-step errors. If a perturbation `δ` to the state grows by a factor
`(1 + λ)` per step, a 40-day window sees it grow by `(1 + λ)⁸`.

When `λ` is small, say 0.5% per step, that is a 4% change in 40 days, which
the loss barely notices. Over a year it is a factor of 1.4, and over eight
years a factor of 18.

**A one-step or short-rollout loss is nearly blind to slow instabilities.**
The only direct signal against them is a loss computed through longer
rollouts.

#980's compression ladder shows the extreme case. It is fitted to one step,
is the best one-step predictor there, and diverges in about three months
([root-cause
analysis](https://github.com/m2lines/Samudra/blob/391b6c64/experiments/compression_baselines/reports/rollout_blowup_rca.md)).

### What GraphCast and Keisler did

- **GraphCast** (Lam et al. 2023) trains on 1 step, then fine-tunes with a
  curriculum of autoregressive rollouts from 2 to 12 steps (12 hours to
  3 days) at a low learning rate. It backpropagates through every step.
- **Keisler** (2022) trains on rollouts of increasing length, up to 12 steps
  (3 days).
- **FourCastNet** and **Pangu-Weather** use related multi-step fine-tuning to
  improve long-lead skill and stability.

For the atmosphere, 3 days is several error-growth timescales. For the
ocean's slow drifts it is much less, which is why this curriculum goes further.

## The change

Fine-tune #984's final checkpoint (`saved_nets/ckpt.pt`: model weights and
EMA, but a new optimizer):

| Epochs | Rollout length | Horizon |
|---|---|---|
| 1–4 | 12 steps | 60 days |
| 5–9 | 24 steps | 120 days |
| 10–15 | 36 steps | 180 days |

**Batch size and memory.** It runs one sample per GPU (global batch 8), at
Adam 3e-5 with cosine annealing. Peak GPU memory grows with samples × steps:
36 × 1 is close to #984's 8 × 4, which peaked at 71.8 GB in #969. Each
update therefore costs about the same, and it covers as many supervised
steps as one of #984's.

**Gradients.** They run through the whole 36-step chain, as in GraphCast.
`model.gradient_detach_interval` could truncate them later if memory
requires it.

`tests/test_normalized_delta.py` runs the recipe end to end on mock data. It
trains a `normalized_delta` model, fine-tunes the checkpoint through a step
curriculum, and checks that the residual scale survives the reload.

## Predictions, made before training

Against #984's final checkpoint, which is the starting point:

1. **Eight-year drift shrinks.** Global-mean SST and surface salinity drift is
   at least 30% smaller in magnitude. ~60%.
2. **The local SST runaway is reduced:** the maximum SST over the rollout is
   at least 1 °C lower. ~55%.
3. **One-step RMSE gets slightly worse,** by under 5%, a known trade-off of
   multi-step fine-tuning. ~60%.
4. **Deep variance damping gets somewhat worse.** Longer-rollout MSE rewards
   predicting the mean even more strongly, the same blurring GraphCast shows
   at long lead times. ~55%.

## How to run

The run waits for #984's training job:

```bash
CODE_COMMIT=<sha> CFG=experiments/nd2in1_rollout_curriculum \
  NAME=2026-11-13-nd2in1-rollout-curriculum AFTER=<#984 train job id> \
  EXTRA_ARGS="--resume_ckpt_path=/scratch/am16581/runs/2026-11-13-normalized-delta-2in1out/saved_nets/ckpt.pt" \
  bash experiments/normalized_delta_2in1out/submit_variant.sh
```
