<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Δ-norm 2→1 + step curriculum

An experiment record, not intended to merge. It trains Δ-norm 2→1 (Yuan's normalized-delta weighting with 2 input
states and 1 output step, PR #984) from scratch with a rollout-length curriculum, instead of the fixed 8-call
(40-day) rollout Δ-norm 2→1 uses for all 70 epochs.

## Why

A rollout of `n` calls feeds each prediction back as input and backpropagates through the whole chain, so the loss
sees errors that grow within `n` steps. Two earlier results motivate this run:

- **Δ-norm 2→1** forecasts one step far better than Samudra 2, but over eight years it develops local SST hot spots:
  one cell reaches 40.0 °C, where OM4 never exceeds about 33 °C.
- **Δ-norm 2→1 + curriculum** (PR #987) fine-tuned Δ-norm 2→1 for 15 epochs on 12 → 24 → 36-call rollouts. That
  removed the hot spots (peak 36.6 °C) for about 2% in one-step RMSE, but made the global cold drift and the
  ocean-heat-content trend worse.

This run asks whether building the long horizon into training from the start, as GraphCast and Keisler do, gets the
stability without a separate fine-tune.

## Prior art

- **GraphCast** (Lam et al. 2023) trains on one 6-hour step for most of training, then lengthens the rollout one step
  at a time from 2 to 12 (3 days) in a short final phase at a low learning rate.
- **Keisler** (2022) lengthens the training rollout in stages, up to 12 steps.

## The schedule

Everything except the rollout length matches `experiments/normalized_delta_2in1out`. The effective batch stays at
32 throughout; when rollouts get long, a smaller micro-batch with gradient accumulation keeps memory in bounds. This
uses the trainer's per-stage `step_batch_size` and `step_gradient_accumulation_steps`, added in a separate feature PR.

| Epochs | Calls per rollout | Horizon | Batch per rank × accumulation |
|---|---|---|---|
| 1–29 | 1 | 5 days | 4 × 1 |
| 30–44 | 2 | 10 days | 4 × 1 |
| 45–54 | 4 | 20 days | 4 × 1 |
| 55–61 | 8 | 40 days | 4 × 1 |
| 62–66 | 16 | 80 days | 2 × 2 |
| 67–70 | 32 | 160 days | 1 × 4 |

The cosine learning-rate schedule spans all 70 epochs, so the long-rollout stages run at a low learning rate, like
GraphCast's fine-tuning phase. Summed calls × epochs are about 65% of Δ-norm 2→1's.

## Predictions, made before training

Against Δ-norm 2→1, on the v2026-09 official 1° data, from the final-epoch EMA checkpoint:

1. **Hot spots stay away:** the hottest SST anywhere in the eight-year rollout stays below 37 °C. ~55%.
2. **One-step skill is about the same:** mean one-step RMSE over depth for temperature and salinity is within ±3%. ~60%.
3. **Global SST drift does not improve:** its eight-year magnitude is at least 0.18 °C (Δ-norm 2→1: 0.20 °C). ~60%.
4. **The ocean-heat-content trend is still missed:** full-depth trend below 5 ZJ/yr (OM4: 9.8). ~80%.
5. **Training is cheaper:** less total training wall time than Δ-norm 2→1 on the same 8 GPUs. ~65%.

## How to run

On NYU Torch, with the shared submit script. The per-stage accumulation assumes 8 ranks, so run on 8 GPUs with
`ACCUM=1`. On 4 GPUs the script passes `--gradient_accumulation_steps=2`, and the config validator rejects the
schedule because its first stage says 1.

```bash
CODE_COMMIT=<this branch's pushed commit> CFG=experiments/nd2in1_step_curriculum \
  NAME=$(date +%F)-nd2in1-step-curriculum GPUS=8 ACCUM=1 bash ~/nd_submit_variant.sh
```
