<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Δ-norm 2→2 + step curriculum

An experiment record, not intended to merge. It trains Δ-norm 2→2 from scratch with a rollout-length curriculum,
matched by horizon to the one in Δ-norm 2→1 + step curriculum. It is an ablation: the same curriculum on the 2→1
and 2→2 objectives.

Δ-norm 2→2 is Yuan's normalized-delta recipe, PR #998. Each call takes 2 input states and predicts 2 chained
five-day steps.

## Why

- **Δ-norm 2→2** brings back the upper-ocean warming trend: +11.8 ZJ/yr at 0–700 m, against OM4's +9.3. Δ-norm 2→1
  manages only +0.3. But 2→2 overshoots at 700–2000 m, and one Kara Sea cell spikes to 49.7 °C in July 2015.
- **Δ-norm 2→1 + step curriculum** (PR #1000) gave the best one-step forecasts of any Δ-norm run (T −11%, S −10%
  against Δ-norm 2→1). But its eight-year rollout blew up from 2018.

This run asks whether the chained two-step objective, which seems to tie the model more tightly to the forcing, keeps
the curriculum's one-step gains while staying stable.

## The schedule

Each 2→2 call covers 10 days, so the schedule halves the call counts of the 2→1 version. The 5-day stage folds into
the first stage. Everything else matches `experiments/delta_norm_2to2`. The effective batch stays at 32 throughout,
using the trainer's per-stage `step_batch_size` and `step_gradient_accumulation_steps` (feature PR #999, cherry-picked
here).

| Epochs | Calls per rollout | Horizon | Batch per rank × accumulation |
|---|---|---|---|
| 1–44 | 1 | 10 days | 4 × 1 |
| 45–54 | 2 | 20 days | 4 × 1 |
| 55–61 | 4 | 40 days (Δ-norm 2→2's horizon) | 4 × 1 |
| 62–66 | 8 | 80 days | 2 × 2 |
| 67–70 | 16 | 160 days | 1 × 4 |

## Predictions, made before training

These are against Δ-norm 2→2, on the v2026-09 official 1° data, from the final-epoch EMA checkpoint.

1. **One-step skill improves:** mean one-step RMSE over depth for temperature and salinity is at least 5% lower.
   ~65%.
2. **The eight-year rollout is still unstable:** the hottest SST anywhere exceeds 60 °C at some point. ~55%.
3. **The Kara Sea cell (72.3°N 83.5°E) spikes harder** in July–August 2015 than Δ-norm 2→2's 49.7 °C. ~60%.
4. **The upper-ocean warming survives:** the 0–700 m OHC trend is at least 5 ZJ/yr. This is only scored if
   prediction 2 is wrong. ~40%.
5. **The rollout validation score is worse:** at the last epoch, the best rollout validation score is higher (worse)
   than Δ-norm 2→2's. ~70%.

## How to run

On NYU Torch, on 8 GPUs with `ACCUM=1`:

```bash
CODE_COMMIT=<this branch's pushed commit> CFG=experiments/delta_norm_2to2_step_curriculum \
  NAME=2026-11-16-delta-norm-2to2-step-curriculum GPUS=8 ACCUM=1 bash ~/nd_submit_variant.sh
```
