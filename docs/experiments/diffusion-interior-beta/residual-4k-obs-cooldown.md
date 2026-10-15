<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

**Execution update:** The user prioritized this over the main residual run and authorized a shorter cooldown for the deadline. The first run uses **500 observation-only updates**, with the entire linear LR decay over those 500 updates and the same 100-update EMA half-life. Final counts are 7,721 OM4 + 4,500 observation updates. The main residual run is checkpointed and paused while its allocation is reused, then resumed. The original 1,000-update reference recipe is documented below.

# Observation-only cooldown from the residual 4k checkpoint

Test whether a short observation-only cooldown improves the residual model's
forecast accuracy and visible grain while retaining its comparatively stable
annual evolution. This is an additional continuation from the frozen **4,000
observation + 7,721 OM4 update** checkpoint, not a change to the main residual
training run. The direct arm remains cancelled at the user's request.

## Training protocol

Fork the exact immutable checkpoint containing raw model weights, Adam state,
RNG state and task counters. Preserve the optimizer moments and sample-stream
position; do not restart Adam or initialize training from averaged weights.
Record the source checkpoint hash and verify the restored state before training.

Run **1,000 observation-only optimizer updates**, retaining the existing
observation objective and all its auxiliary losses and penalties. This branch
performs **no OM4 updates and no OM4 replay updates**. Keep the parent's batch
size, data, masks, diffusion sampling and architecture settings unchanged.
Linearly decay the learning rate from **1e-4 to 1e-6**, using
`1e-4 * (1 - 0.99 * cooldown_update / 1000)` for updates 1 through 1,000.

Reset parameter EMA to the raw source weights at the fork, with **100-update
half-life** (`decay = 2**(-1/100)`). Average parameters after each optimizer
update and copy buffers from the raw model. Save raw weights, EMA, Adam and RNG
state together so restarting does not reset the averaging history. The final
cumulative task counts are **5,000 observation + 7,721 OM4 updates**.

This follows the recommended observation-only recipe in the
[OE multitask cooldown comparison](https://github.com/m2lines/Samudra/blob/experiment/observation-stability-long-training/docs/experiments/observation-d/cooldown-standardization-2026-10-14.md#recommended-common-cooldown).
Its [runner at the verified source commit](https://github.com/m2lines/Samudra/blob/1e90c7b1dbea5954ee658c57b6f02112e84dc56f/scripts/run_observation_cooldown_comparison.py)
preserves Adam and resets EMA. That experiment's source was 4,000 **total**
updates, whereas this source is 4,000 **observation** updates; the training
exposures and model families are not matched.

## Evaluation and interpretation

Evaluate the fixed final **raw and EMA** weights alongside the unchanged
pre-cooldown residual checkpoint. Use the existing validation, native OM4 and
annual observation evaluation cohorts and manifests, with **eight diffusion
members and 32 denoising steps**. Annual comparisons retain the same three
January origins in **2015, 2018 and 2021** and the same day-30/day-365 metric
construction. Report member and ensemble-mean maps, annual error growth and
latent growth, rather than judging smoothness from the ensemble mean alone.

Keep the raw-versus-EMA comparison separate from the pre-versus-post cooldown
comparison. Additional observation training, changed task mixture and learning
rate decay all occur together; this experiment does not isolate their individual
causal effects. Check OM4/interior diagnostics for regressions after removing
OM4 supervision. A smoother image alone does not establish improved accuracy
or calibration.

## Timing and execution

Target initial rollout results before **9am Eastern on October 16, 2026**.
The provisional estimate is **11–12 hours of training, plus queue time and
evaluation**. This is a target, not a guarantee; checkpoint availability,
qualification, allocation delays and measured throughput determine feasibility.
Record actual job IDs, source hashes, timing and completion receipts when the
branch runs. The main residual training job is temporarily paused to prioritize this branch
in its existing allocation, then resumes its original budget and final cooldown.

## Requested grain diagnostics and viewer comparison

The cooldown report will include error maps alongside the usual rollout and reconstruction metrics. Show the reference, ensemble mean, a fixed individual draw, their respective errors, and spread; use common physical color limits across methods and explicit clipping. Include initialization where a correctly aligned target exists, and rollout leads 30 and 365 days (additional saved leads may be exposed in the viewer). Keep member noise distinct from mean bias and do not infer spatial coherence from pointwise calibration alone.

Use existing U-global mixed and observation-only exports as context baselines. These need not match the diffusion architecture or training exposure; label checkpoint selection, update counts, normalization, grid, origins and temporal aggregation differences. Reuse common origins and verification bins where available rather than silently mixing dates or instantaneous and averaged fields.

The saved diffusion annual exports include eight draws at five-day leads through day 365, initial readouts, and selected full-state mean fields. Native OM4 exports include aligned initialization targets. Observation interior monthly targets must not be presented as instantaneous day-zero truth. Add saved-array entries to the existing results viewer with explicit raw/EMA and mean/member labels; missing channels or unavailable reference fields should remain unavailable rather than fabricated.
