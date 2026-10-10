<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Samudra-current: auxiliary subcell kinetic energy

Experiment/results record, not intended to merge. First wave: two fresh seed-15
runs, aligned and seasonal targets. No later seeds are authorized by this wave.

Baseline: stochastic-depth study control, Beta job 210664, W&B `uaei2vbr`,
source `a20c44484`. Two input states, two output states, four autoregressive calls,
70 epochs, InstanceNorm, bfloat16, no stochastic depth, effective batch 32,
Rust reader and `tau_hfds` forcing. Preserve original optimizer, sampler,
dynamic physical loss (limit 20), normalization, splits and fixed final-epoch
selection. The branch starts at the exact baseline source. The source data path
now resolves to project storage; use the verified existing data and normalization.

Two linear 1x1 outputs on the final full-resolution U-Net features predict the
two future surface subcell-variance maps for each call. No fine-grid inputs are
provided. The head is unused at inference and its initialization preserves the
core model and sampler RNG. Eight future dates receive area-weighted MSE,
averaged over output slots, examples and calls. The unchanged physical objective
sums over four calls. Auxiliary loss is added separately; physical channel
logging and DynamicLoss state updates exclude it.

Targets are the prior experiment's spherical-overlap aggregation of quarter-degree
OM4 five-day-mean surface velocities:
`q = 0.5 * (mean(u*u+v*v) - mean(u)**2 - mean(v)**2)`.
This is spatial subcell variance, not all unresolved KE. The target uses a
training-only standardized `log1p(q/scale)` transformation, common native U/V
wet support, and >=50% native wet coverage. The original 2,829 frames are copied and verified, and the additional October 5,
2013 training target is computed from the same quarter-degree store. The
original training-only normalization is frozen. All 2,830 frames and metadata must
pass the existing READY contract, time coverage, coordinate and wet-mask checks.
Seasonal targets are each cell's training-calendar-month average in transformed
space. No held-out values enter either target transform.

Qualification uses four fixed training batches at the common initialization.
Choose one coefficient using aligned targets so its median backbone gradient
norm is 10% of the physical gradient norm; freeze it and use it for both modes.
Record both gradient norms and cosine, date alignment, memory, target hashes,
forward/backward and serialization checks. This initial gradient calibration is
not an adaptive coefficient or a guarantee of later gradient proportions.

Primary checkpoints: final epoch-70 EMA; final raw as a sensitivity analysis.
Use identical evaluation code for historical control and both treatments.
Collect fixed area-weighted normalized RMSE and physical per-channel/depth RMSE,
quarterly starts during 2015-2021 through 360 days (5/30/90/180/360-day reports),
and a matched continuous 2014-2022 rollout for stability and time-mean errors.
Keep trajectory RMSE separate from RMSE of time-mean maps. Produce control,
treatment and signed RMSE-difference maps using common wet support and scales;
report regional and depth breakdowns, bias maps, and paired time-block
uncertainty. Backfill matched observation metrics using original wet masks.
Test maps are descriptive; later hyperparameter choices use validation only.

Production prefers four-GPU Beta hosts with batch four and accumulation two,
the original pinned ARM64 PhysicsNeMo image and original Rust extension.
Qualification runs on beta_test. Follow-up uses a supervised local accounting
watcher plus a timed reminder because Beta's scheduler lacks a suitable CPU-only
callback queue. Training completion triggers evaluation/reporting in this chat.
