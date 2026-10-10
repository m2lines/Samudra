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

## Qualification and frozen coefficient

Beta qualification job **211902** completed successfully in 5m09s on a GB200.
Both modes passed real-batch forward/backward, date alignment, finite-gradient,
head-gradient and state reload/inference checks. Peak allocated GPU memory for
aligned qualification was 13,287,238,144 bytes. All 2,830 target frames passed
read-back verification. The first attempt (211900) exposed a missing October 5,
2013 training target; the isolated target cache now includes that native-derived
frame, with the parent cache and its normalization preserved.

The fixed coefficient for **both** treatments is **0.0340550269485343**, derived
only from four aligned training batches. Physical backbone gradient norms were
4.26–4.42, unweighted auxiliary norms 12.35–13.12, and gradient cosines
−0.016 to +0.0033. This establishes a working, initially modest auxiliary signal;
it does not establish an RMSE improvement. Seasonal qualification had identical
physical losses and unweighted auxiliary norms 11.85–12.50.

Full receipts are retained as `qualification-{aligned,seasonal}.json` under
`/projects/ny/lz1955/multiscale/jrusak/runs/current-ke-aux-20261010` on Beta.
Qualification source archive SHA-256:
`c26f781682cd8e69255c7ddc626e7056d58cdcc5a6bd9044a2121a8f3b217521`.
Target manifest SHA-256:
`9a56838fb2355d0cf428875a4a570b5fa4c9b3aefab1d7e2df96d0ab7f2432a7`.
The receipt records coefficient 0.01 used to check backward propagation before
calibration; the frozen production coefficient is the value above.

Validation: 126 relevant CPU tests passed (four CUDA/manual tests deselected),
and all changed-file pre-commit checks passed, including mypy and schemas.
Each production allocation additionally runs a two-epoch, four-rank debug
preflight before starting a fresh 70-epoch training process.

## Submitted first wave

| Treatment | Beta job | Seed | Initial state |
| --- | --- | --- | --- |
| Aligned subcell KE | 211905 | 15 | Pending, priority |
| Seasonal subcell KE | 211906 | 15 | Pending, priority |

Both jobs use source commit `2506c0805ad94d6f99903cd5c855fb27a2b424ca`.
The transferred production archive was verified with SHA-256
`9d564b8b5f69ed51f438649bbeaf19d22e1d8675ed57aff487ce3cd66bf80576`.
Online W&B is configured; initialization cannot be verified until scheduling.
Supervised scheduler watchers are active and authenticated, with one-hour startup
reminders and terminal-state callbacks to the originating chat. Completion will
trigger the matched evaluation and map work described above.
