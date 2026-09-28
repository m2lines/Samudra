<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Initialization gaps and mixed pretraining with an observation-only finish

Authorized 28 September 2026. Goal: diagnose and improve initialization at missing
surface observations, learn missingness explicitly, and test task-conditioned
mixed pretraining followed by observation-only fine-tuning. This is a new wave;
the preceding three-day investigation is complete. No new long-rollout training,
dense-versus-observable-only OM4 ablation, probabilistic architecture, or resolution
change is included. Annual autoregressive evaluation remains diagnostic.

## Fixed choices and controls

Use the existing one-degree grid, observation products and chronological splits,
original OM4 forcings plus the ERA5 adapter, shared observation-derived scaling,
and v3 integrated-plus-spectral observation validation. One seed (1729), effective
batch eight, AdamW 1e-4, weight decay 0.01, gradient cap one. Keep the current
forecast and monthly T/S reconstruction objectives and one-month training horizon.
The small initializer has 31,243,306 parameters; the D processor has 31,631,677;
the forcing adapter has 387, totaling 62,875,370 before optional task adapters.
This requires fresh pretraining, not conversion of historical BatchNorm weights.

| Literal model name | Missingness policy | Ordering | OM4 / observation updates |
| --- | --- | --- | --- |
| Small legacy scratch | Existing climatology fill and unconditional surface copy | Observation only | 0 / 16,000 |
| Small masked scratch | Zero input placeholders, valid-only copy, learned completion | Observation only | 0 / 16,000 |
| Small masked sequential | Learned completion | OM4 then observations | 8,000 / 8,000 |
| Small masked mixed-finish | Learned completion, shared inputs | Mixed prefix then 2,000 observation-only updates | 8,000 / 8,000 |
| Small conditioned mixed-finish | Learned completion, task-specific input adapters | Same mixed prefix and observation-only finish | 8,000 / 8,000 |
| Small conditioned mixed | Learned completion, same task adapters | Mixed through the end | 8,000 / 8,000 |

All arms use deterministic per-task sample streams. Mixed-finish moves all OM4
exposure into the first 14,000 updates (8,000 OM4 plus 6,000 observations), retaining
2,000 observations at the end. Compare raw matched endpoints as well as validation
selections. For finish arms, report a best-within-finish checkpoint separately
from best-anywhere: otherwise an earlier mixed checkpoint could silently replace
the requested observation-finished model. Retain the boundary and early finish
checkpoints. Scratch 8k and 16k endpoints remain available.

## Frozen-checkpoint gap diagnosis

Use the three completed Samudra2 models. Audit per-variable, per-frame input
validity separately from scoring coverage. Perturb only missing initial SST/SSH
cells with alternative fixed fills, preserving observed cells and all hidden
channels; separately perturb the missing values across the input history to
measure encoder sensitivity. Use normalized zero and a training-only spatial OM4
monthly climatology alongside the unchanged baseline. The OM4 fill is explicitly
a simulation-informed intervention, including when applied to scratch, not an
observation-only forecast claim. No future ocean observations enter initialization.
Verify baseline reproduction and checkpoint hashes before interpreting differences.
Report first-step and 30/90/180/365-day sensitivity, latitude-resolved errors and
maps. These are causal sensitivity probes, not checkpoint selection or new models.

## Missingness learning

Represent unavailable normalized input values by zero plus existing validity
channels. Copy only available surface observations to the initial state; elsewhere
keep network outputs. Missing labels remain missing. Land masking stays separate.
The old discarded surface head is not a trained completion estimator, so corrected
models train anew. Structured artificial gaps persist across input history and
include spatial blocks and bands. Preserve uncorrupted targets for scoring hidden
known observations, without using those targets as forecast inputs.

OM4 pretraining additionally uses observation-training coverage masks, retaining
real OM4 surface targets where hidden. Observation-only completion uses genuinely
available training observations, including available polar observations, with
full-grid wet-area weighting for this new auxiliary task. Its scored forecast
objective remains on the original 60S–60N domain. Add 0.1 times normalized masked
surface reconstruction error at the two initialized history times; do not invent
labels in naturally missing observational cells. Record coverage and loss terms.
Qualification must verify copy invariance, missing-head gradients, no target
leakage, nonzero gradients through initialization/dynamics, and exact resume state.

## Task conditioning

Use separate identity-initialized 1x1 input adapters for OM4 and observations at
the initializer and processor entrances, sharing the expensive backbones. Explicit
source labels select adapters; labels are not inferred from missingness or forcing.
The observation adapter is always selected for observation forecasts. Keep source
forcings unchanged. Conditioning ablation uses the same architecture/backbone
initialization and sample stream with task adapters disabled. No physical meaning
is imposed on otherwise unconstrained latent velocity/deep-state channels.

## Qualification, execution and reporting

First qualify CPU masking/schedule/checkpoint contracts and GPU fitting/resume,
then execute production. Qualification is not scientific success. Production and
evaluation must record immutable source/checkpoint/data hashes and real completion
markers; preserve failures. Prefer RTX capacity, use H200 if scheduling requires it.
Operational planning estimate: 120 allocated GPU-hours including qualification,
diagnostics, production, evaluation and retries. This is not a user-imposed cap:
the user authorized feasible GPU spend and update counts within the three-day
wall-time budget. The controlling deadline is 1 October 2026 at 18:00 UTC.
Qualify throughput and record actual allocated use, including retries; no
assumption that parameter reduction produces an equal speedup. At the deadline,
report any partial results honestly.

Keep the existing hourly interruptible monitoring style; do not restart the old
timer. Notify Jesse on Slack for genuine blockers under the existing authorization.
Push progress and results to codex/d-observation-pilot and update draft PR 892's
contents. Final report includes literal model definitions, component metrics,
learning curves, completion/missingness audits, full-grid initializer and forecast
maps (1:1 or 2:2 pixels per cell), and accumulated GPU-hours. Qualitative improvement
in unobserved regions is not a claim of validated polar accuracy. Existing annual
test cases remain exploratory; visual sanity does not replace observation metrics.

Completion diagnostics use the same nine validation origins for every model and
checkpoint. In addition to intact histories, apply fixed artificial blocks and
hide the polar caps (>60 degrees absolute latitude) across the entire input
history. Score only genuinely observed cells that were artificially hidden, with
physical-unit SST/SSH RMSE and bias separated into polar and nonpolar regions.
Report support counts; absent polar SSH labels produce null scores, not zeros.
Compare learned completion against observation-training monthly climatology and
normalized-zero fill. These diagnostics do not change checkpoint selection.


### Throughput and final runtime decision

H200 qualification 18724971 measured the last 20 updates of each task at
2.320 seconds/OM4 update and 4.140 seconds/observation update. Retain the original
16k-total comparisons in the table above. Scratch jobs receive 20-hour allocations
(19.5-hour internal limits); other arms receive 16 hours (15.5-hour internal limits).
Expected update time across all six arms is 94.2 GPU-hours, before overhead.
Requested production caps total 104 GPU-hours. Track all attempts and evaluation
against a 120-hour planning ceiling, retaining recovery room where actual usage
permits. The controlling constraint is completion within the three-day wall-time
budget (1 October, 18:00 UTC), not the assistant's earlier 100-hour planning limit.
The user explicitly clarified that feasible GPU spend/update counts should be
chosen to meet that wall-time budget. No update counts, batch size, scientific
comparison, or qualification contracts changed. Qualification weights are not
reused for production initialization.
