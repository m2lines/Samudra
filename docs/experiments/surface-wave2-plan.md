<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Surface-initialized ocean adaptation: wave 2

Authorized on September 20, 2026. The target completion deadline is September 23 at
16:51:29 UTC (12:51 p.m. Eastern), including a report and targeted follow-ups needed
to answer the questions below. This implements the [wave-1 proposal](surface-wave1-results/index.md).

## Questions and fixed comparisons

| Arm | Trainable component | Data-order seed | Question |
| --- | --- | --- | --- |
| A | Initializer | 1729 | Can forecast-supervised initialization reduce the inferred/true-interior gap with fixed dynamics? |
| B | Evolution | 1729 | Is adapting dynamics to initializer error sufficient? |
| C | Both | 1729 | Does gentle joint adaptation retain useful gains? |
| D | Both | 1730 | Is the joint result sensitive to the adaptation data order? |

All arms start from the same wave-1 AR pretraining checkpoint and explicitly restore
the original shared initializer. SHA-256 fingerprints record the input files and
initial component weights and buffers. Frozen components retain both weights and
batch-normalization statistics; gradients still flow through frozen evolution to
train the initializer in A. D changes data order, not the initial model.

Use AdamW with learning rate 1e-5, weight decay 0.01, gradient clipping 1, BF16,
four GPUs and global batch eight (two per GPU). Train all six five-day transitions
from the outset. Retain full-variable forecast loss and a reconstruction coefficient
of 0.1 where the initializer is trainable (zero in B). Use the existing mean/std files.

Select checkpoints by equal-weight subsurface temperature and salinity normalized
MSE, averaging leads 5 through 30 days and the same validation origins in every arm.
The starting model is eligible. Save every 20 minutes; stop after six non-improving
checks, with at least one hour of training, or at 3.5 training hours. Each job has a
four-hour allocation. Log forecast and reconstruction losses separately. Save
optimizer state, epoch and cursor for recovery without changing the sample order.

Wave 1 selected on a full-variable score. Therefore a comparison against wave 1
alone cannot isolate learning rate. If needed, run a matched joint 1e-4 control with
this wave's selection rule, starting weights, loss and stopping protocol. Choose
follow-ups using validation and protocol gaps, without tuning to held-out outcomes.
The user has authorized targeted additional runs within the three-day study.

## Data and evaluation

Use `/scratch/jr7309/data/om4_onedeg_v3` on Torch. Preserve wave-1 splits:
training 1975-01-03 through 2013-10-04, validation 2013-10-05 through 2014-10-05,
evaluation 2014-10-10 through 2022-12-24. This uses OM4 five-day states and
interval-start tauuo/tauvo/hfds forcing, with 25 days of SSH/SST history.

Evaluate every selected model at the same 99 origins, six leads and three regions,
including inferred initialization, true initialization, inferred-state persistence,
monthly climatology, the frozen pretraining pair and the selected wave-1 joint pair.
Preserve channel metrics and paired origin-level T/S, SSH/SST and U/V errors and
physical second moments. Expected full outputs per arm: 8,316 channel rows and
64,152 origin/variable rows across all ranks. Record initial reconstruction quality,
forecast skill against common fixed references, and the true/inferred gap. Use paired
year/block resampling to describe uncertainty rather than treating adjacent origins
as independent. One alternate data-order seed does not establish pretraining-seed
reproducibility. Velocity amplitude is a diagnostic, not evidence of realistic circulation.

These are simulation feasibility experiments. They do not yet test real-observation
initialization, daily SSH/SST, ERA5 surface forcing, independent interior observations,
LLC, benefits of higher-resolution simulation data, or continuous multi-year rollout.
Those remain requirements of the observational prediction plan.

## Execution and provenance

Use `samudra.experiments.surface_adaptation` and
`scripts/submit_surface_adaptation.py`, with the immutable commit-named code overlay
and Rust-loader container used in wave 1. Keep eight Rust readers per rank, eight
allocated CPUs and 64 GiB host RAM per four-GPU job. Prepared GPU frame caching is
checked against the native reader. Disable NCCL P2P as qualified in wave 1.

First qualify all three gradient paths with 30 training steps and reduced validation
and held-out origin counts. Require successful completion and matching code commit
before production. Submit A/B together, C after A and D after B; this caps the wave
at eight GPUs. The shared allocation may be occupied by others. Monitor bring-up,
then check roughly hourly; recover failed jobs from checkpoints when safe. Send an
authorized Slack blocker alert if user intervention is needed.

Initial allocation envelope: 64 GPU-hours for the four arms plus 16 for qualification,
evaluation and recovery. Track every attempt and its actual Slurm GPU-hours. Record
additional follow-ups separately. Budget and deadline checks apply before submission;
queued jobs also need monitoring so they do not start beyond the deadline.

Remote results will live under `/scratch/jr7309/runs/2026-09-20-surface-wave2`;
qualification under the same path with a `-qualification` suffix. Per-arm manifests,
initialization fingerprints, validation progress, checkpoint states, aggregate and
origin metrics, utilization telemetry, and submission JSON provide reproducibility.
A completion marker is written only after successful final evaluation.

## Reproducing the paired comparisons

After collecting complete per-arm outputs (A/B/C/D), run:

```bash
uv run python -m samudra.experiments.surface_adaptation_analysis \
  --input /path/to/collected-wave2 --output /path/to/analysis
```

The analysis requires every approved arm and rejects reduced qualification outputs,
duplicate or missing origin/channel rows, nonfinite errors, mismatched data or
starting states, changed fixed references, and disagreement between per-origin and
aggregate metrics. It also checks selected validation scores against logged best
scores and requires runtime frozen-state verification where applicable. It writes
training summaries, grouped metrics, paired comparisons and an audit JSON. CSV inputs
can be stored with an added `.gz` suffix. Follow-up arms can be included using
`--arms A B C D E`; their manifests retain the changed treatment.

Each bootstrap draw resamples calendar-year blocks with replacement using the same
blocks for candidate and reference. Errors remain weighted equally by forecast origin,
including when years contain different numbers of origins. Positive reported percentages
mean lower candidate RMSE. Intervals are descriptive: only nine partly sampled years
are available, and the many region/lead/variable comparisons are not adjusted for
multiple testing. True-initialization comparisons describe an initialization gap;
they are not deployable baselines. The equal T/S summary averages normalized MSE,
with no mixed-unit physical RMSE reported.
