<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Longer early-data comparison with RMSE selection

Seven fresh runs test whether the earlier-OM4 gains persist with 50% more
training, how their validation components evolve, and how much U-global varies
between seeds. The historical [four-arm report](early-fine-wave-2026-10-07.md)
and [cooldown report](early-fine-cooldown-2026-10-08.md) remain unchanged.

Contents: [Runs](#runs) · [Selection](#selection) · [Diagnostics](#diagnostics) ·
[Execution](#execution).

## Runs

| Literal model name | Seed | Recent OM4 1° updates | Earlier OM4 updates | Observation updates | Total |
|---|---:|---:|---|---:|---:|
| U-global | 1729 | 3000 | 0 | 3000 | 6000 |
| U-global-seed1730 | 1730 | 3000 | 0 | 3000 | 6000 |
| U-global-repro4k | 1729 | 2000 | 0 | 2000 | 4000 |
| U-multitask-early | 1729 | 1500 | 1500 at 1° | 3000 | 6000 |
| U-multitask-early-latent | 1729 | 1500 | 1500 at 1° | 3000 | 6000 |
| U-multitask-early-fine | 1729 | 1500 | 1500 at ¼° | 3000 | 6000 |
| U-multitask-early-fine-latent | 1729 | 1500 | 1500 at ¼° | 3000 | 6000 |

U-global uses the original ~63.0M-parameter U-Net with recent OM4 and observations.
The early arms replace half its OM4 updates with 1958–1974 examples. The fine
arms use the existing fine encoder/decoder around the global 180×360 processor;
the latent arms add 10 recurrent channels on all tasks. Architecture, physical
losses, AdamW settings, eight-example accumulation, input adapters, masks and
normalization follow the original report. All models start fresh; qualification
weights never enter production. LR stays **1e-4** for both tasks, without warmup
or cooldown. Recent OM4 coverage and observation splits are unchanged.

The progressive mixture is stretched to the new budget: cumulative observation
updates at fraction f are floor(M*(f/4+3*f²/4)), with M=3000 (2000 for repro4k).
OM4 slots alternate recent and early in the early arms. At total updates
1500/3000/4500/6000 the long runs have 328/937/1828/3000 observation updates.
Thus the long run's update 4000 is **not** an original-schedule reproduction.
U-global-repro4k preserves the original 4000-update training schedule and seed,
but uses the new reporting/selection criterion and the common Torch runtime.
Comparison with the historical Beta run includes hardware/runtime variability.
The two long U-global seeds give only an initial estimate of run variability,
not a reliable confidence interval for every arm or a significance test.

## Selection

The primary score is the equal mean of five RMSE ratios, each divided by the
same frozen validation seasonal-climatology RMSE: SST, SSH-derived geostrophic
velocity, EKE, OHC 0–700 m and OHC 700–2000 m. Lower is better; the climatology
control scores 1. No spectral calculation or spectral qualification contributes
to this wave. This is a new explicit protocol; old protocols retain their behavior.

RMSE uses full physical fields without deseasonalization or detrending. EKE is
formed from velocity anomalies about the evaluated cohort's time mean, not a
seasonal or detrended decomposition. Surface errors average day 5/15/30; OHC
uses calendar-month means. The same nine validation origins, global masks and
strict velocity support are retained. The frozen five climatology denominators
come unchanged from the original U-global selection reference; source hash and
new reference hash are recorded in paths.json. Held-out test metrics never
select checkpoints. Training losses remain unchanged.

## Diagnostics

Validate every 100 total updates and at saved observation-exposure milestones.
Plot the primary blend, all five physical RMSE components and normalized ratios,
OM4 retention, and per-task training losses against both total updates and
observation updates. Also log SST bias, area-weighted uncentered anomaly
correlation and anomaly RMS amplitude ratio using a shared training-only seasonal
climatology, alongside the existing SST variance-map and depthwise T/S RMSE/bias.
These short-cohort diagnostics do not establish long-term temporal behavior.

Save diagnostic checkpoints at 500-observation-update intervals, plus initial,
selected and latest states. Milestones retain weights, manifest, task counts and
RNG metadata; they deliberately omit optimizer tensors. The latest checkpoint
retains the full optimizer/RNG resume state. Use these milestones for matched
rollout and initializer maps/profiles as training proceeds. Report fixed endpoints
alongside validation-selected results and each selected model's own initialized
persistence. Selected held-out evaluation uses the same 96 monthly origins.

## Execution

Target: Torch RTX6000, pinned previous container/runtime, one GPU, 16 CPUs and
192 GB RAM per run, up to four concurrent tasks. Existing datasets are reused.
Scratch headroom at preflight was about 86 GB; estimated additional persistent
outputs are approximately 40–50 GB with weights-only diagnostic milestones.
Node-local source caches are separate from persistent scratch output.

CPU data/source audit precedes all seven fitting/gradient, exact-I/O, deterministic
serialization-replay and measured-throughput qualifications. Production depends
on all seven qualifications passing and starts from fresh random weights.
GPU utilization is recorded throughout, including validation. Preserve failed
attempts; diagnose low utilization before any routine resubmission. No new seeds,
models or scientific protocol changes beyond this table are included.

Status at October 10 **21:18 ET**: CPU audit **19555071** and all seven
qualification tasks **19555072** completed successfully. Saved gates confirm
bitwise exact deterministic serialization replay, exact I/O, matching producer,
seed and selection-reference contracts, and acceptable measured throughput.
All 73 targeted tests and repository commit hooks passed.

Production/evaluation array **19555073** has four early-data models running
(156–361 total updates at this snapshot); the three U-global controls remain
queued behind the four-task concurrency limit. Each running arm has completed
its first RMSE-only validation. These are **startup curves**, not evidence of
relative model quality; the controls have not trained yet.

![RMSE validation against total and observation updates](artifacts/early-rmse-6k-2026-10-10/progress/validation-blend.png)

[All five normalized components](artifacts/early-rmse-6k-2026-10-10/progress/validation-components.png) ·
[Numeric validation history](artifacts/early-rmse-6k-2026-10-10/progress/validation.csv).
Every plotted blend was independently checked against its saved five components
and frozen climatology denominators. Curves contain eight completed validations
through total update 300 (coarse) / 100 (fine), rather than interpolated results.

The initial batch telemetry used physical Slurm GPU IDs, whereas Torch's cgroups
renumber allocated GPUs. Failed telemetry files are retained; separate read-only
observers now record the actual allocated GPU UUIDs. Training, checkpoints and
qualification contracts were not changed. A launcher fix in the reporting branch
applies to future submissions; this wave retains the pinned producer below.

Pinned producer: `eae2b5c56da44eb3686a6eaf72a80aa419a4b5a4`.
Cluster root: `/scratch/jr7309/runs/2026-10-10-early-rmse-6k`.
[Exact run/data/source contract](artifacts/early-rmse-6k-2026-10-10/paths.json) ·
[Slurm requests and dependencies](artifacts/early-rmse-6k-2026-10-10/submission.json).
Runner: `scripts/run_early_fine_rmse_wave.py`; Slurm entry:
`scripts/slurm_early_fine_rmse.sbatch`. The source commit remains fixed even when this report is updated.
