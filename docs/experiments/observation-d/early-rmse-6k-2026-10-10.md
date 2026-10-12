<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Longer early-data comparison with RMSE selection

The initial seven fresh runs and four subsequent auxiliary reproductions completed. **Longer U-global training helps:** the selected
6k model's held-out RMSE blend is 0.8423, versus 0.9082 for the 4k reproduction
(7.3% lower). Its second seed scores 0.8503. **At 6k, best-checkpoint gains from
early data are small**, although late-window means favor the fine-task variants.
The subsequently completed [matched 4k reproductions](early-rmse-4k-reproduction-2026-10-11.md)
repeat the earlier-data advantage: 2.7–3.0% lower validation and 2.6–3.7% lower
test RMSE than U-global-repro4k. The evidence therefore supports a budget- and
checkpoint-dependent effect. See the [quarter-degree follow-on plan](quarter-global-plan-2026-10-10.md).

The historical [four-arm report](early-fine-wave-2026-10-07.md) and
[cooldown report](early-fine-cooldown-2026-10-08.md) use their original selection
protocols. This wave uses RMSE-only selection throughout.

Contents: [Results](#results) · [Runs](#runs) · [Selection](#selection) · [Diagnostics](#diagnostics) ·
[Execution](#execution) · [Auxiliary reproductions](#auxiliary-loss-reproductions).

## Results

Lower is better. Validation uses nine origins and its frozen climatology
denominators. The held-out blend uses all 96 test origins and the same held-out
seasonal-climatology denominators for every model; that control scores 1.
Test data never select weights. Own persistence holds each selected model's
initialized ocean state fixed, without applying its evolution processor.

| Model | Selected total update | Selected validation blend | Endpoint validation blend | Held-out blend | Own persistence blend |
|---|---:|---:|---:|---:|---:|
| U-global | 5,200 | 0.8241 | 0.8689 | 0.8423 | 0.9391 |
| U-global-seed1730 | 5,800 | 0.8419 | 0.8633 | 0.8503 | 0.9513 |
| U-global-repro4k | 4,000 | 0.8861 | 0.8861 | 0.9082 | 0.9829 |
| U-multitask-early | 5,500 | 0.8250 | 0.8648 | 0.8405 | 0.9216 |
| U-multitask-early-latent | 5,200 | 0.8315 | 0.8682 | 0.8300 | 0.9099 |
| U-multitask-early-fine | 5,500 | 0.8160 | 0.8616 | 0.8349 | 0.9160 |
| U-multitask-early-fine-latent | 5,600 | 0.8329 | 0.8360 | 0.8534 | 0.9681 |
| U-aux01-6k | 4,900 | 0.8205 | 0.8596 | 0.8538 | 0.9868 |
| U-aux01-anomaly-6k | 5,700 | 0.8190 | 0.8285 | 0.8349 | 0.9299 |

[Initial-wave physical components and persistence controls](artifacts/early-rmse-6k-2026-10-10/final/scores.csv) ·
[Combined components including auxiliary reproductions](artifacts/auxiliary-rmse-2026-10-11/scores.csv) ·
[Independent completion and checkpoint audit](artifacts/early-rmse-6k-2026-10-10/final/verification.json).

The two 6k U-global seeds improve selected validation by 5.0–7.0% and held-out
blend by 6.4–7.3% relative to the 4k reproduction. Even their fixed 6k endpoints
improve validation, by 2.0–2.6%. The stretched mixture changes exposure at each
intermediate update; this supports the longer training recipe, not a claim that
6,000 is optimal or that the benefit comes solely from extra observation updates.

U-multitask-early-fine beats seed-1729 U-global by 0.98% on validation and 0.87%
on test. The two U-global seeds differ by 2.17% and 0.96%, respectively. Two seeds
cannot estimate significance reliably, but that scale argues against declaring
a robust early-fine win. The early-latent arm has the lowest test blend while
losing to U-global on validation. That held-out ranking does not establish a
validation-selected winner. Latent channels do not consistently improve the
paired arms. The subsequently authorized [quarter-degree plan](quarter-global-plan-2026-10-10.md#choosing-the-variants)
will use matched 30/365-day outcomes to choose follow-on variants explicitly as
exploratory model development, while leaving checkpoint selection unchanged.

![Budget and seed controls](artifacts/early-rmse-6k-2026-10-10/final/u-global-budget-and-seed-controls.png)

![RMSE validation against total and observation updates](artifacts/early-rmse-6k-2026-10-10/progress/validation-blend.png)

[All five normalized components](artifacts/early-rmse-6k-2026-10-10/progress/validation-components.png) ·
[Numeric validation history, including physical units](artifacts/early-rmse-6k-2026-10-10/progress/validation.csv).
All 547 saved validation blends were independently recomputed from their five
components and the frozen reference.

The best-checkpoint table does not summarize all late-training behavior.
Across the final five scheduled validations (5,600–6,000), early-fine and
fine+latent average **0.8524 and 0.8502**, versus **0.8830 and 0.8808** for the
two U-global seeds. This descriptive diagnostic suggests a late-training
advantage despite the close minima; the five correlated points are not
independent trials and do not replace the prespecified selection criterion.
The matched 4k follow-up reports this diagnostic alongside minima/endpoints.

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

Validation ran every 100 total updates and at saved observation-exposure
milestones. The blend is plotted against both total and observation updates;
component, dynamics and retention curves use total updates. SST diagnostics use
the shared training-only seasonal climatology. These short-lead, reinitialized
cohorts do not establish long-term autonomous dynamics.

The models still damp SST anomalies: the six long-run selected models retain
only 0.66–0.72 of the observed day-30 anomaly RMS, with cold biases of
0.15–0.28°C. U-global's day-30 anomaly correlation is 0.659, versus 0.592 for
the 4k reproduction. Lower RMSE has not removed the amplitude deficit.

All selected models improve their overall held-out blend over own persistence,
but **EKE RMSE worsens**: approximately 0.0247–0.0253 m²/s² versus persistence's
0.02255 m²/s². Better SST, velocity and heat-content errors do not imply that
every aspect of variability improves.

![SST dynamics and OM4 retention](artifacts/early-rmse-6k-2026-10-10/progress/validation-dynamics.png)

![Per-task training objectives](artifacts/early-rmse-6k-2026-10-10/final/training-loss.png)

[Monthly temperature/salinity error profiles](artifacts/early-rmse-6k-2026-10-10/final/heldout-depth-errors.png) ·
[Profile numbers, including own persistence](artifacts/early-rmse-6k-2026-10-10/final/heldout-depth-errors.csv) ·
[Training-loss bins](artifacts/early-rmse-6k-2026-10-10/final/training-loss.csv).
Fine and coarse early-OM4 objectives differ, so their loss magnitudes are not
direct measures of relative model quality. The observation objectives are shared.

The day-30 maps compare every selected model with observations and initialized
surface persistence, using the same January/July 2022 examples and common
color limits. Persistence is identical across these models on the plotted
SST/ADT support; their inferred subsurface states and heat-content persistence
still differ. Anomalies subtract the shared training seasonal climatology.
These are five-day means from short rollouts, not annual endpoints.

| Case | SST | ADT / SSH |
|---|---|---|
| January anomalies | [Map](artifacts/early-rmse-6k-2026-10-10/maps/day30-2022-01-anomalies-sst.png) | [Map](artifacts/early-rmse-6k-2026-10-10/maps/day30-2022-01-anomalies-adt.png) |
| July anomalies | [Map](artifacts/early-rmse-6k-2026-10-10/maps/day30-2022-07-anomalies-sst.png) | [Map](artifacts/early-rmse-6k-2026-10-10/maps/day30-2022-07-anomalies-adt.png) |
| July absolute fields | [Map](artifacts/early-rmse-6k-2026-10-10/maps/day30-2022-07-fields-sst.png) | [Map](artifacts/early-rmse-6k-2026-10-10/maps/day30-2022-07-fields-adt.png) |

Diagnostic weights are preserved every 500 observation updates, alongside
initial, selected and latest states. Milestones include weights, manifest,
counts and RNG metadata; only latest checkpoints include optimizer state.
The profiles linked above are monthly forecast-error profiles, not new
initializer-state or annual-rollout evaluations.

## Execution

All jobs completed successfully by October 11 **05:49 ET**: CPU audit **19555071**,
seven qualification tasks **19555072**, and training/evaluation array **19555073**.
Each run used one Torch RTX6000, 16 CPUs and 192 GB RAM, with at most four
concurrent tasks. Actual Slurm allocation totals are **33.0325 GPU-hours**,
including qualification, initialization, training and evaluation; no retries.

All 73 targeted training tests and commit checks passed. Production used fresh
weights after fitting, exact-I/O, deterministic replay and throughput gates.
The final collector verified all expected training events and task exposures,
finite losses/gradients, fixed LR, seeds, producer, independently hashed selected
checkpoints, and complete 96-origin evaluation records. Reporting changes did
not alter the pinned training producer.

Pinned producer: `eae2b5c56da44eb3686a6eaf72a80aa419a4b5a4`.
Cluster root: `/scratch/jr7309/runs/2026-10-10-early-rmse-6k`.
[Exact run/data/source contract](artifacts/early-rmse-6k-2026-10-10/paths.json) ·
[Slurm requests and dependencies](artifacts/early-rmse-6k-2026-10-10/submission.json).
Runner: `scripts/run_early_fine_rmse_wave.py`; Slurm entry:
`scripts/slurm_early_fine_rmse.sbatch`. The source commit remains fixed even when this report is updated.

## Auxiliary-loss reproductions

**Both 6k reproductions completed training and all 96 held-out origins.** Two fresh additions,
**U-aux01-6k** and **U-aux01-anomaly-6k**, use 3,000 recent global OM4 updates plus
3,000 observation updates, the same stretched progressive mixture as U-global,
seed 1729, fixed LR 1e-4, and the unchanged five-component RMSE-only selection.
They start from scratch; they are not continuations of the original 4k runs.

U-aux01 uses a training-only processor head to predict future subcell surface
velocity variance from the original quarter-degree OM4 targets, with accessory
weight 0.01. U-aux01-anomaly predicts the target after subtracting its per-cell
training seasonal mean. Both retain the global 1° physical evolution task and
62,967,809 total parameters. Full definitions, the original RMSE-only ranking,
and the matched 4k runs are in the [4k report](early-rmse-4k-reproduction-2026-10-11.md#auxiliary-loss-reproductions).

The anomaly target modestly improves the selected checkpoint: validation is
**0.6% lower** and held-out blend **0.9% lower** than seed-1729 U-global. The two
U-global seeds differ by about 1% on test, so this is promising rather than a
robustly established gain. The aligned target improves validation by 0.4% but
**worsens test by 1.4%**. Its SST error is 15.2% higher despite 8.3% lower upper-OHC
error. A lower blended validation minimum need not improve each physical field.

| Model | Best validation | Endpoint validation | Final-five mean | Held-out blend |
|---|---:|---:|---:|---:|
| U-global | 0.8241 | 0.8688 | 0.8830 | 0.8423 |
| U-global-seed1730 | 0.8419 | 0.8633 | 0.8808 | 0.8503 |
| U-aux01-6k | 0.8205 | 0.8596 | 0.8736 | 0.8538 |
| U-aux01-anomaly-6k | 0.8190 | 0.8285 | 0.8520 | 0.8349 |

The anomaly recipe's final-five mean improves 3.5% over U-global, similar to
6k early-fine and fine+latent (0.8524/0.8502). These are correlated diagnostic
points, not independent trials or a replacement checkpoint selector.

| Model | SST (°C) | Geostrophic velocity (m/s) | EKE (m²/s²) | OHC 0–700 m (10⁸ J/m²) | OHC 700–2000 m (10⁸ J/m²) |
|---|---:|---:|---:|---:|---:|
| U-global | 0.6099 | 0.11260 | 0.024978 | 6.879 | 3.928 |
| U-aux01-6k | 0.7026 | 0.11273 | 0.024953 | 6.308 | 4.017 |
| U-aux01-anomaly-6k | 0.6028 | 0.11136 | 0.024709 | 6.530 | 4.048 |

Anomaly supervision improves SST, velocity and EKE by about 1.1%, and upper OHC
by 5.1%, while worsening deep OHC by 3.0%. EKE remains worse than own initialized
persistence (0.022554 m²/s²). This does not demonstrate recovered fine-scale eddy
variability. Original auxiliary runs were on Beta; these same-seed reproductions
share the current controls' Torch runtime, not an independent-seed uncertainty
estimate.

![Auxiliary and U-global validation curves](artifacts/auxiliary-rmse-2026-10-11/validation-late.png)

[Full trajectories](artifacts/auxiliary-rmse-2026-10-11/validation-full.png) ·
[Observation-update axis](artifacts/auxiliary-rmse-2026-10-11/validation-observation-exposure.png) ·
[Numeric minima/endpoints](artifacts/auxiliary-rmse-2026-10-11/minima.csv).
The main validation, component, training and depth-profile figures above now
include the two 6k auxiliary runs. The dedicated reproduction comparison contains
510 verified validation events across ten series, including the original runs.

The accessory loss is learned, not numerically negligible: aligned-target MSE
falls from **0.4505 to 0.2611**, and anomaly-target MSE from **0.2872 to 0.2669**,
comparing averages over the first and final quarters of training. At the end,
the weighted accessory term is about **29% of the OM4 physical objective** for
both recipes. These scalar ratios do not measure gradient contribution, and the
accessory loss is absent on observation updates.

![Accessory versus physical training objectives](artifacts/auxiliary-rmse-2026-10-11/accessory-training.png)

[Training-loss values](artifacts/auxiliary-rmse-2026-10-11/accessory-training.csv) ·
[July day-30 SST anomalies and own persistence](artifacts/auxiliary-rmse-2026-10-11/maps-6k/day30-2022-07-anomalies-sst.png) ·
[ADT anomalies](artifacts/auxiliary-rmse-2026-10-11/maps-6k/day30-2022-07-anomalies-adt.png) ·
[January/July absolute and anomaly map set](artifacts/auxiliary-rmse-2026-10-11/maps-6k/).
Maps retain the established five-day-mean convention and two pixels per grid cell.

The four auxiliary reproductions used **14.2461 allocated GPU-hours** including
qualification and 96-origin evaluation, with no GPU or training retries. Training
producer `56aac9375a59a7204cd54d85a0c5a3b8e9383d59` is unchanged. One CPU-only map
export failed on a filename and was recovered without changing weights; its
record is retained in the [completion/accounting audit](artifacts/auxiliary-rmse-2026-10-11/verification.json).
The matched eight-model annual evaluation is now queued/running under audit
**19606941** → evaluation **19606942** → collection **19606943**. It will report
separate four-component day-30/day-365 scores for choosing the quarter-degree
variants, using fixed validation-selected checkpoints.
