<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Learned initialization gaps and mixed pretraining with an observation-only finish

**Work in progress:** the six-model comparison is not complete. At the latest
verified check, five models had finished training and the legacy scratch control
was still running; several evaluations were waiting for GPU capacity. This file
records the methods for the forthcoming results. Current execution status is in
[progress](progress.md); the [approved plan](missingness-wave-2026-09-28.md) defines
the experiment scope.

## Questions and controlled comparisons

The wave tests whether explicit missingness-aware initialization can remove
unrealistic filled surface values, whether OM4 exposure improves observation
learning relative to scratch, and whether task-specific inputs plus an
observation-only finish retain useful source-task behavior. It uses smaller
models and the existing one-degree data to make these comparisons economical.
It does not introduce new annual-rollout training, dense-versus-observable-only
OM4 supervision ablations, or a probabilistic output model.

There are three different standards of evidence. Artificially hiding available
observations gives a scored test of completion. Naturally missing regions give
maps and numerical range diagnostics, but no invented accuracy score. Observation
forecast metrics determine model selection independently of those visual audits.

## Models and training

Every result label below denotes the complete initializer, processor and forcing
adapter system, not just its processor. All models are trained anew with
InstanceNorm. The small history initializer has 31,243,306 parameters, the D
processor 31,631,677, and the forcing adapter 387: **62,875,370 total**. Conditioning
adds 91,176 parameters, for **62,966,546 total**. These are not conversions of the
historical BatchNorm D checkpoints and not the 205.6M-parameter models from the
preceding wave.

| Literal model name | Initialization and task ordering | Final OM4 / observation updates |
| --- | --- | --- |
| Small legacy scratch | Random weights; old climatology fill and unconditional surface overwrite; observations only. | 0 / 16,000 |
| Small masked scratch | Random weights; missingness-aware initialization and learned completion; observations only. | 0 / 16,000 |
| Small masked sequential | Missingness-aware model; 8,000 OM4 updates followed by 8,000 observation updates. | 8,000 / 8,000 |
| Small masked mixed-finish | Missingness-aware model with shared task inputs; interleaved prefix, then 2,000 observation-only updates. | 8,000 / 8,000 |
| Small conditioned mixed-finish | Same mixed-finish schedule, with task-specific initializer and processor input adapters. | 8,000 / 8,000 |
| Small conditioned mixed | Same task-specific adapters, with interleaving continuing through the final update. | 8,000 / 8,000 |

All runs use seed 1729, batch eight, AdamW at learning rate 1e-4, weight decay
0.01 and gradient norm cap one. Each arm has 16,000 optimizer updates, with
separate deterministic sample streams per task. This matches update count,
not exact FLOPs or observation exposure. Scratch's raw 8,000-observation checkpoint
also permits equal-observation-exposure comparisons against the pretrained arms.
There is no separate reconstruction-only stage in this wave: reconstruction is
an auxiliary loss within the joint updates.

Mixed-finish allocates 8,000 OM4 and 6,000 observation updates to its first 14,000
updates, followed by 2,000 observations. The continuously mixed arm has a different
prefix allocation so that its final counts still match. Consequently the
mixed-versus-mixed-finish comparison is a schedule comparison, not a fork from an
identical checkpoint that isolates only the tail. The conditioned versus
unconditioned mixed-finish comparison does use the same schedule and task streams.
Identity initialization of the adapters preserves the common backbone's initial
weights and does not consume its random initialization stream.

The initializer consumes 19 five-day surface-history frames plus forcing and
validity information, and produces two full-state frames. Five fixed context
channels encode location as spherical x/y/z and annual phase as sine/cosine.
These give explicit geography and season; they are not learned coordinate tables.
The processor then takes six autoregressive five-day steps during training.
OM4 uses its original forcings; observation forecasts use the existing ERA5
forcing adapter. Available future ERA5 forcing is prescribed, so these are
forced hindcasts rather than operational forecasts with uncertain atmospheric
forcing.

The observation task uses the existing products and chronological split:
May 1993–July 2013 for training, nine fixed November 2013–July 2014 validation
origins, and 96 monthly test origins spanning 2015–2022. Observation-derived scales
are shared by all arms. Velocity channels use mean zero and unit scale; unobserved
deep T/S channels use the deepest observed T/S scale rather than simulation-derived
normalization. These slots remain unconstrained by direct observation labels.

## What changed about missingness

The legacy model already had validity channels. The correction is a bundle:
zero placeholders for unavailable normalized inputs, copying surface observations
only where valid, retaining predicted surface values elsewhere, structured
artificial gaps across the input history, and a supervised completion loss on
known values that were hidden. This is not a mask-channel-only ablation.
Land masks and missing observation masks remain distinct. Natural missing labels
are never replaced by artificial truth.

During OM4 updates, observation-training coverage masks and artificial gaps hide
inputs while real OM4 targets remain available. During observation updates,
completion targets are only genuinely available observations that were hidden.
The additional surface-completion term has weight 0.1 and covers full-grid wet
observed support, including available polar observations. The original forecast
scoring domain remains 60S–60N. The completion branch can therefore learn polar
structure even though those regions do not contribute to forecast selection.

Conditioning uses separate identity-initialized 1×1 convolutional adapters for
OM4 and observations at the initializer (138 input channels) and processor
(162 input channels) entrances. An explicit source label chooses the adapter;
the expensive backbones remain shared. There are no separate output heads and
no requirement that unconstrained internal velocities become physically
interpretable during observation training.

## Losses, checkpoint selection and controls

For an observation update, the forecast loss is
`0.8 × monthly T/S + 0.1 × SST + 0.1 × SSH`, with normalized squared errors on
available labels. Monthly T/S compares a calendar-weighted aggregate of predicted
five-day states with the monthly interior observation. SST/SSH compare each
predicted five-day state with its matched five-day observation over the rollout;
they are not evaluated only at the end of the month.

An additional `0.1 × monthly T/S reconstruction` applies to the initializer's
monthly reconstruction. This auxiliary path slides the initializer through the
month using contemporaneous surface histories, then averages its reconstructed
states; those later surface observations do not enter the autoregressive forecast
path or forecast evaluation. The corrected models add `0.1 × hidden-surface
completion`. These coefficients neither sum to one across the full objective
nor specify fractions of the actual gradients. OM4 updates use balanced
full-state forecast errors plus 0.1 times initializer reconstruction and the
same completion term where enabled.

Checkpoint selection uses the frozen integrated-plus-spectral observation
validation score. Half is the mean of five climatology-normalized errors:
SST, geostrophic velocity, EKE, OHC 0–700 m, and OHC 700–2000 m. Half is the mean
spectral error over the prequalified regional/lead keys. Spectral error is in
**dex**, meaning base-10 logarithmic power-ratio error. Test tables use test
seasonal climatology denominators with the same frozen spectral keys; validation
selection continues to use the fixed validation denominators.

Finish models select their best checkpoint only within the observation-only
finish. Best-anywhere remains separately recorded. Results must include both
these validation selections and the raw matched-budget endpoints, rather than
choosing whichever test score happens to look better. Persistence holds each
selected model's own initialized state fixed; anomaly persistence also carries
its interior anomaly relative to monthly climatology. These are initializer-aware
controls, not a climatological full ocean state shared by all models.

## Diagnostics and limits of interpretation

Completion audits use the same nine validation origins and fixed artificial
blocks or hidden polar caps. RMSE and bias are pooled with wet-area weights over
known hidden labels, with separate geographic support counts. Empty support stays
null. The observation-training monthly climatology and normalized-zero fill are
explicit controls. Full-grid maps show initialized SST, SSH, subsurface temperature,
salinity and velocities as learning progresses; any values beyond plotting limits
are counted separately rather than silently treated as plausible.

Initialized profiles show wet-area means and RMS over the same nine origins,
through the 14 observed depth levels. They describe what the initializer produces;
velocity profiles are not observation-validated full-state reconstructions.
Frozen-state interventions replace initialized velocities by zero or the five
unobserved deepest T/S levels by their normalization means. Changes in subsequent
observation errors test sensitivity to these slots, not whether they represent
correct physical velocities or uniquely explain a pretraining benefit.

Annual evaluation uses the three existing January 2015, 2018 and 2021 origins.
Day-365 numbers refer to the last five-day output at that lead, not the mean error
over a year. The cases have been inspected in earlier work and are exploratory;
they do not influence selection. Full-grid maps can reveal polar artifacts beyond
the observation scoring domain, but cannot establish polar accuracy without labels.

One seed and a fixed small-scale budget do not establish statistical significance,
the best possible scratch baseline, or a resolution-independent result. Cross-arm
rankings can differ between short-range metrics, completion and annual behavior.
Hardware changed from H200 to regular RTX during recovery; exact checkpoint and
optimizer/RNG restoration was qualified, but arithmetic across GPU families is
not claimed bitwise identical.

## Frozen-checkpoint diagnosis already completed

The [frozen-model gap report](initialization-gap-diagnostics-2026-09-28.md) uses the
three prior large InstanceNorm models. Replacing only missing initial surface
values with training-only OM4 monthly climatology reduced sequential/mixed
three-case day-365 SST error by about 5%, but did not remove annual artifacts.
Changing missing values across the entire history worsened short-range errors.
Those interventions preserve available observations and the original weights;
they establish sensitivity, not the result of retraining a missingness-aware model.

## Provenance and qualification

Training producer: `053e34947932fdb56c6baf2c717a23f2df3cc733`.
Evaluation producer: `9999caf5937a8d0983dc0b6b6883011dc6a1a450`.
The [qualification records](artifacts/2026-09-28-missingness/qualification-records.json.gz)
were read back and hash-verified. All production arms passed fitting and resume
qualification before submission;
completion-head gradients, copy invariance, target separation, schedule counts,
and finish-restricted selection have explicit checks. Local qualification had
54 targeted tests, with two additional evaluator scoring tests. The frozen
baseline reproduced all nine model/origin exports exactly before interventions.

Remote experiment root:
`/scratch/jr7309/runs/2026-09-28-observation-missingness`.
PRODUCTION_DAG.json and EVALUATION_DAG.json identify current jobs; original records
and failed/preempted attempts remain preserved. Final collection checks actual
checkpoint hashes, exact task counts, no partial-completion markers, evaluation
signatures and all required origin lists. GPU-hours include retries and allocated
evaluation time, not requested wall limits. The [progress log](progress.md) records
recovery and scheduling history; final accounting and result artifacts will be
added here after the outstanding evaluations finish.
