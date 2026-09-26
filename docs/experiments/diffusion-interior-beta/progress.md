<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Execution status

25 September 2026: preparation in progress; no diffusion training jobs submitted.

- Created `codex/diffusion-interior-beta` from the upstream observation branch.
- Confirmed beta access, B200 partition and a four-GPU minimum for standard/test QoS.
- Located existing half-degree OM4 and observation/annual bundles on beta.
- Started exact-release one-degree data staging and existing-data audits.
- Upstream fresh InstanceNorm OM4 pretraining completed at 14,261 updates, but
  scratch observation production and transfer calibration are still running.
  The final chosen baseline model/checkpoint is not available.
- H0/H1 has been inserted before E; half-degree targets cannot alter forcings,
  surface inputs, latent grid, state scales or observation targets.

## Baseline handoff requirements

Record the upstream chosen model's branch and immutable producer, model class and
normalization, complete run manifest, selected checkpoint and SHA256, final
training completion/selection evidence, observation statistics/grid/score
reference hashes, and corresponding OM4 pretraining checkpoint where applicable.
Preserve architecture/adapter contracts and distinguish the final observation
baseline from its OM4 pretraining source. Use completed validation selection;
do not choose a live `best.pt` or select with held-out scores.

Qualification on beta must strictly reload those weights and reproduce the
upstream validation metrics within a documented precision tolerance. Record any
source-branch changes needed before production on this new branch. Do not mutate
or push the upstream branch.


## Prepared launch gate

`scripts/watch_diffusion_beta.py` checks hourly for completion of both final
InstanceNorm runs (`scratch-main`, `transfer-main`) and their matching monthly
and annual held-out reports. It then takes the lower **final upstream validation
score**, never the held-out score, with identical data/statistics/selection
references required. Live best checkpoints cannot qualify. This implements the
upstream declared selection criterion; a changed model/selection contract must
be reconciled before submission.

The selected full observation model and its manifest, completion, score reference
and checksum are transferred through Torch's DTN to beta and verified again.
The first submitted job is bounded at **4 GPUs x 2 hours = 8 GPU-hours**:

- reproduce the selected baseline's nine-origin validation score on beta;
- ten-update real OM4 full-state deterministic initializer fitting;
- matched ten-update real OM4 full-state diffusion fitting, requiring gradients
  in the surface initializer, improving fixed-noise training loss and strict reload;
- fixed-latent-grid / larger-target-grid synthetic gradient and memory smoke.

The last probe is not an H scientific result; registered target-coordinate
geometry remains required for H production. The new module is qualification
scaffolding, not a completed C/D recurrent processor or observational diffusion
training implementation. A/B production caps and sampling/observation-operator
qualification follow the fitting report. No production runs auto-submit from
this watcher. A durable submission intent prevents duplicate allocation on an
ambiguous SSH response; failures require inspection before resubmission.

Existing-data audits passed: all 350 observation checksums, all 416 annual
checksums, 379,625 half-degree chunks and representative field decoding. Six CPU
regressions cover end-to-end gradient reach at both target sizes, channel/area
weighting and readiness rejection of unfinished reports or mismatched weights,
normalization and selection evidence. One-degree staging remains in progress.


## Deployment update — 25 September, 22:42 UTC

Draft [PR #895](https://github.com/m2lines/Samudra/pull/895) is open and assigned
to jder, with a results table of contents. Qualification producer `9ee94d411`
is pushed and its source archive passed read-back verification on beta.
The hourly dependency watcher is active. Its first live check is waiting for
unfinished upstream training and the exact-release one-degree data verification.
No GPU job has been submitted by this campaign.

Existing-data verification covers 393,574,212,039 bytes of half-degree chunks,
21,888,642,966 bytes of monthly observation payloads and 1,025,294,370 bytes of
annual payloads. The new one-degree copy is progressing. Once its full source
read-back succeeds, a final audit checks cross-resolution timestamps and data
readiness automatically.

The watcher will submit one bounded qualification and monitor it hourly,
collecting all four result records and checking matched A/B input hashes. It
stops for a report/production-sizing decision after qualification; it does not
implement unattended execution of the entire 576-GPU-hour campaign.


## Upstream complete; beta qualification queued — 26 September

Both final InstanceNorm paths and their monthly/annual reports completed.
The transfer checkpoint won the frozen validation criterion: **0.5341906**, versus
**0.5439096** for scratch. Its selected observation joint update is 6,400 (scratch:
4,900). Selection did not use held-out scores. The selected full model, matching
normalization/grid contracts and the original pre-observation OM4 checkpoint are
staged and checksum-verified on beta. New-arm OM4 pretraining can start from that
pre-observation source; the final observation-selected model remains the reference.

Exact-release one-degree staging passed full remote-source read-back, all declared
chunks are present, and all 4,745 one-/half-degree CF timestamps, units and calendars
match. Observation and half-degree audits previously passed.

Qualification **206768** is queued for priority; no GPU time has yet been consumed.
The live walltime request was shortened from two hours to **30 minutes** to improve
backfill eligibility, reducing its allocation ceiling to **2 GPU-hours**. The
original submission record is retained. The broader qualification reserve remains
24 GPU-hours within the 576-GPU-hour campaign ceiling. The initial scheduler start
estimate is the following morning; this is not a guaranteed reservation.

An active monitoring goal now covers the campaign through the final report or a
confirmed blocker, with hourly or shorter checks and Slack notification in either
case. No completion/blocker notification has been sent while ordinary queueing
and useful implementation/preparation work remain.


CPU preparation now includes a multivariate Heun sampler with optional activation
checkpointing for gradients through partial observation-operator losses. Tests
verify finite samples, reproducible draws, exact supplied-surface/land constraints
and gradients reaching the initializer. This provides plumbing for observation
fine-tuning without fabricating missing interior labels; it does not yet qualify
the complete monthly observational objective or establish GPU throughput.
Seven targeted CPU tests pass. Queued qualification remains on its original
immutable producer; these additions do not alter that allocation's code.


## Observation-objective preparation — 26 September

Added masked, area- and channel-balanced fair CRPS for independent trajectory
ensembles using the existing monthly interior and five-day surface operators.
CPU tests verify that monthly averaging precedes scoring, missing labels have
zero gradient, unsupervised velocity/deep channels do not acquire invented labels,
and temporal gradients follow the duration weights. Nine targeted CPU tests
pass. This is observation-loss plumbing; real-grid sampler/rollout qualification
and the production training harness remain outstanding. Qualification 206768
remains queued on its original immutable code; no new allocation was submitted.


The matched A/B forecast wrapper now connects surface-history conditioning,
joint initial-pair decoding/sampling, and the frozen upstream physical stepper.
CPU integration tests verify anchored historical surfaces, independently drawn
members, gradients through frozen dynamics into the initializer/adapter/decoder,
and invariance to changes in future observed surface values. Eleven targeted tests
pass. Real-data GPU qualification, replay/pretraining orchestration, checkpointed
production runs and scientific evaluation are still outstanding.


The A/B pretraining objective now accepts the existing Rust-backed
`InitializerWave.model_sample` tensors: observation-scaled states with unchanged
native OM4 flux normalization. It bypasses the ERA5 adapter, supervises both
historical full states, and uses the same known-surface noise convention as
sampling. Tests verify native forcing sensitivity, initializer/decoder gradients,
and absence of adapter/dynamics updates in this objective. Thirteen targeted CPU
tests pass. The production training loop remains to be connected and qualified.


A second real-data qualification entry point is prepared, but **not submitted**.
It uses four independent GPUs for A and B with 8/16/32 sampling steps, after
ten anchored OM4 fitting updates from the verified pre-observation source. It
checks a training month's observation-gradient path, memory/time, frozen dynamics
and strict whole-model/optimizer reload. This measures implementation behavior
and sampling cost; ten fitting updates cannot establish sampler convergence or
scientific skill. Its optional Slurm stage uses unique job-specific output paths.
The original queued qualification and its immutable producer remain unchanged.


## Resumable OM4 runner prepared

The A/B pretraining entry point now uses the existing Rust-backed frame cache and
shared observation state scales. Production is gated on baseline reproduction and
a matching arm/producer/data/source qualification, which is not yet available.
It selects A by deterministic full-interior validation MSE and B by fixed-noise
validation denoising loss; these are within-arm pretraining selection objectives,
not comparable skill results. Observation-based final selection remains required.

Optimizer-boundary checkpoints preserve full weights, optimizer and random-number
state; data order is reconstructed from the completed update count. A CPU test
with stochastic dropout confirms bitwise identical weights and optimizer state
for interrupted/resumed versus uninterrupted training, and rejects changed
protocols. Fifteen targeted tests pass. Cache preparation, GPU throughput and
observation fine-tuning integration still require real allocation evidence; no
production job has been submitted.


A point-evaluation adapter now applies the unchanged upstream monthly/surface/OHC
metrics and frozen selection criterion to the mean of separately evolved ensemble
members. Fixed evaluation draws are independent of training RNG. A nonlinear
regression test verifies averaging after evolution rather than evolving the mean
initial state. This is point scoring only; calibration and individual-member
structure remain required in scientific reports. Sixteen targeted CPU tests pass.


The runner is now `samudra.experiments.diffusion_train` with explicit `om4` and
`observation` phases. Observation fine-tuning strictly loads the completed arm's
selected pretraining checkpoint, trains the initializer/decoder/ERA5 adapter with
frozen physical dynamics, and retains matched native-OM4 replay supervision.
Validation uses only the fixed nine-origin cohort and frozen upstream score on
ensemble means. Replay defaults are one additional example every four updates
at weight 0.1, recorded in the resume protocol along with reference/checkpoint
hashes. Real-grid fine-tuning, replay memory cost and throughput remain unverified
until GPU qualification; no production run has been submitted.


Ensemble-report diagnostics now include area-weighted additive sums for mean
squared error, unbiased member variance, empirical and fair CRPS, central 80/90%
interval coverage and widths, and rank histograms with fractional tie handling.
They apply the same monthly interior and five-day surface observation operators;
missing observations do not contribute. Closed-form tests check score values,
rank mass conservation and collapsed ensembles. Pooling uses sums and support
weights rather than averaging unequal cohorts. Raw finite-ensemble coverage is
reported with the member count; it is not a claim of resolved tail calibration.
Eighteen targeted CPU tests pass; real forecast diagnostics are still pending.


Coordinate-registration preparation now includes a parameter-free differentiable
feature mapper that uses actual source/target latitude centers and periodic
longitude. Tests cover nonuniform latitudes, longitude wrapping, polar extension,
gradients and unchanged source tensors. This is interpolation of conditioning
features, not conservative regridding of targets. Integration into the H decoder
and verification against the staged native grids are still outstanding. Twenty
targeted CPU tests pass; no H experiment or geometry qualification is claimed.


C/D preparation now includes a surface-feature projection into two persistent
latent memory slots and a shared residual processor driven only by coarse forcing
and context. Decoded interiors and diffusion noise never enter the recurrence.
The processor rejects changed conditioning-grid shapes. CPU tests verify that
future deterministic and denoising losses reach the initializer and processor
through three latent steps, with sensitivity to forcing. Configurable default
width/depth/grid are preparation values, not a frozen production choice; A/B
results and the wave gate still determine C/D sizing. Twenty-two targeted tests
pass. C/D training/evaluation integration and all GPU evidence remain pending.

## Engaging migration and bounded-memory loading

The campaign now targets Engaging with user-approved existing OM4 releases.
The pending beta qualification was canceled before allocation (zero elapsed
allocation time); the same total 576 GPU-hour cap applies across clusters.
Observation/checkpoint transfer and an x86 container build are in progress.
No Engaging GPU result is available yet.

Production A/B training now defaults to native Rust batch loading rather than
requiring a resident whole-dataset GPU cache. `--device-cache` remains an explicit
option for sufficiently large devices. This changes storage behavior, not
training examples, 19-frame histories, forcing alignment, normalization, or
losses. A temporal-alignment test compares both sampling paths exactly;
13 initializer, physical-diffusion and resumable-fit tests pass. Real L40S
memory use and throughput still require qualification.
