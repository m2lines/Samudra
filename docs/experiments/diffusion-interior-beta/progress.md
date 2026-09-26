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

Before Engaging qualification, review found that A/B's historical surface
anchoring ignored per-example observation validity. Missing SST/SSH cells would
have been held at their filled value (climatology in the prepared observations). The anchor mask now respects validity
separately for each example and historical time; missing cells remain model
outputs. Tests cover both deterministic and sampled reconstruction, exact
preservation of valid observations, and batch-dependent channel support. This
fix does not change the selected upstream reference model.

The monthly A/B reporting entry point is now prepared as
`python -m samudra.experiments.diffusion_report`. It requires the selected weights
from a completed observation stage, unchanged observation-manifest and score
reference hashes, and all 96 held-out origins. It exports the upstream point
metrics and field arrays plus stochastic per-origin additive CRPS, spread,
coverage and rank statistics. Monthly aggregation is applied separately to each
member before scoring; a test distinguishes this from averaging instantaneous
scores. This command neither selects on held-out scores nor replaces the annual
rollout, structural diagnostics, controls, figures or final scientific report.
No completed A/B weights are available yet to run it.


## Engaging staging complete; qualification submitted

The destination audit completed successfully on September 26: 350 monthly and
416 annual observation payloads match their recorded checksums; both checkpoint
hashes match the selected handoff; one- and half-degree OM4 stores each contain
379,625 declared chunks with matching 4,745-frame CF times and sampled field
decoding. The reused local one-degree fields occupy 98,393,557,082 payload bytes;
half-degree fields occupy 393,574,212,039 bytes. The copied normalization stores
match beta across all 900 files.

The x86 runtime executes on compute nodes and its Rust read pool constructs
successfully. The transfer-node build job created the image but failed its final
execution check because user namespaces are disabled there; compute-node checks
resolved that issue without rebuilding.

The first Engaging qualification requests four L40S GPUs on one host for at most
30 minutes (2 allocated GPU-hours). It is pending priority at this update. The
immutable qualification producer is `ef1d351e8`; later reporting-only changes do
not alter that producer. The earlier beta job was canceled before allocation.
Data readiness, runtime checks and submission are not scientific results.


## Initial L40S qualification passed

[Machine-readable qualification evidence](engaging-qualification.json) records
the bounded implementation checks. The four-task allocation completed in
37 seconds, consuming 0.0411 allocated GPU-hours. This is the only GPU allocation
consumed so far in this campaign; CPU staging jobs do not add GPU-hours.

| Check | Result | Peak allocated GPU memory |
| --- | --- | --- |
| Frozen upstream validation reproduction | 0.53425195 versus 0.53419061 upstream; 0.0115% relative difference | 1.79 GiB |
| Deterministic 154-field fitting probe | Loss 1.03977 → 0.87203; strict reload exact; initializer gradient nonzero | 3.57 GiB |
| Diffusion 154-field fitting probe | Loss 1.01618 → 0.87727; strict reload exact; initializer gradient nonzero | 3.57 GiB |
| Synthetic larger-target shape probe | Passed, with fixed 45×90 conditioning and 360×720 targets | 0.98 GiB |

The fitting objectives differ and these losses are not an A/B skill comparison.
The geometry probe is synthetic and does not qualify registered half-degree
production. Next, a separate four-L40S, 30-minute qualification measures real
observation-gradient flow and sampling cost for A and B with 8/16/32 sampling
steps. Production has not been submitted.

The first observation-gradient qualification stopped before fitting because
bfloat16 autocast produced an initializer feature buffer whose dtype differed
from the supplied float32 historical surfaces. That failed allocation consumed
20 seconds × 4 GPUs (0.0222 GPU-hours). The anchor buffer now preserves the
observation dtype; deterministic and diffusion autocast regression tests check
exact float32 anchoring. The initial qualification plus this failed attempt total
0.0633 allocated GPU-hours. Observation qualification must be rerun before
production.

## Observation-gradient qualification passed

After the autocast fix, all four observation probes passed in one 30-second,
four-L40S allocation. Gradients reached initializer, adapter and decoder;
physical dynamics remained frozen, and strict checkpoint reloads matched.

| Probe | Forward/backward seconds | Peak GPU GiB | Training-month objective |
| --- | --- | --- | --- |
| A, deterministic | 0.973 | 4.05 | MSE objective 0.14847 |
| B, 8 sampling steps | 1.574 | 5.56 | Fair CRPS objective 0.21609 |
| B, 16 sampling steps | 2.232 | 6.75 | Fair CRPS objective 0.21066 |
| B, 32 sampling steps | 3.866 | 9.13 | Fair CRPS objective 0.20964 |

These use one training month after only ten OM4 fitting updates. They establish
execution, gradient flow and approximate cost, not sampling convergence on a
trained model or comparable A/B skill. Start B with 16 sampling steps and revisit
8/16/32 convergence on trained weights. Total allocated GPU time through these
checks is 0.0967 hours, including the failed attempt.

Before the longer first-wave fits, exercise the production runner with native
Rust loading, real validation and checkpoint writes using bounded 100-update
OM4 runs for A/B at seeds 1729/1730. These are execution probes with separate
outputs, not the production comparison. The launchers can version their shell
code separately through `LAUNCHER_COMMIT` while retaining the qualified Python
producer in `CODE_COMMIT`; training still requires its qualification to match
that exact Python producer.

The first native-loader runner probe exposed a substantial read bottleneck:
updates took roughly ten seconds with four concurrent tasks, compared with the
subsecond in-memory OM4 fitting probe. Its generic rollout loader read full
interior histories and future labels that A/B reconstruction did not consume.
The probe was stopped before its 100-update cap to replace that path. It is not a
completed fitting result; all allocated time remains charged to qualification.

A/B now prepares two native Rust views: 19-frame surface/forcing history and the
aligned two-frame full-state reconstruction target. Normalization and masking
remain in the same native preparation path. Preparation asserts bitwise equality
against the full reader at the first and last usable origins before allowing
training. CPU tests cover history/channel/target alignment; real-data parity and
throughput still require a new bounded runner probe.

The first narrowed-reader probe failed before fitting because its two-channel
surface view still carried the full 77-channel mask. The immutable source view
now subsets both its prognostic mask and channel layout. The regression fixture
checks those dimensions as well as the requested history and target indices.
Real-data bitwise parity remains a required startup gate for the replacement run.

## Production runner qualified; first-wave caps

All four narrowed-reader OM4 probes completed 100 updates, final validation, and
checkpoint writes. Both training and validation startup checks passed exact
parity against the full reader. Observed update intervals were 2.61–2.63 seconds,
versus 10.01–11.75 seconds in the stopped full-reader probe. These runs used
different nodes/cache histories, so the timing ratio is operational evidence,
not an isolated benchmark of storage behavior. The completed allocation lasted
346 seconds. All four subsequent two-update observation probes also completed,
including loading selected pretraining weights, fixed validation, masked losses,
OM4 replay, and checkpoints; their allocation lasted 80 seconds.

The first production wave uses A/B with paired seeds 1729 and 1730, batch size
one, AdamW learning rate 1e-4, native Rust loading with four reader threads, and
the qualified Python producer `bdc1e62b2`. B uses 16 sampling steps, two training
members and eight validation members. Observation replay remains every fourth
update at weight 0.1. Checkpoints save every 100 updates; validation runs every
500 updates plus the initial/final checks. Selection remains validation-only.

- OM4 pretraining: at most 12,000 updates or 11 training hours per arm; the
  four-GPU allocation has a 12-hour walltime cap (48 GPU-hours maximum).
- Observation fine-tuning: at most 6,000 updates or 7 training hours per arm,
  with at most 8 walltime hours per GPU (32 GPU-hours maximum for all four).
  Submit after checking pretraining results; these phases belong to the same
  authorized A/B wave.
- These 80 allocated GPU-hours fit inside the 120-hour A/B planning ceiling.
  Qualification, failed attempts, evaluations and any preemption/restarts are
  charged separately against their reserves and the 576-hour total. Revisit
  sampling-step convergence on trained weights before scientific conclusions.

Completed setup/probe allocations through the observation-runner check total
1.2078 GPU-hours. These short fitting runs are execution probes, not the A/B
scientific comparison, and do not replace the planned held-out report.


Production OM4 pretraining has started on one four-L40S host with the caps above.
Experiment logs are stored locally in W&B offline format and JSON records.
Observation fine-tuning will follow review of completed pretraining; C/D, H and E
retain their later-wave report/approval gates.

## C/D model preparation while A/B runs

`LatentOceanForecast` now connects the surface initializer, two-slot latent
processor, and shared-capacity deterministic/diffusion readouts. OM4 supervision
covers initial and future joint state pairs, with gradients through intervening
latent steps. Interior targets never enter recurrence. The observation interface
uses the existing ERA5 adapter and ignores future observed surfaces; only valid
historical surface cells are anchored. Physical dynamics are absent from this
model. D draws conditional readouts independently at each lead, so these outputs
do not establish coherent uncertain trajectories; E remains a separate question.

Six focused CPU tests cover latent/model gradient flow, target-independent
recurrence, fixed latent trajectories across sampling seeds, future-observation
exclusion, and exact known-surface preservation under autocast. C/D training and
evaluation runner integration, real-grid qualification, hyperparameter selection
and later-wave approval remain pending. The running A/B producer is unchanged.

The A/B report command now has an optional `--annual` path for the frozen 2015,
2018 and 2021 continuous-year test origins. It reuses the upstream annual point
metrics and exports rather than defining a new score. B members evolve separately
before averaging; annual point scores and spectra of that mean do not establish
member calibration or member field structure. These remain separate diagnostics.
The annual path verifies payload hashes and uses only historical observed surfaces
with prescribed ERA5 thereafter. Five existing annual/evaluation tests pass; the
new report path has not yet run on completed A/B observation checkpoints.
