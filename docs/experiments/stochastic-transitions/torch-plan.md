<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# One-pass stochastic transitions on Torch

Three fresh models compare deterministic evolution, initialization uncertainty,
and stochastic transitions. Both initializer and processor are real repository
Samudra multiscale ConvNeXt U-Nets, widths 64/128/256, expansion 2, InstanceNorm.
The processor reads two physical states and predicts a residual increment.
The initializer reads 19 five-day surface/forcing frames and predicts two 77-channel
physical states. Inputs include validity and geographic/calendar context.
No persistent external noise or learned recurrent channels are introduced.

| Arm | Initializer | Processor | Objective |
|---|---|---|---|
| A | Deterministic | Deterministic | MSE |
| B | FiLM stochastic | Deterministic | Fair CRPS |
| C | FiLM stochastic | FiLM stochastic | Fair CRPS |

FiLM modulates normalized bottleneck and decoder features with global and coarse
spatial Gaussian noise. Heads start at zero; all arms share initial backbone
weights for seed 1729. Each member feeds its own state history forward. There
are no hard surface anchors or additional artificial observation gaps.

## Matched exposure

4,000 alternating mixed optimizer updates (2,000 OBS / 2,000 OM4), followed by
1,000 OBS-only cooldown updates. Effective batch 8 examples; four training members
for B/C, one for A. All stochastic evaluations use eight members; deterministic
members are identical. Main LR 1e-4, linear cooldown to 1e-6, retain Adam moments,
reset EMA at cooldown, EMA half-life 100 updates, gradient clip 1. No replay, latent
bound, separate OM4 pretraining, or spatial-gradient loss. Checkpoints every 500.

Shared eight-channel ERA5 passes through the same learned 8→32→3 adapter for both
sources. Fixed OM4 normalization and frozen OBS atmosphere statistics; full-lat
wet/finite target support. OBS losses: 0.5 monthly interior T/S + 0.5 surface forecast
+ 0.1 latest-surface initialization. OM4: 0.5 monthly + 0.5 native forecast + 0.1 latest
full-state initialization. Normalized point losses are MSE (A) or fair CRPS (B/C).

This compares forecasting systems, not an isolated stochasticity parameter:
MSE versus CRPS differs; monthly fields here are calendar-weighted physical
trajectories rather than direct-window diffusion readouts; all native forecast
leads are scored instead of random single-lead denoising subsampling. The latest
OE latent-diffusion reference carries two latent slots and does not feed sampled
decodes back. Keep these distinctions in results and cost comparisons.

## Data and qualification

Observation bundles on Torch match Beta grid/statistics/manifest hashes.
Existing Torch OM4 source metadata differs, so use the exact packed Beta cache
with its exact metadata/static source companion. Preserve original proof files
and explicitly audit relocation. Required transfers are about 59.5 GB plus small
source companions; no replay data is required.

Qualification runs 24 mixed updates then resumes for 4 OBS cooldown updates per arm,
checks source counts and finite training, and evaluates one observation origin.
Record per-second nvidia-smi GPU utilization and memory, update time and exposed
data wait. Require mean GPU utilization strictly above 70% for each arm over warm
updates 5–24 before any production submission. Save failed qualification evidence;
address loader/batching/kernel issues before repeating. Qualification weights
never initialize production. Pin source/config/data hashes at release.

## Evaluation

Nine frozen origins Nov 2013–Jul 2014, eight members, raw and EMA every 500updates.
Use the exact OE five-component climatology-normalized RMSE metric: SST,
SSH-derived velocity, EKE, OHC 0–700 m and OHC 700–2000 m. Surface leads 5/15/30 days;
monthly OHC from calendar-weighted trajectories. No spectral term in selection.
Report individual components and selected/final checkpoints. Keep ensemble mean,
member maps and variability diagnostics separate. Annual evaluation can follow
on fixed selected checkpoints; annual test data must not select the checkpoint.

## Bring-up record, October 28

Data staging passed full read-back verification: 5.60 GB shared ERA5,
53.93 GB packed OM4 (all 2,702 frame hashes), and 672 exact source companion
files. Torch paths and explicit manifest relocation are recorded under
`/scratch/jr7309/data/stochastic-onepass-v1/STAGING_VERIFIED.json`.

Qualification 21172669 was canceled without running after Slurm refused release
from its user hold. Replacement 21181552 failed after six seconds, before Python,
because the launcher requested a nonexistent Apptainer module. The corrected
launcher uses the verified `/share/apps/apptainer/bin/apptainer` executable.
Qualification 21181743 uses frozen source SHA256
`afa4fc4e2956f7740110bdaaec1a49c63658619b6b8cdacd67d0d157272558c4`.
No production arm has been submitted. GPU utilization is not yet measured.

Qualification 21181743 completed A's 24 mixed and four cooldown updates with
finite losses and retained Adam history, but failed the telemetry gate: the
container's GPU ordinal did not match the Slurm global index, so the logger
recorded no rows. No utilization claim follows from that attempt. Its files are
preserved as `qualification-failed-21181743`. The corrected logger resolves CUDA
ordinal zero to a UUID, matches that UUID in NVIDIA telemetry, and fails before
training if no row is available. Empty-output and device-selection guards were
checked locally without GPU execution. B/C and validation remain unqualified.

Qualification 21182534 measured 35.65% warm GPU utilization for A (113 samples),
with OM4 updates averaging 8.49 s and OBS updates 3.25 s. The >70% gate failed;
B/C and production were not run. The next qualification uses microbatch four,
no activation checkpointing, four-update prefetch, a smaller 64-example OBS
cache, and verified node-local packed OM4/ERA5 copies. Unused anchor allocations
were removed without changing outputs, gradients or RNG state in CPU tests.
The combined model, training and evaluation suite passes 16 tests.

Qualification 21183569 verified 59.54 GB of node-local caches in 91.43 s.
A completed all training/resume checks, but its 45 warm GPU samples averaged
exactly 70.0%, below the strict release requirement. OM4 updates averaged
2.73 s, OBS 1.94 s, and exposed data wait 0.16 s. B/C remain unqualified.
The next retry removes repeated loss-side GPU synchronizations. It uses
microbatch four for A and one for B/C, whose four ensemble members multiply
the activation footprint; effective batch remains eight examples in every arm.

The optimized objective passes 23 combined CPU tests, including numerical and
gradient equivalence and absence of scalar-extraction profiler events.
Qualification 21184506 uses source archive SHA256
`f47d5a9bc6812e8ac7e68440853d46ea5419d35c68c5ea6998de2955e5f44133`.

Qualification 21184506 on gr102 failed at 15.20% utilization: OM4 17.82 s,
OBS 1.97 s, exposed data wait 7.57 s. All 28 A training/resume updates remained
finite. The earlier 70% attempt ran on gr105, so these measurements do not
isolate the loss-code change. The next fix removes unused full-interior history
reads, normalization and transfers from this campaign loader, preserving future
and initialization supervision. B/C and production remain unqualified.

The compact native loader passes 28 combined CPU tests. For a six-bin future
window it transfers 589 channel-frames instead of 1925 (69.4% less), preserving
exact model inputs, targets, losses and gradients in the comparison tests.
Qualification 21185817 uses source archive SHA256
`1dd15995e5ed09322eb11741e9dc7c4c17ee887a32aa7696447466576901c818`.

Qualification 21185817 on gr101 remained below the utilization gate (33.02%).
A averaged OM4 7.38 s / OBS 2.01 s with 2.70 s exposed data wait. Compact
loading reduced CPU RSS to about 30.5 GB and GPU memory to 30,493 MiB.
The next qualification uses an explicit resident normalized native cache to
remove recurring native disk reads and normalization. One-time preparation
time and RAM use will be reported separately from steady training throughput.

Resident-cache implementation passes 33 combined CPU tests, including exact
inputs/loss/gradient equivalence and no recurring raw reads or normalization.
Qualification 21186841 uses source archive SHA256
`8ade59f9f0750c869a14b9cb197344522226bbfac24136f3049b2da48d0431ca`.
It collects all arms even if an earlier utilization check fails; production still
requires all arms above 70% plus finite training/resume/evaluation checks.

Qualification 21186841 passed all three arms and evaluations. Mean GPU
utilization: A 77.89%, B 83.41%, C 82.43%; exposed data wait approximately
0.00011 s/update. OM4/OBS update seconds: A 2.04/1.83, B 7.77/6.98, C 8.09/7.30.
Peak process RSS approximately 109 GiB; cache preparation took 163–169 s per
process. These are engineering checks, not scientific comparison results.
Production A/B/C submitted as Torch jobs 21190337/21190338/21190339, respectively,
with unchanged qualified code/data, 4,000 mixed + 1,000 OBS-only cooldown updates,
effective batch eight, checkpoints every 500, and online W&B logging.

Production A completed successfully in 3h01m14s (job 21190337), including
5,000 updates and all 20 raw/EMA evaluations on the nine validation origins.
EMA blend improved from 0.888229 at 4k to 0.835800 after the OBS cooldown;
final raw blend is 0.834774. Lower is better. B/C are still running; no
stochastic-versus-deterministic conclusion is available yet. Exact receipts:
[JSON](results/A-evaluations.json), [validation curve CSV](results/A-validation.csv).

Production B completed in 10h34m27s (job 21190338), with all 20 evaluations
verified. Final/best EMA blend is 0.816268, versus A 0.835800 (2.34% lower).
B improves from 0.856716 before cooldown to 0.816268 afterward; final raw
blend is 0.818677. This is one seed on nine monthly validation origins. B also
changes the objective from MSE to fair CRPS, so this comparison does not isolate
stochastic initialization alone. C remains in its OBS cooldown. Receipts:
[B JSON](results/B-evaluations.json), [B curve CSV](results/B-validation.csv).
