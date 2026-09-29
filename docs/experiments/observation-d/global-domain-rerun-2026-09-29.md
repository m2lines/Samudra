<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global observation supervision: matched Small conditioned mixed rerun

Authorized September 29: repeat Small conditioned mixed with no polar latitude
cutoff in inputs, observation training losses, checkpoint selection or evaluation.
The completed restricted-loss run is the control; it is not retrained.

The treatment preserves the 62,966,546-parameter InstanceNorm model, task-specific
input adapters, observed-only surface copying, zero missing placeholders, learned
completion, seed 1729, AdamW learning rate 1e-4, accumulation 8, and the original
mixed schedule: 8,000 OM4 plus 8,000 observation updates, without an observation-only
finish. Both runs start with random weights. OM4 forcings, ERA5 adapter,
reconstruction and completion terms, data splits, normalization constants and
regional spectral diagnostics remain fixed.

The old loss zeroed all observation weights beyond 60S–60N. The treatment uses
cosine-latitude area weights over all finite targets on the model wet grid,
including polar SST/SSH and available interior T/S. Inputs already include all
available latitudes and remain unchanged. Missing values do not become targets;
land remains masked. Normalization scales remain identical to the control (their
original estimation cohort/domain is not changed); this is an affine scaling,
not an input or loss-support mask. Surface completion already used global support.

Global integrated-plus-spectral validation (protocol v4) selects checkpoints;
its climatology reference is recomputed globally from the same training-only
climatology and nine validation origins. The existing named spectral regions
remain regional rather than claiming a global spectrum. The geostrophic diagnostic
still excludes the equatorial band where its conversion is ill-conditioned;
there is no polar cutoff, and SST/SSH/T/S retain equatorial support.

The primary comparison is the two immutable endpoints with exactly 8,000 OM4 and
8,000 observation updates, scored with the same global evaluation code. This avoids
confounding a training-domain change with a checkpoint-selection-domain change.
Original selected control and globally selected treatment are secondary results;
their different selection domains must be stated. Monthly evaluation uses all 96
test origins with inferred persistence and climatology controls. Annual evaluation
uses the existing three cases, with global scoring and full-grid map exports.
No new long-rollout training method is introduced.

The new producer must pass a real fitting probe and mixed-task resume-equivalence
probe before production. Slurm after-success dependencies and runtime qualification
contracts enforce this; the domain flag is included in both fitting and resume
contracts. Global evaluations use separate output directories and explicit domain
fingerprints, preserving previous reports/results.

Root: `/scratch/jr7309/runs/2026-09-29-observation-global`.
Control: `/scratch/jr7309/runs/2026-09-28-observation-missingness/conditioned-mixed`.
Submission: `scripts/submit_observation_global.py`.

Status at 18:17 UTC: producer `79025a163817a577ab81af95b401f5cb0563cd12`
is pushed. All 31 targeted tests and repository hooks pass. CPU overlay-build job
18811814 completed successfully. The complete GPU DAG is submitted:

| Stage | Slurm jobs | Status at snapshot |
|---|---|---|
| Fitting qualification | 18811858 | Pending, QOSGrpCpuLimit |
| Mixed-task resume qualification | 18811859 | After successful fitting |
| Global conditioned-mixed training | 18811860 | After successful resume qualification |
| Control endpoint monthly / annual | 18811861 / 18811862 | Pending, QOSGrpCpuLimit |
| Control original-selected monthly / annual | 18811863 / 18811864 | Pending, QOSGrpCpuLimit |
| Global endpoint monthly / annual | 18811865 / 18811866 | After successful training |
| Global selected monthly / annual | 18811867 / 18811868 | After successful training |

Each job requests one RTX6000 and uses the LZanna partition/account. No GPU is
allocated yet (0 consumed GPU-hours). Slurm confirms the after-success dependencies,
requeue behavior, pinned producer and 16-hour production allocation. The shared
QOS CPU limit is currently delaying the fit and control evaluators despite idle
RTX nodes; these jobs are queued, not running. The production runtime will also
verify actual qualification markers and resume contracts before training.
Scratch quota check reports 4.31 TB used of 5 TB, sufficient for the approximately
13 GB training checkpoint set and the additional evaluation exports.

## Additional matched arm: ten autoregressive memory channels

The user subsequently authorized a third run: **Small conditioned mixed global +
10 latent channels**. It matches the global treatment's seed, mixed 8k/8k update
budget, losses, normalization, task conditioning and global input/loss/score support.
The initializer now emits two 87-channel states; the processor consumes the two
87-channel states and predicts the next 87-channel state. The first 77 slots keep
their physical interpretation and losses. The final ten have no direct target,
physical normalization or auxiliary penalty: they learn through their effect on
future physical predictions on both tasks. They are initialized by the network,
updated at every step, and never reset or detached during the training rollout.
They use the union of physical ocean wet masks, without a latitude restriction.

The parameter count changes from **62,966,546 to 63,363,656** (+397,110, about
0.63%). The input/output layers and processor task adapters expand; the U-Net widths,
blocks and normalization stay fixed. This is an update-matched comparison, not an
exact FLOP match. The existing two jobs remain pinned to their original producer.

The third arm must pass its own fitting and mixed-task resume qualifications.
The fitting probe additionally requires physical-loss gradients in both latent
initializer and latent processor output heads. It shares the global arm's frozen
validation reference, with an after-success dependency on that reference's fitting
job. Local tests verify that the latent state is forwarded unchanged into the
next step, physical-only future losses train latent outputs, OM4 losses accept the
unlabelled state, and zero-latent models retain checkpoint-compatible shapes.
Annual exports also retain initial and lead-time latent arrays for diagnostics.

Root: `/scratch/jr7309/runs/2026-09-29-observation-global-latent10`.
Status: code implemented; 37 targeted tests pass; cluster submission pending.
