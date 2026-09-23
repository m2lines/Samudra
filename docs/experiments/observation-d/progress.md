<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Deterministic D observational pilot: execution record

Report deadline: 23 September 2026, 10:00 EDT / 14:00 UTC. This is an interim
execution record, not the deadline report or a completed scientific result.

## Data preparation

- Exact D grid, 77 channel masks, source normalization and channel order retrieved
  from Torch's `om4_onedeg_v3`; grid shape 180 by 360, 44,892 surface wet cells.
- Grace qualification array 96565 passed SST, DUACS and ERA5; IAP failed because
  GSW was absent. An isolated ARM64 `gsw==3.6.23` installation resolved this.
- Qualification array **96569** completed all four products, including salinity
  conversion and native-depth OHC integration. Two daily/monthly samples per product.
- Full preparation array **96574** is running on Grace with eight CPU workers.
  A prior 123-task submission was rejected by the scheduler submission-count limit;
  the accepted array groups the same 123 product/year tasks into eight jobs.
- Output: `/mnt/home/jrusak/data/obs_full_range/d-observation-pilot/daily`.
  Read-only-source processing covers 1993–2022 and five extra daily fields in 2023
  for the last forecast's complete five-day support. No source store was modified.
- Executed preparation code and target grid are frozen under
  `.../d-observation-pilot/code/frozen-preparation`, with SHA256SUMS checked at each task.
  The repository version changes only the optional GSW import mechanism from that
  qualified executable. Grace uses NumPy 2.4.6, SciPy 1.15.2, Xarray 2025.6.1.

Coverage uses at least 90% of the *full target cell area*, a deliberately strict
coastal interpretation of the proposal. IAP labels never interpolate across a
missing depth bracket. OHC reference integrals use original IAP depth centers,
midpoint layer edges, 1035 kg/m3 and 3850 J/kg/K, and require complete depth support.
This is a stricter common-column scoring domain than legacy partial-column metrics;
report that distinction and accepted coverage explicitly.

## Implementation and selection

The branch adds conservative remapping, explicit monthly/5-day sample construction,
training-only climatology/statistics, per-frame initializer validity, an 8→32→3
forcing adapter, unchanged pretrained tensor shapes, frozen BatchNorm statistics,
and activation checkpointing. Optimization and cluster model qualification are
still pending.

Selection protocol is versioned in `observation_metrics.PROTOCOL`: half the score
is the mean of five integrated observation errors (SST, geostrophic velocity, EKE,
and OHC in two layers), each divided by its fixed validation seasonal-climatology control
error. Half is the mean spatial spectral error in dex for qualified SST/ADT/EKE
region/lead pairs. Smaller is better. Freeze the available spectral keys before
training; missing required spectra or nonfinite components are errors, not permission
to fall back to RMSE. Report every component and curve alongside the scalar.

Surface diagnostics use fixed 5-, 15- and 30-day leads across the nine validation
origins, never a stitched trajectory. OHC uses calendar-month predictions. Spectral
wavelengths must span at least four grid cells. Annual trends, deseasonalized
variance and long-period temporal spectra are unavailable for this short selection
window; longer-held-out diagnostics remain part of the final comparison.

## Verification to date

17 targeted preprocessing, initializer-validity, selection and existing spectral
kernel tests passed. Tests include periodic longitude, spherical integrals, missing
spatial/vertical support, leap-year month weights, exact legacy-mask-path equivalence,
and a case where spectral collapse loses despite improving pointwise errors.
All pre-commit checks passed for the initial implementation. Full-checkpoint strict
loading, full-grid GPU gradients, observational fitting and training are not yet
verified. No observational skill result is available yet.

## Training-path implementation update

The optimizer now supports adapter warming, initializer reconstruction adaptation,
and joint forecast adaptation, with atomic resume checkpoints and validation-only
selection. An adapter-only frozen-core arm uses the same data and selection rule.
The held-out evaluator refuses to run before selection completes and checks the
selected checkpoint checksum. Its controls include source zero-forcing transfer,
inferred persistence, inferred anomaly persistence and seasonal climatology.

The new identity test caught and fixed an axis-order mismatch after the existing
geostrophic kernel returned latitude before time. Fifty targeted tests now pass,
including the existing metric suite. Missing labels do not contribute gradients,
and the copied surface temperature is excluded from interior loss.

GPU execution is still pending. The first code-layer build exhausted login-node
`/tmp`; its replacement uses scratch for the source checkout. The builder now
fetches only source and lockfiles, avoiding approximately 691 MiB of documentation,
while retaining the exact resolved commit, lockfile comparison and read-only layer.
Monthly assembly job **96584** waits on the complete coarsening array **96574**.


## Matched controls and pending jobs

The common selection reference is now seasonal climatology estimated exclusively
from training observations, evaluated on validation. It is shared conceptually
across transferred and fresh-weight arms; no arm-specific initializer determines
its denominators. Require usable SST, ADT and EKE spectral groups before optimization.
This is protocol version 2 and is fixed before any observational training.

A fresh-weight control path uses observation-only normalization, retaining only the
common architecture/grid/masks. Its unobserved velocity scales are zero/one and its
unobserved deeper T/S scales use the deepest observed scale, with no invented labels.
Scratch BatchNorm learns its running statistics; recomputation restores buffers so
activation checkpointing does not count each forward twice. Transferred BatchNorm
keeps its source statistics. A unit test checks buffer and gradient equivalence.
Use learning rate 1e-4 for scratch core parameters, 1e-5 for transferred core parameters.
A capped scratch run is an optimization pilot, not evidence of a converged best
analysis-only baseline.

Torch full-grid contract qualification: **18287283**, initially queued. The
preparation chain is Grace **96574 → 96584 → 96619** (coarsening, monthly samples,
publication plus full read-back verification). Torch DTN access to the OSN source
was checked successfully; the prepared dataset has not yet been transferred.

## Coarsened spectral support probe

The 1–2 January 1993 preparation samples support all three existing boxes (North
Pacific, Gulf Stream, Agulhas) for SST, ADT and DUACS-derived EKE. After requiring
wavelengths of at least four model grid cells, each box has only **three radial
spectral bins**. These are broad-scale diagnostics, not resolved mesoscale spectra.
This probe verifies data support, not predictive skill or full validation coverage.
Within the model's 60°S–60°N wet-cell domain, finite observation fractions by cell
count are 93.95% for SST, 94.19% for ADT and 94.12% for EKE. Remaining cells are not
silently filled as targets. Full validation support will be frozen before training.

Torch DTN staging process **4037693** is alive and waiting for the publication
proof. It will copy the dataset and verify all 350 NPZ source hashes before creating
`DATA_READY.json`. The trainer refuses datasets without that verification marker.

## Scheduler and qualification update

RTX qualification 18287283 remained pending with `QOSGrpGRES`; it never ran.
After a scheduler test accepted the preemption-only H200 route, replacement
qualification **18288326** was submitted with requeue enabled and the original
queued RTX job was cancelled. This changes the execution hardware, not the source
checkpoint or task. The synthetic check qualifies initializer loading/equivalence
and gradients; the full observational forecast path still requires its own fitting
check.

The training implementation is staged in immutable code layer
`samudra-code-652796441b3fd65bdaef858fe574b397b7baeb15.img` on Torch. A subsequent
training-only fitting gate performs ten updates on one training month, requires
finite nonzero gradients in initializer, evolution and adapter, and requires the
fixed-sample loss to decrease. Its selection reference is frozen before those
updates. No test observations enter this gate.

Current targeted verification: **52 tests passed**, including future-ocean-input
independence, forcing visibility by lead, and scratch BatchNorm recomputation.

## Full-grid runtime result

H200 job **18288326 completed successfully** in 30 seconds. Strict loading passed
for the recorded D checkpoint, the zero-adapter initializer output was bitwise
identical to its original path, and gradient norms were 1.2885 for D and 0.005716
for the adapter. Peak GPU allocation was **2.82 GiB**, with output shape
`1 × 2 × 77 × 180 × 360`. This remains a synthetic contract test, not an
observational skill result.

A gated submission driver will wait for verified data, submit the observational
fitting check, then submit primary and adapter-only runs behind its successful
completion. Production validates the fitting proof's code/data hashes and shared
selection-reference hash. Scratch fitting/training follow both transfer arms.
Evaluation and scratch jobs are ordered to keep at most two GPUs active. Evaluation
artifacts are atomic and resumable at predictor boundaries for preemptible jobs.
The nine-hour H200 production request passed the scheduler's `--test-only` check.

## Execution controller staged (22 September, 23:13 UTC)

Production code commit `d8949bb15d9fa92e6c8c94702092360bbe9005ef` is pushed and
published as a verified Torch overlay (SHA-256
`3c68d4475f45a139631c55f0711f252af781a4e578966952f91bb055ca25a952`).
Controller PID **2228710** is running on the Torch login host, waiting for
`DATA_READY.json`; no production GPU job has been submitted yet. It will submit
the fitting gate and dependent comparison/evaluation jobs after staging completes.
At 23:11 UTC, all eight preparation workers were running, with monthly
materialization and publication waiting on their successful completion.

A read-only snapshot collector captures controller process state, Slurm accounting,
run manifests, the integrated selection score and spectral curves, and completed
held-out results. Its first live staging capture succeeded; it does not evaluate or
modify models. Captures are timestamped and refuse to overwrite existing evidence.

## Resource clarification (23 September, 00:36 UTC)

The user's one-host/four-GPU constraint applies to beta training, not Torch.
The existing Torch pilot layout is retained (at most two concurrent GPUs).
No four-GPU production job was submitted. The waiting controller was briefly
stopped during clarification and restarted as PID **2799320**; data preparation
and the DTN receiver continued uninterrupted. EAI preparation is reached through
the `alpha` login alias but runs on the Grace compute host `betagg12`.
The compact prepared dataset is published to OSN with full read-back verification,
then received through Torch's `dtn011` and checked against all source file hashes.

## Scratch capacity audit (23 September)

The earlier wave-2 report records a quota failure resolved after user cache cleanup.
A fresh `myquota` read now reports **3.66 TB / 5 TB** scratch usage (1,646,160 /
5,000,000 files), approximately **1.34 TB free**. The aggregate filesystem's free
space was not used as the user's quota allowance. No cleanup is currently required.

Conservative storage estimates: monthly sample arrays 30,848,947,200 bytes;
fifteen evaluation exports 29,859,840,000 bytes (allowing float64 throughout);
checkpoint/optimizer/atomic-replacement allowance 26,988,293,472 bytes. Including
one temporary evaluation export gives **89,687,736,672 bytes** (~83.5 GiB).
The model state size was computed from the actual architecture: 613,458,648 bytes.
Reserve **150 GiB** for the pilot, plus **128 GiB** additional free headroom.
Existing container/cache usage is already included in current quota usage.

The waiting receiver was paused during this audit, then replaced with a guarded
version. Before copying it reruns `myquota`, subtracts a rounding margin, requires
the pilot budget plus headroom, and rejects an incoming NPZ payload over 40 GiB.
Malformed quota output also prevents copying. Guard checks cover current capacity,
tight quota, an oversized payload and malformed output. Existing datasets and
checkpoints were not deleted.

## Coarsening complete; first monthly sample checked

All eight tasks of Grace job **96574** completed with exit code 0; the longest
elapsed time was 2:23:36. Materialization **96584** is running. At the 00:51 UTC
check it had written 73 training months totaling 4,590,567,473 bytes; these counts
are an intermediate observation, not the final publication manifest.

The actual May 1993 training sample was independently read locally. Its 26 bins
contain 19 historical and seven future bins on the exact 180 × 360 grid; all eight
atmospheric fields are finite. Monthly weights are six times 5/31 plus 1/31.
SST and ADT finite fractions over 60°S–60°N model surface cells are 93.95% and
94.19%. These are cell-count support figures for one sample, not full-record or
area-weighted coverage claims. Interior arrays have the expected 2 × 14 depth
channels. [Full shapes, ranges and checksum](materialized-sample-audit.json) retain
the sample's provenance. No training benefit follows from this preprocessing check.

For the same 26-bin sample, deriving geostrophic velocity from coarsened ADT and
comparing it with separately coarsened DUACS velocity gives **0.0656 m/s vector
RMSE**, versus 0.1673 m/s reference RMS speed (39.2%). This retains the existing
geostrophic kernel and common finite support, excluding 5°S–5°N. Coarsening and
spatial differentiation do not generally commute; this is a processing-consistency
diagnostic, not forecast error or an irreducible bound. Keep the published metric
reference definition fixed for all candidates and include this caveat in result
interpretation. [Recorded diagnostic](coarsening-consistency.json).
