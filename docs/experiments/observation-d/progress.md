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

## Data verified; Torch QoS change and beta qualification

Materialization **96584** completed in 33:25; publication **96619** completed in
57 seconds. All 243/9/96 monthly examples plus grid/statistics are published:
**350 NPZ files, 21,888,642,966 bytes (20.4 GiB)**. OSN full read-back found zero
differences across the 352 payload/manifest files. Torch verified every NPZ hash
and wrote `DATA_READY.json` at **01:21:44 UTC** on 23 September. The live pre-copy
quota guard passed with 1.32 TB conservative free space after its rounding margin.

Torch jobs were submitted: fitting **18294932**, primary **18294934**, adapter
**18294935**, scratch fitting **18294939**, scratch **18294946**, and evaluations
**18294974–18294976**. The fitting job is pending because the live `gpu48` QoS now
has a per-user GPU maximum of **zero**. Earlier qualification used that same QoS
successfully. A test-only `gpu168` request was rejected as inappropriate for the
job; its restriction is not bypassed by inflating requested wall time.

A beta fallback qualification, **204384**, is submitted as **one host × four GPUs**
under `ny_lz1955_multiscale`, QoS `test`. It uses the already prepared EAI data and
the identical SHA-verified D checkpoint transferred via Torch DTN → OSN. The pinned
ARM64 image has prior one-host training evidence; this pilot still requires its
own live qualification. The independent-arm coordinator passes two subprocess
sequencing/device/failure-isolation tests. It can run transfer and scratch arms
inside one four-GPU allocation, with evaluation on the fourth GPU. This fallback
is not yet a successful fitting or skill result, and Torch production has not run.

Beta's live account association permits `priority` QoS. Test-only requests for
one-host/four-GPU qualification and production were accepted with that QoS.
Qualification **204384** was updated to `priority`, 15 minutes; production
**204388** is queued with `afterok:204384`, `priority`, one host/four GPUs and a
12-hour allocation cap. Production source is
`e57997d2c42f2a8b57d7d245c81ec7ce114ac579`, staged read-only on EAI. The coordinator
runs the same bounded optimization arms, with scratch concurrent with transfer
arms and completed-arm evaluation on the fourth GPU. No production has started.
At approximately 01:30 UTC, beta's scheduler forecast qualification around
10:30–11:10 UTC; forecasts are provisional and imply only early progress may be
available for the 14:00 UTC report.

Torch rejected attempts to hold the duplicate pending production jobs with an
unspecified scheduler error. Those seven never-started jobs were then cancelled;
queue inspection confirms only Torch fitting **18294932** remains pending.
No running experiment was stopped. If Torch becomes available first, its fitting
check remains useful, but production will use only one selected backend.

## Torch reopened; real-data fitting passed and production restored

At 01:35 UTC the live `gpu48` limit had risen to 16 GPUs per user. Fitting
**18294932** completed successfully in **1:06** (01:33:33–01:34:39 UTC). The fixed
May 1993 training-sample objective decreased from **0.461634 to 0.063591** over ten
updates, and finite gradients reached initializer, evolution and adapter. Peak GPU
allocation was **5.57 GiB**. All **27** requested validation spectral components
qualified. The initial pretrained/zero-adapter validation composite was **2.494985**.
These observations qualify optimization and metric support; they are not held-out
skill or evidence that the ten-update probe generalizes.

Never-started beta jobs **204384** and **204388** were cancelled before restoring
Torch production. Their fallback implementation remains available, but no beta
training or qualification ran. A CPU-only validation-control diagnostic was also
submitted as **18295846**; it does not train or select any checkpoint.

The first Torch resubmission hit an expired completed-job dependency. The launcher
now retains live dependencies and omits an expired prerequisite only after `sacct`
proves `COMPLETED` with `0:0` exit status; five regression cases cover live,
successful, failed and missing prerequisite evidence. Existing fitting qualification
and data/reference-hash guards remain enforced by the trainer.

Restored jobs submitted at 01:38 UTC: primary **18295884**, adapter-only **18295885**,
scratch fitting **18295886**, scratch **18295887**, primary evaluation **18295888**,
adapter evaluation **18295889**, scratch evaluation **18295895**. The production
model source remains `d8949bb15d9fa92e6c8c94702092360bbe9005ef`; only submission
recovery changed. Prior cancelled submission records are preserved in a separate
history directory. Keep at most two production GPUs active as originally planned.


## Production is running on Torch

Primary **18295884** and adapter-only **18295885** both started at **01:44:12 UTC**
on 23 September, each on one H200. By 01:46 UTC, both had entered the adapter
warm-up and completed actual optimization steps with finite loss and gradients.
Online W&B runs are [primary](https://wandb.ai/ocean_emulators/observational-transfer/runs/af7af77a1573)
and [adapter-only](https://wandb.ai/ocean_emulators/observational-transfer/runs/fdca5f8e2b72).
There is not yet a post-adaptation validation result at this capture.

The independent CPU climatology diagnostic **18296459** completed in 44 seconds.
It reproduced all 27 spectral components and the surface integrated control errors
exactly; the two OHC errors differed from the GPU reference by less than 2e-8
relative, consistent with reduction precision. The production GPU reference remains
unchanged. The seasonal control composite is **1.3151773**, compared with the
initial pretrained/zero-adapter **2.4949851**. Initial pretrained OHC errors are
8.46 and 11.67 times climatology, while velocity and EKE errors are 0.726 and
0.857 times climatology. These opposing components make reporting the individual
metrics essential; they do not establish any fine-tuning outcome.

Earlier CPU diagnostic attempts failed before computation because the standalone
batch used an unavailable module name and then the container's system Python.
The successful job uses the existing Apptainer binary and `/workspace/.venv/bin/python`,
matching the production environment. These diagnostic launch errors did not affect
training or its fixed selection reference.


## Post-selection reporting staged

`analyze_observation_predictions.py` now supplements the selected-checkpoint
exports with pooled area-weighted anomaly correlation, RMS anomaly amplitude and
bias for SST/ADT at days 5/15/30 and monthly OHC. Both prediction and reference
subtract the same training-only seasonal climatology. A constant climatology
prediction has zero anomaly amplitude and undefined correlation, recorded as null.
The report requires all 96 test origins and identical reference arrays across
methods. It also recomputes the existing integrated and spectral scorer separately
for each of the eight held-out calendar years, retaining the frozen validation
normalization. Paired annual composite differences quantify year-to-year variation;
they are explicitly not confidence intervals or eight independent trials.

Three targeted tests passed, covering known anomaly scaling, missing prediction
rejection, degenerate climatology, all eight paired years and changed-support
rejection. No production training or selection code changed. CPU-only reporting
jobs **18297298** (primary), **18297299** (adapter-only) and **18297300** (scratch)
are submitted after their respective held-out evaluations, each capped at 30 minutes,
four CPUs and 32 GiB. Report implementation `eea0ccde1` records its script and input
hashes. Results are pending actual evaluation exports.


## First post-adaptation validation, 01:58 UTC

Both runs completed their first 100-update adapter validation. Primary composite
**2.4536713**, adapter-only **2.4535973**, versus source **2.4949851**; both still
trail the seasonal-control composite **1.3151773**. The two arms have not yet
unfrozen different weights, so their close warm-up results are expected and are
not a full-versus-frozen-core comparison.

Primary SST error increased from **1.0430 to 1.1722 °C**; geostrophic vector error
fell from **0.14280 to 0.14131 m/s**; EKE error changed from **0.020119 to 0.020075
m²/s²**. OHC 0–700 m increased from **6.1155e9 to 6.2284e9 J/m²**, while
700–2000 m decreased from **4.2532e9 to 3.9871e9 J/m²**. Mean spatial spectral
error changed from **0.40146 to 0.40523 dex**. Thus the small composite improvement
contains real tradeoffs; training loss alone would not show them. No held-out
result is available yet.

Immutable evidence:
`/scratch/jr7309/runs/2026-09-22-observation-D/snapshots/adapter100-20260923T0158Z.json`.
Component and day-30 spectral plots were generated and visually inspected from
that capture. Three bins per regional spectrum span approximately 600–3000 km.
They must remain labeled broad-scale diagnostics.


## Reconstruction adaptation started, 02:10 UTC

Both live jobs completed 200 adapter updates in approximately 24 minutes and
entered reconstruction. Primary **18295884** now updates the initializer and
adapter; adapter-only **18295885** keeps both pretrained networks frozen. At
02:10 UTC, primary had completed seven reconstruction updates with finite losses
and gradients, and adapter-only eleven. These losses are optimization diagnostics,
not held-out accuracy.

Warm-up selected composite: **2.4074208** primary, **2.4074118** adapter-only.
Primary mean spectral error **0.40339 dex**, versus initial **0.40146 dex**.
The selected score still trails seasonal climatology (**1.3151773**). The next
reconstruction validation will be the first test of updating the pretrained
initializer. Immutable capture:
`/scratch/jr7309/runs/2026-09-22-observation-D/snapshots/reconstruction-start-20260923T0210Z.json`.

Live scratch quota at 02:05 UTC: **3.69 TB / 5 TB** (about **1.31 TB** free).
Prepared samples occupy 21 GiB, pilot run outputs 4.1 GiB. No cleanup has been
needed or performed.


## First preemption and automatic requeue

At **02:15:28/29 UTC**, Torch preempted primary **18295884** and adapter-only
**18295885**. Both show `Restarts=1`, `Requeue=1` and are pending again under the
same job IDs; this is scheduler preemption, not a numerical or application failure.
At 02:20 UTC the final recorded reconstruction steps were 48 and 56 respectively.
Both have `reconstruction-last.pt` recovery checkpoints (primary written 02:15,
adapter-only 02:14). The completed adapter phase and selected-checkpoint records
remain intact. No manual duplicate submission was made; all dependent scratch,
evaluation and reporting jobs remain attached to the existing job IDs. Actual
resumption from saved optimizer/RNG state still needs live verification.

The snapshot collector now requests duplicate Slurm accounting records so that
preempted attempts remain visible alongside the requeued job, rather than only
showing its latest scheduling state.


Recovery metadata was inspected directly in the PyTorch ZIP archives using
`pickletools` (without loading tensors or executing pickle constructors). Primary
saved reconstruction step **48**, elapsed **368.905 s**; adapter-only saved step
**52**, elapsed **367.030 s**. Thus the primary's last recorded update is retained,
and the adapter-only job will repeat four updates after resumption. Both records
retain their best score, patience counter and RNG payloads. Successful tensor and
optimizer loading still requires observing the resumed job.


## Recovery verified and first initializer-adaptation comparison

Both jobs restarted at **02:34:09 UTC**, on `gh108` (primary) and `gh115`
(adapter-only), under their original IDs. The first resumed updates were exactly
primary reconstruction **49** and adapter-only **53**, following the inspected
recovery states. W&B resumed the same two run IDs. This verifies execution resumed
from the saved phase/optimizer/RNG path rather than repeating adapter warm-up.

At reconstruction update 100, primary validation composite was **0.8544912** and
adapter-only **2.3929534**. Primary OHC errors were **1.04123e9 J/m²** (0–700 m)
and **7.15992e8 J/m²** (700–2000 m), compared with initial source errors of
6.11550e9 and 4.25319e9 J/m². Primary SST error was **1.16884 °C**, geostrophic
vector error **0.138371 m/s**, EKE error **0.0200167 m²/s²**, and mean spatial
spectral error **0.442292 dex**. Adaptation has greatly reduced the source's
interior mismatch; spectral error has increased from the initial 0.401463 dex.

The primary beats climatology's **composite** (1.3151773), not every component.
Its errors divided by climatology errors are **1.371 SST**, **0.703 velocity**,
**0.853 EKE**, **1.441 upper OHC**, and **1.965 lower OHC**. Thus climatology still
wins SST and both OHC RMSEs, while the primary's much better spectral structure
than climatology, velocity and EKE offset those deficits in the frozen score.
Do not reduce this to an unconditional skill claim. These are nine-month validation
forecasts after reconstruction training; joint forecast training and held-out
reporting have not yet occurred.

Immutable capture:
`/scratch/jr7309/runs/2026-09-22-observation-D/snapshots/reconstruction100-20260923T0243Z.json`.
Updated comparison figures are generated locally from that capture.


## Reconstruction update 200, 02:56 UTC

Both jobs remain running after recovery. Primary selected composite **0.8493170**,
adapter-only **2.3838305**. Primary OHC errors improve further to **9.27860e8**
and **6.42251e8 J/m²**, but SST error rises to **1.21098 °C** and mean spectral
error to **0.48993 dex**. The small composite improvement over update 100 therefore
continues the OHC-versus-surface/spectral tradeoff. Joint forecast training has not
yet begun. Capture: `snapshots/reconstruction200-20260923T0256Z.json` beneath the
Torch run root.

The plotting tool now accepts `--split test` for completed 2015–2022 evaluations.
It requires a completed training selection, matching selected-checkpoint hashes,
all 96 origins and matching spectral references. Held-out component bars use
held-out climatology errors for descriptive normalization; this does not modify
the frozen validation selection score. Four report tests pass, including refusal
to relabel incomplete or mismatched evaluations as held-out results.


## Analysis-only control moved earlier under Torch flexibility

At 02:58 UTC the scratch control was scheduled independently of primary/adapter
completion so it has more training time before the report. This uses at most a
third Torch GPU, consistent with the user's clarification that the one-host/four-GPU
constraint applies to beta and Torch layout is flexible. Every arm retains its
original step/hour cap; no extra experiment or duplicate training was added.

Torch rejected changing the pending dependency with an unspecified scheduler
error. Only never-started scratch-chain jobs **18295886**, **18295887** and
**18295895**, plus pending CPU reporters **18297300** and **18297299**, were
cancelled and replaced. Their records are preserved in
`retired-submissions-before-parallel-scratch`. Active primary **18295884** and
adapter-only **18295885**, and their GPU evaluation dependencies, were untouched.

Current scratch fitting **18303300** has no outstanding dependency; scratch
**18303301** waits for its qualification, and scratch evaluation **18303304** waits
for scratch training and adapter evaluation. CPU report jobs are now adapter-only
**18303306** and scratch **18303307**; primary reporting remains **18297298**.

A reporting-path mismatch was found before execution: the existing adapter GPU
evaluation writes `adapter-evaluation`, whereas the new CPU reporter and plotting
helper had expected `adapter-only-evaluation`. Both now use the actual path; the
pending adapter CPU reporter was replaced with the corrected batch script. Nine
focused reporting/submission tests pass, including a regression check that the
adapter evaluation is included. Producer code for all training remains `d8949bb15`;
launcher/reporting correction is `580e61feb`.


## Scratch qualification passed; second scheduler preemption

Scratch fitting **18303300** completed **03:04:07–03:04:55 UTC**, exit 0. The
training-only probe loss decreased **1.291290 → 0.242077** over ten updates, with
finite gradients reaching initializer, evolution and adapter; peak allocation
**5.568 GiB**. Its manifest records `normalization_mode: observation-only` and
`source_checkpoint_sha256: null`. Initial fresh-weight validation composite was
**6.758859**. Production scratch **18303301** is eligible and queued; the probe's
fitted weights are not used to initialize production.

Primary and adapter-only were preempted again at **03:05:08/09 UTC**, after about
31 minutes of their second allocations, and automatically requeued. Last observed
reconstruction steps were **282 / 302**; ZIP metadata confirms recovery steps
**275 / 300**, so seven and two updates respectively will repeat. Latest selected
scores remain primary **0.849317** (update 200) and adapter-only **2.376090**
(update 300). No numerical failure was observed.

Alternative routing was checked without submitting a duplicate job. The permitted
regular RTX6000 allocation has an eight-GPU group limit and all eight are allocated.
A complete `sbatch --test-only` request allowing normal plus preemptible H200
capacity routed to `h200_public` with a forecast start after the snapshot deadline.
That forecast is provisional; it does not justify replacing the currently
productive, resumable preemption-only jobs. Keep following their existing IDs.


## All three arms running; scratch first validation

The next allocations began at approximately **03:14 UTC**. Primary resumed with
reconstruction update **276** and adapter-only with **301**, exactly after their
saved states. Scratch **18303301** began from fresh weights; its live manifest
confirms observation-only normalization and no source checkpoint hash. W&B:
[analysis-only scratch](https://wandb.ai/ocean_emulators/observational-transfer/runs/e035b83d0956).
At 03:28 UTC the three live arms had recorded reconstruction updates **320**,
**416**, and **106**, respectively.

Primary update 300 scored **0.8496353**, slightly worse than update 200's
**0.8493170** despite further OHC improvements. Selection correctly retained
update **200**. At update 300, SST error was **1.23592 °C** and mean spectral error
**0.51472 dex**, while OHC errors fell to **8.70841e8 / 6.12856e8 J/m²**.
Adapter-only selected update 400 at **2.3693335**.

Scratch's first 100-update validation composite was **6.4633877**, versus initial
**6.7588586**. SST error was **8.75904 °C**, and mean spectral error **4.06762 dex**.
This is expected to be a weak forecast at this stage: the scratch evolution
network is still randomly initialized and frozen during reconstruction training.
Do not interpret this interim difference as an established pretraining benefit;
compare after joint forecast optimization and report actual completed budgets.

Immutable capture: `snapshots/all-arms-training-20260923T0329Z.json` under the
Torch run root. No held-out evaluation has begun.


## Depth-resolved report diagnostics

The plotter now accepts `--grid` to draw monthly forecast temperature/salinity
RMSE and bias against physical depth, separately from surface/spectral charts.
It verifies the supplied grid's SHA-256 against the fitting manifest. The locally
cached grid matches exactly; the new figure was rendered and visually inspected
using the immutable reconstruction-100 snapshot. These are monthly **forecast**
errors after reconstruction training, not direct reconstruction scores.

That early diagnostic shows substantial mean-state correction: temperature bias
at **105 / 165 m** changes from **+3.742 / +3.821 °C** in the source to
**+0.100 / +0.135 °C** in primary update 100. Corresponding RMSE drops from
**4.532 / 4.526 °C** to **0.884 / 0.779 °C**, but remains above seasonal
climatology's **0.541 / 0.453 °C**. At 2.5 m, practical-salinity bias changes from
**−0.3843** to **+0.0127**, while RMSE falls **0.9833 → 0.2383**, still above
climatology's **0.1240**. These observations support interpreting the initial OHC
gain as substantial mean-state adaptation; they do not alone establish superior
dynamical skill. Refresh the figure from the morning-selected checkpoint.


## Metric selection continues to reject worse later checkpoints

At 03:43 UTC all three arms were still running. Primary reconstruction update
400 scored **0.8606395** with mean spectral error **0.53280 dex** and SST error
**1.27118 °C**; OHC errors were **8.54340e8 / 6.10656e8 J/m²**. Update **200**
remains selected at **0.8493170**, rather than following the improving interior
training objective. Adapter-only update 500 is selected at **2.3632209**.
Scratch update 200 scored **6.4840486**, so its update **100** remains selected at
**6.4633877**. The fixed patience/update/hour limits continue unchanged.

Morning capture must preserve any still-active arm's selected checkpoint under
an immutable filename and verify the copied bytes against its best-record SHA-256.
A running job may replace its live `best.pt` after the report, so the filename alone
is insufficient snapshot provenance. Completed-arm selections and immutable JSON
captures already provide stable metric records.


## Third transfer preemption; first scratch preemption

At **03:45:29 UTC**, all three allocations were preempted and automatically
requeued. Retained reconstruction checkpoints are primary **400 updates / 3840.43 s**,
adapter-only **546 / 3991.46 s**, and scratch **223 / 1751.31 s**. Last recorded
updates were 420, 550 and 233, so 20, 4 and 10 updates will repeat. Selected metric
records are unchanged. Snapshot: `snapshots/preemption-20260923T0351Z.json`.
The three completed transfer allocations were each about 31 minutes; this describes
observed scheduling, not a guaranteed allocation duration or policy.


## Held-out pipeline no longer waits unnecessarily for a slower arm

At 04:10 UTC primary had resumed on `gh114` and reached reconstruction step 441;
adapter-only and scratch remained pending on priority. With three Torch GPU slots
now permitted, the old barrier delaying primary evaluation until adapter-only
finished is unnecessary. The launcher now releases primary evaluation after
primary training alone. Adapter evaluation explicitly requires its own completed
training plus completion of primary evaluation; scratch evaluation requires its own
training plus adapter evaluation. This retains serialized evaluation and at most
three concurrent GPUs. Two DAG tests verify each evaluation's own-training gate
and the maximum concurrent-job bound in both the two- and three-GPU modes; all
seven submission tests pass.

Only never-started evaluations and CPU reporters were replaced. Current GPU eval
jobs: primary **18310184**, adapter **18310185**, scratch **18310186**. CPU reports:
primary **18310192**, adapter-only **18310194**, scratch **18310196**. Training IDs
and all model/selection settings are unchanged. Prior submission records remain
in `retired-submissions-before-independent-evaluation`. Launcher revision
`29fb1b83c`; all GPU producers remain pinned to `d8949bb15`.


## Held-out control coverage verified on CPU

CPU-only job **18311289** completed in **1:40**, exit 0, and wrote
`cpu-seasonal-test.json` at **04:34:57 UTC**. It evaluated training-only seasonal
climatology over all **96 held-out target months (2015–2022)**. All **27 frozen
spectral components** are supported. Using the unchanged validation denominators,
the held-out control composite is **1.3780513**. Integrated errors: SST
**0.903905 °C**, velocity **0.200221 m/s**, EKE **0.0309310 m²/s²**, OHC 0–700 m
**8.38172e8 J/m²**, OHC 700–2000 m **3.72852e8 J/m²**. This is a control and
coverage audit, not a trained-model held-out result. No training, key set or
selection weight changed after inspecting it.

The CPU helper records its own script hash, the pinned metric producer, data hash
and frozen selection-reference hash. The snapshot collector now includes both
CPU control diagnostics; capture `snapshots/heldout-control-20260923T0436Z.json`
contains this result. Post-selection CPU reports additionally include physical
pointwise SST/ADT RMSE alongside anomaly correlation/amplitude and bias; the new
RMSE calculation passed its known-scaling test and was staged while all three
report jobs were still pending.


## Primary joint forecast training started

Primary reconstruction stopped automatically at **700 updates**, with **6157.53 s**
(1.71 hours) retained phase time and **five non-improving validation checks**.
The phase restored update **200**, selected at **0.8493170**, before starting
joint optimization. Its first joint update completed at **04:55:03 UTC** with
finite loss and gradients. Initializer, evolution and adapter are now trainable;
the fixed joint forecast objective retains the predeclared 0.1 reconstruction term.
This is a training-stage transition under the original protocol, not a manual
change prompted by held-out results.

Adapter-only and scratch resumed again on `gh124` and `gh129` and remain in
reconstruction. Snapshot: `snapshots/primary-joint-start-20260923T0455Z.json`.
The report plotter also now provides matched-lead SST/velocity/EKE error curves,
so the average over days 5/15/30 does not hide lead-dependent behavior. The new
figure was rendered and inspected on recorded validation evidence.

## Controls that isolate dynamics from the selected initializer

At **05:03 UTC**, the pending evaluation DAG was replaced with producer
**5b8d391230e9a0b97afd232520335c8036363248**. It adds persistence and anomaly
persistence initialized by each arm's **selected** initializer, preserving that
arm's normalization. Existing source-initializer controls remain. This separates
forecast-dynamics skill from the benefit of correcting the inferred starting
state; it does not change training, checkpoint selection, or metric definitions.
The seven-method CPU report includes both new controls, with paired annual
differences on the fixed validation-normalized composite.

Training jobs and producer **d8949bb15d9fa92e6c8c94702092360bbe9005ef** are unchanged.
New GPU evaluation jobs: primary **18314442**, adapter **18314443**, scratch
**18314444**. Their CPU reporting jobs are **18314449**, **18314450**, and
**18314452**, respectively. All six replaced jobs were verified pending with zero
runtime before cancellation; records are preserved under
`retired-submissions-before-selected-state-controls`. Dependencies still bound
Torch GPU concurrency to three and require each arm's completed selection.

Thirteen focused tests passed, including selected weights and scratch
normalization across control evaluation, immutable protocol fingerprints on
resume, unchanged training producers, and DAG concurrency. Pre-commit checks
passed. The evaluation overlay's dependency manifests match the existing SIF;
its SHA-256 is
`11f2e39b0b151de570df76b27b1fb26bfbd01d1b91d2f8f0344e3982cc913599`.
Core training, model, data, and metric modules are unchanged from the training
producer. All seven methods will be reported; the overview plots retain a
smaller set of curves, with the original controls explicitly labeled source-state.

## First joint validation

At **05:10:31 UTC**, primary joint update **100** became the selected checkpoint,
with composite **0.7424292** (previous reconstruction selection **0.8493170**).
Snapshot `snapshots/primary-joint100-20260923T0512Z.json` records all components
and the still-running jobs. Selected checkpoint SHA-256:
`d0f0675a60adad38aba141c9fb06d56e14a1761a21f8a4c3926e2dbc6e044410`.

| Validation component | Joint 100 | Seasonal climatology |
| --- | ---: | ---: |
| SST RMSE, °C | 0.612139 | 0.852329 |
| Geostrophic velocity RMSE, m/s | 0.143203 | 0.196765 |
| EKE RMSE, m²/s² | 0.0207353 | 0.0234677 |
| OHC 0–700 m RMSE, J/m² | 7.48995e8 | 7.22737e8 |
| OHC 700–2000 m RMSE, J/m² | 4.51636e8 | 3.64326e8 |
| Mean spatial spectral error, dex | 0.563752 | 1.630355 |

SST RMSE at days 5/15/30 is **0.443752 / 0.620396 / 0.772268 °C**; each
improves on reconstruction step 200. The spectral error increased from **0.489933**
to **0.563752 dex**, while SST and both OHC errors fell enough to improve the
unchanged composite. Both OHC errors remain above seasonal climatology. This is
validation evidence, not held-out skill or proof that every component improved.
The matched-lead figure was regenerated and visually inspected from this capture.
Adapter-only is selected at reconstruction 900 (**2.3471170**); scratch remains
selected at reconstruction 100 (**6.4633877**) before its joint dynamics training.

At **05:22 UTC**, all three arms had reached joint training. Scratch reconstruction
stopped at update **600** after five non-improving checks (4647.93 retained seconds),
restored update 100, and reached joint update **51** with finite loss/gradients.
Adapter-only completed reconstruction at update **1000** (7204.68 retained seconds),
selected **2.3468604**, and reached joint update **13**. Primary was preempted after
observed joint update **133** and was pending automatic requeue under the same job
ID; its selected joint-100 result remains intact. This is another scheduler
interruption, not an experiment failure. Capture:
`snapshots/all-arms-joint-20260923T0522Z.json`.

## All arms have joint-validation evidence

By **05:54 UTC**, all three jobs had resumed on `gh118`, one allocated GPU each.
Primary joint update **200** selected **0.7060693**, adapter-only joint update
**100** selected **2.3393796**, and scratch joint update **100** selected
**1.6767055**. Mean spectral errors are **0.529542 / 0.416356 / 1.781731 dex**,
respectively. The adapter's relatively good spectrum does not offset its large
interior/OHC errors under the fixed composite. Scratch's large improvement from
its reconstruction-stage score demonstrates why that earlier random-dynamics
result was insufficient for a pretraining-benefit claim.

Primary integrated errors at joint 200: SST **0.580489 °C**, geostrophic velocity
**0.142043 m/s**, EKE **0.0206008 m²/s²**, OHC **6.92470e8 / 4.27745e8 J/m²**.
The upper OHC error is now below validation climatology's **7.22737e8 J/m²**;
the deeper error remains above its **3.64326e8 J/m²**. These remain selected
validation results, with production and held-out evaluation unfinished.
Capture: `snapshots/all-arms-joint-validated-20260923T0554Z.json`.

A fresh **05:33 UTC** `myquota` check reported scratch **3.70 TB / 5 TB** and
1,646,874 of 5,000,000 files. The compact dataset occupies **21 GiB** and the pilot
run tree **17 GiB** (`du -sh`, rounded). No cleanup or full-source transfer was needed.

## Joint update 400 comparison

Capture `snapshots/joint400-20260923T0649Z.json` records primary and scratch at
selected joint update **400**, scoring **0.6693107** and **1.0427907**. Scratch
first crossed below the climatology composite (**1.3151773**) at joint update 300.
Its SST and both OHC errors still exceed climatology, so this is not an
across-the-board improvement. Primary's mean spectral error is **0.488466 dex**,
scratch's **0.807396 dex**. Primary integrated errors: SST **0.557140 °C**,
velocity **0.140177 m/s**, EKE **0.0203540 m²/s²**, OHC **6.45310e8 / 4.09691e8
J/m²**. Adapter-only remains at **2.3374584**, selected at joint update 300.

The reporting plotter now includes a joint-stage forecast-validation trajectory
using every recorded validation check, deduplicating repeated updates after
requeue. It was rendered and visually inspected from this immutable capture.
Its horizontal axis is retained joint updates, explicitly not matched GPU time;
the arms also differ in initialization, normalization and earlier phase budgets.
The continuing decline, particularly for scratch, argues against calling these
bounded runs converged baselines. No held-out model result is available yet.

At **07:21 UTC**, primary had reached joint update 619, selecting update **600**
at **0.6441195**. Its mean spectral error is **0.465091 dex**; integrated errors
are SST **0.545617 °C**, velocity **0.138740 m/s**, EKE **0.0201660 m²/s²**, and
OHC **6.21156e8 / 3.83172e8 J/m²**. The deeper OHC error is still slightly higher
than climatology. Adapter-only and scratch were pending on scheduler priority
after preemption around 07:02 UTC, with no Slurm start estimate. Their last
observed joint updates were 476 and 495; no jobs or budgets were replaced.
Capture: `snapshots/primary-joint600-20260923T0722Z.json`.

## Primary completed its approved training budget

Primary job **18295884** completed successfully at **08:34:49 UTC** (Slurm
`COMPLETED`, exit 0), after **200 adapter, 700 reconstruction, and 1000 joint
updates**. It selected joint update 1000 at **0.6141372**. The last validation
still improved, so reaching this cap does not establish convergence. Summing
the ten allocation attempts, including nine preemptions, gives **18,443 GPU
seconds = 5.1231 allocated GPU-hours**. Retained phase clocks were
1447.92 / 6157.53 / 9812.83 seconds; allocated time also includes startup and
replayed work.

The final selected checkpoint SHA-256 was independently read back and matched
both `best.json` and `TRAIN_COMPLETE.json`:
`11631058d74f0de699ee58951c6884c1471720a15ec633d29fcf872d680a012e`.
A checksum-verified copy and selection metadata are preserved under
`selected-checkpoints/primary-<sha256>.pt` and `.json` in the run root.

Final selected validation components: SST **0.534409 °C**, geostrophic velocity
**0.136382 m/s**, EKE **0.0198038 m²/s²**, OHC **5.95806e8 / 3.77586e8 J/m²**,
mean spatial spectral error **0.423321 dex**. Day-5/15/30 SST errors are
**0.421613 / 0.551435 / 0.630177 °C**. The deeper OHC error remains above
climatology. The other components and spectrum are below it. This is completed
validation selection; **held-out evaluation 18314442 is pending on priority**.
Adapter-only and scratch remain active within their original budgets.
Capture: `snapshots/primary-complete-20260923T0837Z.json`.

## Evaluation launcher recovery

Primary evaluation **18314442** failed after **17 seconds** at **08:44:23 UTC**,
before loading the model or creating evaluation outputs. The pinned container's
`torchrun` parser treated evaluator option `--run` as an ambiguous abbreviation
of `--run-path` / `--run_path`. All five downstream jobs were automatically
cancelled with zero runtime. Their submission records and terminal accounting
are preserved in `retired-submissions-after-torchrun-argument-failure`.

Launcher commit **08eecaee0** adds an explicit `--` separator before the
evaluator's arguments. The exact pinned-container argument parser reproduced
the failure and accepted the corrected argument list. Nine focused tests passed,
including forwarding through the torchrun parser and unchanged training/evaluation
producers. An optional full `torchrun --help` probe on the login host did not
launch a child and was stopped; it is not counted as runtime qualification.
The real GPU allocation will provide that qualification.

At **08:50 UTC**, replacement evaluations were submitted as primary **18328848**,
adapter **18328849**, scratch **18328850**; CPU reports are **18328851**,
**18328852**, and **18328853**. Primary training completion was independently
proved with accounting; the other dependencies retain the original training IDs.
Training producer **d8949bb15d9fa92e6c8c94702092360bbe9005ef**, evaluation producer
**5b8d391230e9a0b97afd232520335c8036363248**, weights, cohorts, and metric definitions
are unchanged. The scheduler confirms H200, one GPU per evaluation and requeue
enabled. No model job was stopped or restarted for this launcher fix.

## All training complete; anomaly-control compatibility repair

Adapter-only completed at **08:58:50 UTC**, selecting joint update **800** at
**2.3343760** after 200/1000/1000 phase updates and **5.1000 allocated GPU-hours**.
Scratch completed at **09:20:31 UTC**, selecting joint update **900** at
**0.9128040** after 600 reconstruction and 1000 joint updates and **4.1950 allocated
GPU-hours**. Both jobs exited 0. All three selected checkpoints and metadata have
checksum-verified copies under `selected-checkpoints/`; adapter SHA-256 is
`6756d13b81ad8df3b5ee24d9d7a34cc5f5e8dd9b78fe6a62fe8be4057a1a791b`, scratch is
`4d2c15b4dba7ea6869ea9b3884653481aed998fcf52206f42f88af8af0800b78`.
Total allocated training time across the three arms is **14.4181 GPU-hours**,
excluding qualification and evaluation. Capture:
`snapshots/all-training-complete-20260923T0922Z.json`.

Primary evaluation **18328848** passed the launcher and completed the selected
forecast and selected-initializer persistence over all 96 held-out origins, then
failed in anomaly persistence at **09:08:28 UTC**: container pandas rejected a
NumPy Unicode scalar passed to `Timestamp`. Producer **7a8923203** converts that
scalar to a Python string. A year-boundary regression checks seasonal adjustment,
normalization, unchanged input, and preserved surface/unsupervised channels;
nine focused tests and a functional run in the exact pinned container passed.
Training and scoring code paths are unchanged. Completed method files are reused,
with their hashes and original **5b8d39123** producer recorded in
`primary-evaluation/reused-output-provenance.json`.

Replacement evaluations: primary **18329670**, adapter **18329671**, scratch
**18329673**. CPU reports: **18329674**, **18329675**, **18329676**. The failed chain
and its zero-runtime cancelled descendants are preserved in
`retired-submissions-after-anomaly-timestamp-failure`. Evaluation overlay SHA-256:
`083a445a28cddaf76dacd8482a80c3d0450fb0574ee2815b601e1415d49f194e`.

The two completed held-out methods already show a tradeoff: primary forecast
composite **0.6578108**, selected-state persistence **0.6223205**. Forecast SST
RMSE is **0.504931 versus 0.791236 °C** and velocity RMSE **0.140903 versus
0.193295 m/s**, but spectral error is **0.424640 versus 0.215470 dex**. Under the
frozen weighting, persistence is ahead despite worse SST/velocity errors. This
is a partial comparison; anomaly persistence and the remaining arms are unfinished.
The overview plots now use persistence controls from the selected initializer;
source-initializer controls remain in the complete tables. Four reporting tests
passed, including completed-cohort and selected-checkpoint gates.

## Completed primary/adapter held-out evaluation and operator audit

Primary held-out evaluation **18329670** completed at **09:33:28 UTC**, followed
by its CPU anomaly/year report **18329674** at **09:36:00 UTC**. Adapter evaluation
**18329671** completed at **09:47:12 UTC**, and report **18329675** at **09:48:58 UTC**.
All exited 0 and cover all 96 origins and seven methods. Primary selected forecast
composite is **0.6578108**, selected-initializer persistence **0.6223205**, and its
interior-anomaly persistence **0.6333984**. Both persistence controls beat the
learned forecast's composite in all eight calendar-year rescoring checks. Primary
beats seasonal climatology (**1.3780513**) in all eight years. Annual variation is
not a confidence interval. Adapter-only selected forecast is **2.4573085**;
its selected-initializer persistence is **2.3260292**. Scratch evaluation remained
queued at the 09:49 capture. Live scratch quota then was **3.71 / 5.00 TB**;
no source deletion or raw-store transfer was needed.

A reporting-only CPU audit uses target-time observed SST/ADT in the unchanged
scoring pipeline, with training climatology at missing surface cells. Job
**18330224** completed in 17 seconds and confirmed zero direct SST/OHC and SST/ADT
spectral errors. This is not a forecasting candidate or a new selection criterion.
Its full-support velocity RMS disagreement is **0.160342 m/s** and EKE spectral
error **0.635343 dex**; median day-30 EKE power is **0.2420** of the DUACS reference
across the nine region/bin comparisons. Thus much of the EKE power mismatch is
already present without a learned forecast.

The stencil audit's initial assumption that DUACS u/v masks match failed loudly
in **18330253** (15 seconds); component-wise support is required by the actual
scorer. Producer **bb7c3b787** preserves those separate masks and asserts that the
partitioned full-support RMSE reproduces the unchanged scorer. Job **18330279**
produced the qualified partition: complete observed gradient stencils cover
**94.621%** of weighted component/origin support and have **0.0729–0.0731 m/s** RMS
disagreement; the remaining **5.379%** has **0.6186–0.6207 m/s**, about 80% of the
full-support squared error. Filling is verified not to change a complete observed
stencil. The report retains this as a limitation of physical interpretation and a
future mask/operator decision; training, validation selection and held-out scores
were not changed. Outputs and all submission records are retained under the run root.

## Final morning snapshot: all planned runs and reports complete

Scratch evaluation **18329673** completed at **10:06:59 UTC** (12m38s), and its
CPU report **18329676** at **10:08:56 UTC** (1m54s), both exit 0. It selected joint
update 900 on validation, then scored **0.9827232** on all 96 held-out origins.
Its selected-state persistence scored **0.7170391**, and interior-anomaly
persistence **0.7208893**. Every arm's selected-state persistence beats its own
learned forecast composite in all eight annual rescoring checks. Full fine-tuning
is the strongest learned forecast under these bounded budgets; this does not
establish convergence, a general pretraining advantage, or operational skill.

The final immutable evidence capture finished at **10:10:27 UTC (06:10 ET)**:
`snapshots/final-report-20260923T1010Z.json`. No jobs remained in the queue.
Prepared data occupied 21 GiB and the run tree 29 GiB; the 10:11 UTC quota check
reported 3.71 / 5.00 TB used. No deletion was performed.

The [snapshot report](snapshot-2026-09-23.md), [technical methods](snapshot-2026-09-23-methods.md)
and [artifact index](artifacts/2026-09-23/README.md) contain all 21 arm/method
rows, physical component errors, integrated-plus-spectral scores, anomaly/year
diagnostics, validation trajectory, spectra and selected-checkpoint provenance.
The EKE spectral panels include the target-time observed-ADT operator diagnostic,
clearly separate from forecasting candidates. Plotting verified common references
and selected checkpoint hashes; four report tests passed. Full live operator
partitioning reproduced the unchanged velocity scorer and checked that filling
cannot alter a complete observed stencil. The score/selection protocol was never
changed after inspecting held-out results. No follow-up training was launched.

## Requested follow-up: rollout maps and model description

Added fixed January/July 2022 day-30 surface maps from saved predictions: SST and
ADT absolute fields for July, and training-climatology anomalies for both seasons.
All candidates share target-valid masks and fixed color limits; no new forecast
or training was run. CPU extraction job **18331812** completed successfully in
31 seconds. It checked all source NPZ hashes against the completed CPU reports,
verified identical observation slices, and produced a 2,814,917-byte bundle. The
local copy matched SHA-256
`fa75429ad70040ed73f3b919664062e5424596ae9673a030303ee976bcf13d47`.
The figures record source/checkpoint lineage and include observations, full fine-tuning,
scratch, adapter-only, unadapted D and matched persistence. Day 30 is the last common
exported surface lead, not a continuous multi-year rollout endpoint.

Direct model instantiation on the meta device counted **121,684,010** initializer,
**31,631,677** evolution and **387** adapter parameters: **153,316,074** total.
Both full fine-tuning and scratch train that full parameter set in the joint phase;
adapter-only trains 387. The methods now describe architecture widths, channel
interfaces, normalization, optimization and phase-specific trainable counts.

## Requested follow-up: exact grid-cell pixels

Regenerated all six maps as lossless PNGs with 720 × 240 pixel panels for
360 × 120 grid locations: exactly 2 × 2 pixels per cell, without interpolation.
All 36 panels were checked pixel by pixel against the source cell colors,
including masked cells. Updated native-grid PDFs and report links. Predictions,
checkpoints and scores are unchanged; no new training or inference was needed.

## One-seed update-allocation wave submitted — 23 September, 17:53 ET

Authorized the four D allocation arms plus the observation-only random control,
with the latter extended to 8,000 joint updates to assess plateauing. Producer
`5b3456699` is pushed on `codex/d-observation-pilot`; the Torch overlay was built
and checksum verified on a CPU node. No input data transfer or scratch cleanup was
needed. Live quota was 3.71/5.00 TB used. The login node's full temporary filesystem
was avoided by building on a CPU node; an initial abbreviated-ref fetch was fixed
by using the full commit ID. Neither build attempt used GPUs.

Root: `/scratch/jr7309/runs/2026-09-23-observation-budget`.

| Stage | Slurm job | Scheduled work |
|---|---|---|
| Calibration | 18376773 | Two 200-update rate trials each for OM4, transfer and scratch, with fitting checks |
| OM4 prefix | 18376774 | 3,000 joint updates, retaining 1,000/2,000/3,000 terminal checkpoints |
| D → observations | 18376775 | 1,000 observation reconstruction + 4,000 joint updates |
| Observation-only random | 18376776 | 1,000 reconstruction + 8,000 joint updates; immutable 4k/6k/8k snapshots |
| D → 25% OM4 → observations | 18376777 | 1,000 reconstruction + 3,000 observation joint updates |
| D → 50% OM4 → observations | 18376778 | 1,000 reconstruction + 2,000 observation joint updates |
| D → 75% OM4 → observations | 18376779 | 1,000 reconstruction + 1,000 observation joint updates |

Initial scheduler verification: calibration pending for priority, all later stages
pending dependencies; H200 routing, one GPU per job, requeue and preemption comment
confirmed. This records submission, not observed training or successful completion.
The dependency graph permits at most four concurrent GPUs. Requested wall caps
sum to 81 GPU-hours; reserve 19 for recovery within the proposed 100-hour envelope.
Final evaluations are included in production stages, with separate random-4k and
random-8k selected-checkpoint evaluations. Selection remains integrated-plus-spectral
validation; the previously viewed historical test does not select models. All
23 focused tests and full repository hooks passed before publishing the producer.

### RTX routing update — 23 September, 17:57 ET

Following the user's preference for RTX access, replaced all seven pending H200
jobs with RTX PRO 6000 jobs on account `torch_pr_347_lzanna`, partition
`rtx6000_lzanna`. All superseded H200 jobs were cancelled with zero elapsed time.
The training producer and update budgets are unchanged; only the launcher routing
and output root changed. New root:
`/scratch/jr7309/runs/2026-09-23-observation-budget-rtx`.

| Stage | Replacement job |
|---|---|
| Calibration | 18376929 |
| OM4 prefix | 18376930 |
| D → observations | 18376931 |
| Observation-only random, extended | 18376932 |
| D → 25% OM4 → observations | 18376933 |
| D → 50% OM4 → observations | 18376934 |
| D → 75% OM4 → observations | 18376938 |

Calibration started on `gr102` at 17:57:19 ET. The immutable code layer and W&B
online mode were verified in the actual job log. Production remains dependency-
gated; first optimizer-update bring-up is checked separately below. A Slurm
`--test-only` request was accepted before replacement; its predicted next-day
start proved pessimistic, so queue estimates were not reported as guarantees.

RTX bring-up verified at **18:00:10 ET**: the first OM4 calibration trial completed
optimizer update 1 with finite loss **0.0262172**, effective batch eight, peak GPU
memory **73.61 GiB**, and peak host RSS **5.54 GiB**. Both training and validation
resident caches passed exact native-data equivalence checks. This establishes
single-RTX OM4 training compatibility; calibration and production are not yet
complete. Observation-only fitting remains separately qualified by the staged job.

## Hourly monitoring enabled — 23 September, 18:06 ET

User requested roughly 60-minute monitoring, sooner when appropriate. Enabled local
user timer `oe-obs-budget-monitor-01a0d027.timer`, which queues checks into this same
Codex conversation rather than creating a separate agent. The service was tested:
Codex acknowledged queued message `01a0d04e-3e6a-7f43-982d-10344d47387b`. The timer
is active; next routine wake is **19:06:23 ET**, repeating hourly. A pending marker
prevents overlapping monitor prompts. The monitor must clear it after each check
and disable the timer when reporting is complete. Scheduling relies on this local
host and its Codex service remaining available.

Monitoring covers scheduler/accounting, structured training and validation logs,
checkpoint/evaluation completeness, accumulated GPU-hours, and routine recovery
within the approved single-seed scope and 100-GPU-hour envelope. It does not
authorize further experiment waves. Latest live check: OM4 LR 3e-5 calibration
completed 200 updates with finite validation T/S MSE 0.00337271; the LR 1e-4
calibration was running. Production remains held behind calibration completion.

## Scheduled check: Torch access unavailable — 23 September, 20:07 ET

The authenticated Torch control socket was absent. One ordinary SSH connection
attempt remained unresponsive for about a minute and was terminated locally. No
remote scheduler query returned, no authentication retry loop was started, and no
cluster jobs were modified. Current training state and accumulated GPU-hours are
therefore **unverified**, not inferred from the previous healthy snapshot. The
hourly timer remains enabled; user assistance may be needed to restore Torch access.

## Monitoring change requested — 23 September, 20:09 ET

Stopped and disabled `oe-obs-budget-monitor-01a0d027.timer` and removed its pending
marker at the user's request to avoid repeated context injection. Subsequent checks
will use sleep within the existing conversation, normally at hourly intervals.
The user is restoring Torch access before the next check and authorizes Slack
notification if a blocker remains. Training jobs and the scientific protocol are
unchanged.

## Restored access and cancellation recovery — 23 September, 20:37 ET

Authenticated Torch access is restored. Accounting shows calibration 18376929
was canceled by UID 0 at 20:00:18 ET after 7,379 allocated GPU-seconds
(**2.0497 GPU-hours**); Slurm records a termination signal but no explanatory
reason. All six downstream jobs canceled without allocating GPUs. This was not
an observed numerical failure. Both 200-update OM4 trials and the transfer fitting
qualification completed. Transfer LR 1e-5 completed all 1,000 reconstruction
updates (last validation composite 0.85955, best 0.80211) and logged three finite
joint updates before cancellation. No observation joint validation or final
calibration choice exists yet. These reconstruction scores are not wave results.

Preserved original submission/accounting records in cluster-root
`recovery-20260924T0037Z/`; retained checkpoints, events and source hashes.
Resubmitted the same pinned producer `5b345669965ce5da3fcf6894049df1bd1099a584`,
arguments, RTX resources and dependency graph after scheduler preflight. Existing
exact-command signatures, data manifests and fitting qualification remain valid.
Completed child stages and reconstruction will be skipped; the three uncheckpointed
joint updates will be repeated from reconstruction-best. No code fix or scientific
protocol change was required.

| Stage | Replacement job | Dependencies |
|---|---:|---|
| Calibration | 18385561 | none |
| OM4 prefix | 18385563 | calibration |
| obs0 | 18385570 | calibration |
| Random | 18385576 | calibration |
| obs25 | 18385580 | OM4 prefix |
| obs50 | 18385581 | OM4 prefix |
| obs75 | 18385582 | OM4 prefix and obs0 |

At 20:37 all replacements are **pending**, calibration for priority and production
for dependencies. Maximum concurrency remains four GPUs. Replacement wallcaps
sum to 81 hours; adding the failed attempt gives 83.0497 GPU-hours worst case,
leaving 16.9503 of the approved 100 for further recovery. Scratch usage is
3.72/5.00 TB. Monitoring continues via sleep; the timer remains disabled.

## Low-utilization diagnosis and cache recovery — 23 September, 20:42 ET

User clarified that root cancellation around two hours often indicates GPU
utilization below 50%; this operational guidance was added to the torch-train
skill. The canceled attempt's 492 telemetry samples average **32.64% GPU
utilization**, median 1%; 66.3% of samples are below 50%. This is consistent with
the policy, although no administrator explanation was obtained. A short CPU-only
benchmark on the allocated node measured 0.426–0.437 seconds per compressed sample
read, about 3.4 seconds per effective batch of eight. The resumed joint training
was finite but took about 8 seconds/update.

Stopped the retry and its pending descendants to address the bottleneck. Retry
18385561 consumed 209 GPU-seconds; cumulative allocation is now **2.1078 GPU-hours**.
All original artifacts remain at the original root. The correction caches up to
256 raw CPU samples, covering the 243 training and 9 validation months (20.50 GiB).
Normalization, GPU transfers, sampling, objectives and scores retain their existing
paths. Wave jobs request 64 GiB host RAM for cache/checkpoint headroom.

Use a fresh successor root `2026-09-23-observation-budget-rtx-cache` with a new
pinned producer and fresh calibration/qualification. Do not migrate old producer
qualifications or checkpoints into new manifests. This conservatively repeats
calibration from the same original D checkpoint, resolves the exact-producer resume
contract without weakening it, and keeps the interrupted trials as charged recovery
work. The new 81-hour request caps plus both attempts total at most **83.1078
GPU-hours**, before any further recovery. No scientific protocol changes.

Validation: 29 focused tests passed, including cached/uncached tensor and missing-value
equality, normalization changes, fixed-budget milestones, submission contracts and
the four-GPU dependency bound. Cluster cache throughput and utilization still need
verification; a code change alone does not establish that the bottleneck is fixed.

Cache recovery producer: `79e8e6fde70e27317cfe89f308d0ab1212bcb6c4`. CPU build
18386015 succeeded and submitted the replacement DAG at 20:43 ET. Calibration
18386026 is allocated on gr101; production jobs remain dependency-pending:
OM4 18386029, obs0 18386038, random 18386056, obs25 18386058, obs50 18386059,
obs75 18386060. The final arm depends on both OM4 and obs0, retaining the
four-GPU maximum. New root is `/scratch/jr7309/runs/2026-09-23-observation-budget-rtx-cache`.
All six production jobs are gated by successful calibration, directly or transitively.

## Cache throughput verified — 23 September, 21:05 ET

Calibration 18386026 remains running on gr101. Both OM4 200-update trials and
transfer fitting qualification completed under the new producer. Transfer LR 1e-5
reconstruction reached step 122/1,000 with finite loss 0.01659. Recent ten-step
throughput is **2.82 seconds/update**, versus approximately 6.2 seconds late in
the uncached reconstruction. GPU utilization averages **55.85% over the latest
40 telemetry samples** (roughly ten minutes), and 53.3% over the latest 20.
This verifies a substantial loader improvement, but not immunity to utilization
cancellation; continue checking through the two-hour boundary.

All six production jobs remain dependency-pending; calibration has not selected
rates yet. Accumulated allocated GPU time, including both canceled attempts and
the current 1,305 seconds, is **2.4703 GPU-hours**. Routine monitoring returns to
hourly sleeps within this conversation.

## Hourly check — 23 September, 22:06 ET

Calibration 18386026 is running at 1:22:37 elapsed. Transfer LR 1e-5 completed
1,000 reconstruction plus 200 joint updates; final joint integrated-plus-spectral
validation score is **0.7140466**, also its best score. Transfer LR 3e-5 is at
reconstruction step 22. These are calibration trials, not production comparisons.
`calibration.json` is absent and all six production jobs remain dependency-pending.
GPU utilization is **66.46% over the last hour**, 63.36% over the allocation so far.
Cumulative allocated GPU time including canceled attempts is **3.4847 hours**.
No new failures, retries or protocol changes.

## Hourly check — 23 September, 23:06 ET

Calibration remains running beyond the previous failure boundary, at 2:23:10.
Transfer LR 3e-5 reached joint step 169/200 after completing reconstruction;
its step-150 validation score is 0.68665. Scratch calibration has not begun.
GPU utilization averages 70.35% over the last hour and 66.24% over the allocation.
Cumulative usage is 4.4939 GPU-hours. All production remains dependency-pending;
no calibration-selection marker or production/evaluation completion exists.

## Hourly check — 24 September, 00:07 ET

Both transfer calibrations completed: joint-validation composites 0.7140466
(LR 1e-5) and 0.6712186 (LR 3e-5). Scratch fitting qualification passed and
scratch LR 3e-5 reached joint step 105/200 after its 1,000 reconstruction updates;
step-100 validation composite is 2.13216. Scratch LR 1e-4 remains to be run, so
there is still no final calibration marker and all production stays dependency-pending.
Calibration elapsed 3:23:49, last-hour GPU utilization 70.74%; cumulative allocation
including retries 5.5047 GPU-hours. No new failures.

## Hourly check — 24 September, 01:07 ET

Scratch LR 3e-5 finished its 200 joint updates with validation score 2.0219343.
The final calibration trial, scratch LR 1e-4, finished reconstruction and reached
joint step 32/200, with finite loss. Last-hour GPU utilization is 71.35%;
cumulative allocated usage is 6.5128 GPU-hours. No production has started.
Next check is shortened to verify the expected calibration-to-production handoff.

## Calibration complete — 24 September, 01:28 ET

Calibration 18386026 completed successfully (`COMPLETED`, exit 0:0) after
16,629 seconds. Verified `calibration.json` selects OM4 LR 3e-5 (T/S validation
MSE 0.00337037 versus 0.00343186 at 1e-4), transfer LR 3e-5 (observation
composite 0.6712186 versus 0.7140466 at 1e-5), and scratch LR 1e-4 (1.5404643
versus 2.0219343 at 3e-5). Observation choices use joint integrated-plus-spectral
validation only. Calibration artifact is mirrored under
`artifacts/2026-09-24-budget/calibration.json`.

OM4, obs0 and random are now eligible but **pending `QOSGrpGRES`**, with no
production logs or GPU allocations. The remaining three arms retain their
dependencies. This is a scheduler capacity wait, not a training failure. Total
allocated usage including recovery is **6.7269 GPU-hours**. Continue hourly sleeps.

## Scheduler wait — 24 September, 02:33 ET

All production remains pending. Eligible OM4/obs0/random jobs report `QOSGrpGRES`
with no estimated start; no production logs or completion markers exist. Usage
remains 6.7269 GPU-hours. The requested Slack notification was rejected by automatic approval review and
was not delivered. A shorter notification was also rejected despite checking the
current authenticated Slack profile; explicit destination confirmation was requested
and remains pending. This is a resource
wait, not an SSH or numerical failure. No rerouting or extra
allocations submitted; continue hourly checks without duplicate blocker notices.

## Production routing change — 24 September, 04:34 ET

Repeated hourly checks showed no production start and unchanged 6.7269 GPU-hours.
Live QoS inspection identifies the constraint: partition QoS `rtx6000_lzanna`
caps the pool at eight GPUs, all occupied by two other four-GPU allocations.
The shorter running allocation had more than eleven requested hours remaining;
this is not a completion forecast. H200 preflight estimated a start around
09:07 ET for both the two-hour prefix and 27-hour random request (not guaranteed).

User already authorized either H200 or RTX, preferring RTX when accessible.
Moved only the six verified, zero-elapsed pending production jobs to the existing
H200 preemption route. Archived RTX DAG/submission/accounting records under
`routing-h200-20260924T0834Z/`. The root retains its historical `rtx-cache` name,
but **calibration ran on RTX; production now requests H200**. Report hardware
separately when comparing actual GPU-hours; update matching remains unchanged.

| Stage | H200 replacement | Dependencies |
|---|---:|---|
| OM4 prefix | 18409909 | calibration already completed |
| obs0 | 18409910 | calibration already completed |
| Random | 18409920 | calibration already completed |
| obs25 | 18409921 | OM4 prefix |
| obs50 | 18409922 | OM4 prefix |
| obs75 | 18409924 | OM4 prefix and obs0 |

Verified partition `h200`, account `torch_pr_347_general`, single GPU, 64 GiB
RAM, requeue enabled and preemption comment. All are still pending. Producer
remains `79e8e6fde70e27317cfe89f308d0ab1212bcb6c4`; no training command or
scientific protocol changed. Completed calibration was proved by accounting
`COMPLETED 0:0`; each observation production arm performs a fresh fitting
qualification on its assigned hardware. Maximum concurrency remains four GPUs.
Remaining production wallcaps sum to 71 GPU-hours, so current use plus full
remaining caps is 77.7269 hours within the approved 100.

Publishing this update was blocked by automatic approval review, which requested
explicit authorization to push scheduler/job/quota/hardware/budget details to the
repository. The user explicitly approved publishing these operational reports on 24 September;
publication resumes on `codex/d-observation-pilot`.

## H200 bring-up and automatic preemption — 24 September, 05:36 ET

OM4 began at 04:54 ET; obs0 and random began at 05:04 ET. Both observation
fitting probes passed on NVIDIA H200 with gradients reaching initializer, evolution
and adapter. Finite reconstruction training reached obs0 step 250 and random
step 284 before scheduler preemption at 05:35. Both have atomic reconstruction
checkpoints saved at 05:33 and are requeued automatically. OM4 was preempted
after 1,886 seconds, restarted at 05:34, and is warming its GPU data cache.
These are explicit `PREEMPTED` events, not root-utilization cancellation.

Use `sacct -D -X` from now on: ordinary accounting omits previous allocations
of requeued job IDs. At this check the three preempted allocations consumed
1,886 + 1,818 + 1,818 GPU-seconds, and resumed OM4 consumed 108 seconds.
Total including all prior attempts is **8.2908 GPU-hours**. No more than three
GPUs have run concurrently. Remaining observation arms are dependency-pending.
H200 reconstruction is slower than RTX here (roughly 5–6 seconds/update); recent
utilization was 43.7% for obs0 and 26.1% for random. Monitor resumed utilization
and checkpoint progress; do not assume the RTX speedup transfers quantitatively.

## OM4 complete; four observation jobs active — 24 September, 06:47 ET

Verified OM4 prefix `TRAIN_COMPLETE.json`: step 3,000, finite T/S validation MSE
0.0032690, producer `79e8e6fde70e27317cfe89f308d0ab1212bcb6c4`, final checkpoint
SHA256 `5890f61019b581eee38d5545020e9a9c280acf0006e45d52586d817892c787f6`.
OM4 completed after one preemption and successful resume. Obs0, random, obs25
and obs50 are now running, exactly four GPUs; obs75 remains dependency-pending.
Latest finite reconstruction steps are 894, 385, 380 and 97 respectively.
Obs0 has resumed twice and random once. `sacct -D -X` total, including all
preemptions and earlier cancellations, is **10.2608 GPU-hours**. No observation
production training or evaluations are complete yet.

## Requested live update — 24 September, 07:11 ET

All four started observation jobs are currently pending after verified scheduler
preemptions; obs75 remains dependency-pending. Last logged progress: obs0 finished
1,000 reconstruction updates and reached joint step 204/4,000; random reconstruction
594/1,000; obs25 reconstruction 536/1,000; obs50 reconstruction 302/1,000. These are
logged steps, potentially ahead of the last atomic resume checkpoint. Losses remain
finite; no production training/evaluation completions exist. Total allocation
including every preempted attempt is **11.3711 GPU-hours**. Frequent short H200
allocations are now the main throughput limitation; do not mistake queue time for
training or extrapolate a completed comparison from calibration scores.

User explicitly approved both operational report publication and future Slack
blocker alerts to their confirmed account. No stale capacity notification was sent
after the previous blocker cleared.
