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

## Hourly check — 24 September, 08:10 ET

Obs0 last logged joint step 537/4,000 and is queued after preemption; obs25
and random are running at joint steps 161/3,000 and 152/8,000. All three finished
1,000 reconstruction updates. Obs50 is queued with reconstruction step 858; obs75
is dependency-pending. Latest losses are finite. No production joint-validation
checkpoint has yet reached its scheduled interval, and no production evaluation
is complete. Both four-GPU allocations still fill the RTX quota. Total charged
GPU time including every preemption/retry is **13.9836 hours**. Requeue can reset
Slurm wallcaps, so ongoing enforcement uses cumulative accounting, not only the
sum of original requested limits.

## Hourly check — 24 September, 09:11 ET

Obs0, obs25 and obs50 are running joint training at logged steps 892/4,000,
621/3,000 and 286/2,000; random is queued after preemption at 514/8,000.
All four have completed reconstruction. Obs75 remains dependency-pending.
No production joint-validation interval has been reached yet; no training/evaluation
completion markers exist for observation arms. Losses remain finite. Total
allocation is **16.7447 GPU-hours**, counting repeated job allocations. RTX remains
fully occupied at its eight-GPU pool limit.

## First production joint validation — 24 September, 10:12 ET

Four observation jobs are running. Latest logged steps: obs0 1,506/4,000,
obs25 1,044/3,000, obs50 882/2,000, random 889/8,000. Obs75 is still gated.
First scheduled joint-validation composites are **0.5846329** for obs0 at
1,000 updates, **0.6079953** for obs25 at 750, and **0.6196708** for obs50 at
500. These are different observation/total budgets and are not a final allocation
ranking. Random has not yet reached its first joint-validation interval.
Total allocation including all retries/preemptions is **19.8736 GPU-hours**.

RTX pool occupancy fell from eight to four GPUs, opening quota. Physical RTX
hosts have many unallocated GPUs but tight CPU availability (0–7 cores free per
node). Test-only scheduling estimates about 12:13 ET for both four- and two-CPU
single-GPU requests. No jobs were canceled or rerouted at this check; retain
ongoing H200 work until a useful RTX handoff is established.

## Hourly check — 24 September, 11:14 ET

All four started observation arms are running. Logged joint steps: obs0 1,929,
obs25 1,622, obs50 1,314, random 1,399. Latest joint scores: obs0 0.5846329
at 1,000 updates; obs25 0.5743713 at 1,500; obs50 0.5866827 at 1,000; random
1.0146102 at 1,000. The obs0-versus-random contrast is matched in additional
observation updates, but does not charge the original historical D pretraining;
it is preliminary evidence, not the completed plateau or allocation comparison.
Total all-attempt allocation is **23.1536 GPU-hours**. RTX preflight still estimates
about two hours of waiting, so no running jobs were interrupted for rerouting.

## Hourly check — 24 September, 12:16 ET

All four started observation arms are running. Latest joint steps: obs0 2,471,
obs25 2,095, obs50 1,823/2,000, random 2,045. At matched 2,000 additional
observation updates, obs0 validation composite is 0.5586355 and random is
1.0023157. Obs50 last validated at 1,500 with 0.5601224; obs25 remains at its
1,500-update validation score 0.5743713. No production completion/evaluation
markers yet. Obs75 remains gated on obs0. Total allocated GPU time is
**26.6947 hours**. RTX quota still has four occupied GPUs; no routing change.

## First completed production arm; release obs75 — 24 September, 13:19 ET

Obs50 completed its full 2,000 OM4 + 1,000 reconstruction + 2,000 observation
joint-update pathway. Selected validation score is **0.5560779** at joint step
2,000. Verified successful Slurm completion and `obs50-evaluation/COMPLETE.json`
with 96 historical origins, selected/source checkpoint hashes, and selected-model
plus persistence/climatology output files. This is the first completed arm; it
does not establish the final allocation ranking. At 13:16, obs0 was at joint
2,952, obs25 at 2,871, random at 2,665. Cumulative allocation was **29.8175
GPU-hours**.

With obs50 complete, only three other production arms remain, so obs75's
obs0 dependency is no longer needed for the four-GPU bound. Scheduler rejected
an in-place dependency edit with an unspecified error and left the job unchanged.
Canceled only the zero-elapsed pending obs75 job 18409924, preserved its records
in `obs75-release-20260924T1718Z/`, and submitted replacement **18438483** on
RTX (`rtx6000_lzanna`, one GPU, 64 GiB RAM, same six-hour cap and producer).
Both true predecessors, OM4 and obs50, were independently verified `COMPLETED
0:0` by the submission helper. Existing fitting and exact-command contracts
remain unchanged; obs75 has not trained previously. At 13:19 it is **pending
Resources**, while the other three H200 arms continue. Preflight had estimated
an immediate slot; that estimate did not guarantee a live allocation. No running
H200 job was interrupted. The four-GPU maximum remains intact.

## Concurrency authorization updated — 24 September, 13:56 ET

The user explicitly lifted the four-GPU concurrency cap. Additional GPUs may be
used for useful independent work within this same one-seed wave; the existing
100 allocated GPU-hour ceiling and scientific protocol remain unchanged. All five
observation arms have already launched (obs50 completed; obs75 verified training
on RTX at reconstruction step 152 at 13:29). No new seeds or experiment arms
are implied by this resource authorization.

## Requested ETA and separate 4k evaluation — 24 September, 14:26 ET

Obs25 and obs50 training and historical evaluations are complete. Latest joint
steps: obs0 3,556/4,000; obs75 234/1,000; random 3,303/8,000. Random's
net wall-clock progress is approximately 519 updates/hour over the last hour
and 575/hour over the last two hours, including pauses. This implies roughly
8.2–9.1 hours remaining for its full extension at the observed rate.

Estimated main matched-budget comparison: 16:00–17:00 ET today. Estimated full
scratch-extension/plateau report: 23:00–00:30 ET, subject to further scheduler
preemption. To avoid delaying the main comparison, evaluate the immutable random
4,000-update snapshot in a separate allocation as soon as it exists, using the
same evaluator/producer and output directory that the original driver would
produce after 8,000 updates. Its existing completion check will reuse the finished
evaluation; this adds scheduling parallelism, not a new seed, checkpoint-selection
rule, cohort or experiment arm. Extra allocation time remains charged to the
100-GPU-hour ceiling.

## Report preparation — 24 September, 14:39 ET

User requested completion before the morning. Obs25 completed its full pathway;
selected validation composite **0.5488323** at 3,000 observation joint updates,
with the 96-origin held-out evaluation complete. Obs50 selected validation remains
0.5560779. Current joint steps: obs0 3,695/4,000, obs75 417/1,000, random
3,464/8,000. Random improved to **0.9208738** at 3,000, from 1.0023157 at
2,000; this is not yet a plateau. Cumulative allocation, including all requeued
attempts and earlier failures, is **33.8075 GPU-hours**.

CPU reporting jobs **18447281 (obs25)** and **18447283 (obs50)** are running the
existing anomaly/paired-year diagnostics on completed exports. They retain the
original frozen validation reference, held-out cohort, and training producer
79e8e6fde70e27317cfe89f308d0ab1212bcb6c4. Reporting script SHA-256 is
7649143658163f20ef0ac22bd9cbebd342ded77f86ce923f15921bdaf188243b.
The map exporter now accepts explicit method definitions and reads plot labels
from the audited bundle, preserving the fixed January/July cases and native grid
geometry for the new wave's models.

## Matched-budget comparison complete — 24 September, 16:12 ET

All four transfer arms and the immutable random-4k model completed their historical
96-origin evaluations and CPU anomaly/year diagnostics. Held-out integrated-plus-
spectral scores, ordered 0%/25%/50%/75% OM4 then random-4k, are **0.58605 /
0.59213 / 0.60779 / 0.62667 / 0.93434**. The best transfer recipe is 37.3% below
random-4k, conditional on historical D pretraining; zero versus 25% differs only
1.0%. The first three transfer arms now beat their own inferred-state persistence
in all eight annual checks. Random-4k does not. These are one-seed, update-matched
results; observation updates require many more initializer forwards than OM4.

Separate random-4k evaluation 18453534 failed at launch after 13 seconds because
`torchrun` ambiguously parsed `--run`. Applied the previously tested explicit `--`
separator (documented in the original pilot), preserved the failed submission in
`random-4k-launch-recovery/`, and kept its job in the DAG accounting. Replacement
**18455079** completed successfully in 370 seconds on RTX. No producer or evaluation
protocol changed. The frozen checkpoint SHA-256 is
`aa24e4428c6f9b954b8b104ad2d1fe8d24882172d9bdcf13e573ac16dca5dade`.
The original random driver will reuse the completed evaluation after 8k.

CPU report jobs 18453507/18453508 (obs0/obs75), 18455082 (random-4k), and map
extraction 18455083 all completed successfully. Original CPU reports 18447281/83
also succeeded. Pixel-exact map checks passed. Main report:
[compute-allocation-2026-09-24.md](compute-allocation-2026-09-24.md).
Cumulative allocation is **36.5531 GPU-hours** including retries. Random is still
running, last recorded at joint update 4,221/8,000. Its 4k validation is 0.8803078;
6k/8k extension and plateau report remain pending. Timer stays disabled; monitoring
continues in this conversation with sleeps.

## Earlier D history requested — 24 September, 17:30 ET

Created draft PR #892, stacked on source-model PR #885. User then requested earlier
D errors to extend the learning-curve view backward. Live audit found original
production D best/final weights only; earlier reconstruction validation metrics
remain in logs. A same-recipe separate 12,501-update pilot is retained. No dense
original-production checkpoint sequence or early evolution checkpoints was found
at the recorded run paths; no new training has been launched for this request.

Read-only diagnostic **18460911** completed on RTX in 37 seconds, using the pinned
observation scorer and nine validation origins. Unadapted observational composite:
short pilot **2.4623694**, selected production D at initializer update 43,439
**2.4950400**. Recomputed production score differs from the recorded baseline by
0.0022% across hardware. Both retain the evolution checkpoint selected around
1.26 million earlier OM4 updates. These are separate-run reference points, not a
continuous pretraining curve or a fine-tuned performance comparison.

[Backward-history report](d-training-history.md) includes both same-observation-
metric points and a separately labeled original OM4 reconstruction curve. The
scratch extension continues; its last fresh check at 17:10 recorded 4,766/8,000
joint updates and 37.3728 total allocated GPU-hours before this diagnostic.

## Scratch extension check — 24 September, 18:12 ET

Random is running at joint update **5,253/8,000**. Its 5,000-update validation
composite is **0.8635285**, down from 0.8803078 at 4,000 (1.9% improvement).
Training remains finite; no new intervention. All-attempt allocation, including
37 seconds for the requested historical-D diagnostic, is **38.1928 GPU-hours**.
The frozen 4k comparison is unchanged. Continue to the approved 6k/8k milestones;
a smaller one-interval gain alone does not establish a plateau.

## Conditional 16k extension authorized — 24 September, 18:15 ET

User authorized continuing the same observation-only run to **16,000 joint
updates if it has not plateaued at 8,000**. Preserve the immutable 4k/6k/8k
milestones and their existing selections/evaluations. No new seed or transfer arm.
Review the 6k–8k validation trend before extending; use the same integrated-plus-
spectral objective. The morning report remains a snapshot even if 16k is still
running. The 100 allocated GPU-hour ceiling remains in force.

The pinned pilot enforces exact manifests and completed-phase markers. Therefore
an extension must use an explicitly audited continuation directory, copying the
terminal 8k model/optimizer/RNG checkpoint and selected-best state, with only the
output/name, joint cap and additional milestone list changed. Original 8k outputs
must remain intact. Reconstruction must be reused, not repeated; warm-up/sample
ordering and optimizer state must continue at step 8,000. Verify that contract
before submitting any extension. No running job or training producer changed.

## Scratch-gap diagnostic — 24 September, 18:44 ET

User requested diagnostics of the unexpectedly weak scratch result, explicitly
without seeking the strongest possible baseline. Read-only job **18466058**
completed in 63 seconds on RTX. Same frozen random-4k checkpoint, same nine
validation origins: stored-running-statistics score **0.8800987** versus
**0.5726404** with per-sample BatchNorm statistics. Normal transfer score is
**0.5270783**. About **87.1% of that validation gap disappears without training**.
Transfer instead worsens to 0.9100843 under per-sample statistics, consistent with
its frozen-BatchNorm training recipe. Scratch training-probe objective drops from
0.014433 to 0.002091 when using per-sample statistics; validation objective drops
from 0.016408 to 0.005469. This is a major inference-normalization confound, not
clean evidence of intrinsic observation-only difficulty or an isolated OM4-data
benefit. No official selection, checkpoint or active training policy was changed.

The main report now prominently qualifies the original gap and links
[scratch-gap-diagnostics.md](scratch-gap-diagnostics.md). Full scalar/metric and
script evidence is retained. Follow-up **18467051** localizes the same effect to
initializer versus evolution; this remains bounded read-only diagnosis on fixed
weights. New diagnostics stay charged to the 100-GPU-hour budget. Any inference
policy change to the official comparison would need to be explicit and separately
reported, not silently substituted into the existing table.

## Gap localization and anomaly check — 24 September, 19:04 ET

Read-only localization job **18467051** completed in 41 seconds. Scratch validation
score with only initializer BatchNorm switched: **0.8332313**; only evolution:
**0.5816592**; both (prior job): 0.5726404. Most sensitivity lies in evolution.
Anomaly diagnostic **18467352** completed in 71 seconds. On the same nine validation
months, scratch per-sample BN has day-30 SST anomaly correlation **0.7107** versus
D **0.7118**, with amplitude ratios **0.6646 versus 0.7594**. SST persistence
correlation is -0.0091. ADT correlations remain **0.5741 versus 0.7096**; interior
anomaly amplitude also favors D. This argues against a simple mean-ocean-only
interpretation, while narrowing the plausible remaining pretraining benefit.
Full evidence and qualifications are in the diagnostic report; official results
and training policy remain unchanged.

At 19:03, random is running at **5,588/8,000** joint updates; no new validation
beyond 5k yet. Total allocated GPU-hours including all diagnostic jobs and retries
is **38.9492**. The conditional 16k continuation utility has three passing tests
and all repository hooks passing; it is not yet launched. Continue hourly checks.

## Batch-statistics curves — 24 September, 19:40 ET

Requested all-model retrospective rescoring completed: job **18469640**, 108
allocated GPU-seconds, 12 retained checkpoints, unchanged producer and weights.
Scratch per-sample validation improves from **0.5726404 at 4k** to **0.5645622
at 5k**. All transfer checkpoints were also rescored; per-sample statistics
worsen D, consistent with its frozen-statistics training. Published both the
training-consistent comparison and common per-sample policy in
[batch-stat-curves.md](batch-stat-curves.md); the main report displays the former.
Earlier overwritten checkpoints are explicitly missing; no dense retrospective
trajectory or replacement official selection is claimed. Add 6k/8k as available.

## Report navigation and original D training

Added an explicit PR table of contents and consolidated original D architecture,
data splits, optimizer, selection, update counts and source lineage in the
training-history report. Clarified that the allocation study only explores
continuation after D; earlier pretraining is a justified next hypothesis, not an
established optimum or proof of severe overfitting. No new training authorized
or launched by this documentation change.

## Scratch 6k and report focus — 24 September, 20:25 ET

Immutable 6k batch-statistics validation completed in job **18473594** (35 GPU-seconds): **0.5572997**, improved from 0.5645622 at 5k and 0.5726404 at 4k. Original stored-statistics score is 0.8410361. Training job 18409920 resumed after preemption and is running on gh126. The report now leads with scratch versus D → observations, per user direction; extra-OM4 arms remain supporting evidence. Last full all-attempt accounting was 39.7147 GPU-hours before this short diagnostic. No new training arm or policy change.

## Wave complete — 25 September, 00:25 ET

Random completed 8,000 joint updates and its 96-origin held-out evaluation.
Official best is 6k; original validation 6k/7k/8k = 0.84104/0.84252/0.86768.
Read-only job **18488061** completed in 34 GPU-seconds: per-sample BN at 8k
is **0.5627822**, worse than **0.5572997 at 6k**. No 16k continuation was
launched because the authorized condition of continued improvement was not met.
This is a bounded operational plateau decision, not a universal convergence claim.
CPU audit **18488644** completed in 117 seconds; all six original evaluation
cohorts and selected hashes passed report checks. Total all-attempt allocated
GPU use **42.9886 hours**. Queue empty; timer verified disabled/inactive.
Published [final plateau report](scratch-plateau-2026-09-25.md), updated graphs
and full original-policy evidence. Corrected-policy comparison remains
validation-only; original held-out scratch scores remain normalization-confounded.
Monitoring ends with all authorized runs and evaluations complete.

## Requested checkpoint maps and initializer characterization — 25 September, 09:15 ET

After Torch authentication was restored, read-only RTX job **18511276** completed
in 132 seconds. Added the six retained checkpoint day-30 maps, initial T/S/U/V
maps and nine-validation-month depth profiles to the final report. No unadapted
D point added, per user correction. T550 monthly reconstruction RMSE is similar
(0.21–0.22°C), while scratch U105 RMS is 0.76–0.79 m/s versus D ~0.069 m/s.
Report distinguishes matched monthly reconstruction errors from unlabeled
initial velocity amplitudes and identifies unsupervised deep-state limitations.
All map pixels verified at 2×2 per grid cell. Total including this diagnostic:
**43.0253 allocated GPU-hours**. No new training, selection change or timer.

## Train-only BatchNorm recalibration — 25 September, 09:35 ET

User-authorized diagnostic job **18512340** completed on RTX in 331 seconds.
All 243 training months, forecast path only, cumulative reset BN statistics;
weights and original checkpoints verified unchanged. D3k original/recalibrated/
per-sample scores: **0.52708/0.88636/0.91008**. Scratch6k:
**0.84104/0.84029/0.55730**. Recalibration did not recover scratch's
training-consistent performance. Full held-out recalibration evaluation is not
warranted by these results and was not launched. Report preserves the unresolved
normalization confound rather than attributing the remaining gap to OM4 weights.
All integrity checks passed; total **43.1172 GPU-hours**.


## Fresh InstanceNorm wave — 25 September

User authorized fresh InstanceNorm with original OM4 forcings, ERA5 adapter and
separate observation reconstruction retained. Initial RTX qualification jobs
18528438/18528440 completed (282 GPU-seconds combined): observation fitting loss
1.29126 → 0.24330, OM4 validation normalized T/S MSE 1.20469 → 0.48909.
These short checks do not establish model skill. Final shared-scaling and v3
scoring producer `3dba9696` is queued for qualification via CPU build 18529768.
Annual data preparation completed on Empire AI (101098); OSN transfer awaits
approval. See [the execution plan](instance-norm-wave.md) for contracts and
provenance. No long production runs have started; the old monitor remains off.

Calibration registration 18529840 is queued behind that build. Its gate will
check both final qualification outputs before submitting two 500-update OM4
rate pilots and a dense observation phase/checkpoint probe, capped at three
GPU-hours total. This does not launch main production.


## Approved annual transfer and InstanceNorm advancement — 25 September

Annual bundle publication approved and completed: Empire AI 101147, then DTN
PID1623986, 421 files matched by full read-back and all 416 payload SHA256 checks
passed. No transfer blocker remains. Final-contract qualifications 18530032/33,
dense probe 18530319 and resume 18530614 completed. OM4 rate pilots 18530317/18
completed 500 updates each; LR 3e-4 won (T/S MSE 0.0156985 versus 0.0205135).
Fresh OM4 main 18530880 is queued with 24k/eight-training-hour cap. Scratch
pilots 18530615/16 are running. Evidence-gated continuation is registered for
scratch rate selection and OM4 source export/transfer qualification/calibration;
main observation paths remain 1k reconstruction + 8k joint, one seed.
Evaluation-only producer `8056405ac` adds the continuous-year evaluator; its
validation plumbing qualification 18530921 is queued. See the
[updated plan](instance-norm-wave.md) for exact gates, budgets and semantics.
New-wave all-attempt ceiling 80 GPU-hours; completed probes and OM4 pilots used
0.6494 hours before active/later allocations. Timer remains disabled.

Annual qualification 18530921 completed in 33 seconds with 73 continuous steps,
calendar-month OHC and map exports. This is execution evidence, not a skill
result. Fresh OM4 main 18530880 reached structured per-step training logs.
Dependent monthly/annual evaluation registration is installed for both main
paths, after completed selection. Completed GPU use is 0.6586 hours before
active allocations; the new-wave ceiling remains 80 hours.

State-diagnostic qualification 18532019 completed on the real validation cohort.
The evaluator exports full-grid initializer/reconstruction/day30 fields, profiles,
and forecast changes when initial velocity or deep T/S channels are reset.
Post-training diagnostics are registered at the retained early and late
checkpoints; selected annual checkpoint diagnostics are registered as well.
Training remains pinned to `3dba9696`; these additions use separate evaluation
producers. The execution plan now includes paper-style model/training definitions.


## Observation phase-handoff defect and recovery — 25 September, 14:50 ET

Both scratch LR pilots completed 1k reconstruction + 250 joint, but reconstruction
handoff selected the untrained initializer (forecast score 5.410358) because
random frozen evolution made all reconstruction forecast scores worse. Thus the
phase discarded its learned weights. Their rate-selection results are superseded.
Stopped scratch main 18536182 and retired its dependent evaluation pipeline and
old transfer-export gate 18531062; all files/failed-attempt records retained.
OM4 main 18530880 continues. A tested explicit last-reconstruction handoff will
apply symmetrically to scratch and transfer, preserving integrated-plus-spectral
final selection. All 26 relevant tests passed; replacement observation producer
and fresh qualification/calibration directories will be pinned before resubmission.
At 18:48 UTC, all-attempt use was 4.0272 GPU-hours, including active allocations.

Recovery producer `d186758213bf556456de319bba9e311c604bacd2` passed fitting
18537338, dense-phase 18537397 and resume 18537405. CPU audit 18537484 proved
that joint-zero tensors equal reconstruction-last exactly and differ from the
untrained reconstruction-best tensors. Replacement equal-budget rate pilots
18537519/18537520 were then submitted under `handoff-v2`; all continuation and
evaluation gates were rebuilt there. OM4 pretraining remains on producer `3dba9696`.
The canceled old scratch main used 267 GPU-seconds; all attempts stay counted.


## Corrected scratch calibration completed — 25 September, 16:10 ET

Replacement pilots 18537519/18537520 completed 1,000 reconstruction + 250 joint
updates each. LR 3e-4 won the integrated-plus-spectral score (0.8369917 versus
1.0584885 at 1e-4). Fresh scratch main 18540597 is running under `handoff-v2`;
its evaluation/diagnostic descendants are correctly waiting on completion.
OM4 main 18530880 is beyond two hours and approximately 8,400 updates; recent
utilization averaged 95.6%. The two corrected pilots averaged 72–77%. No blocker.


## Monitoring — 25 September, 17:10 ET

Corrected scratch main 18540597 completed 1,000 reconstruction updates and reached
joint update 229; the latest validation at 200 was 0.9291353. OM4 main 18530880 was
at 11,960 updates. Its selected normalized T/S MSE was 0.0050670; the latest was
0.0065541 with four non-improving checks under the existing six-check rule.
No new failures. All-attempt allocated use, including superseded work, was
**9.7731 GPU-hours**. User scratch quota was 3.96/5 TB, leaving about 1.04 TB.
The annual daily-archive grid and Torch training grid were also verified
byte-identical by SHA256, in addition to the annual payload/time audits.


### 25 September, 21:55 UTC — OM4 source complete; transfer calibration starts

Fresh InstanceNorm OM4 job **18530880 completed successfully** at 14,261 joint
updates, after six non-improving validation checks. Its allocation was 14,597
GPU-seconds (4.0547 GPU-hours). The completion marker reports selected validation
T/S normalized MSE **0.00506701** over eleven valid origins; this is an OM4
metric, not the observation selection score. Immutable source export verified
checkpoint SHA256 `83392ec061400d28eb6a4cd9594376318a6bcf3409f9c9dbecf2bafeedea7557`
and the common observation-derived state scales.

Transfer fitting qualification **18549360** passed under corrected producer
`d186758213bf556456de319bba9e311c604bacd2`: training-only loss decreased from
0.11132 to 0.03932 and gradients reached initializer, evolution and adapter.
Transfer LR pilot **18549413** (3e-5) is running; **18549415** (1e-4) is pending
with `QOSMaxGRESPerUser`, not running. Their selection gate **18549491** waits
for both successful pilots before choosing the observation score minimum and
launching the fresh transfer main run. Scratch main **18540597** continues
normally, past joint update 760. No new scientific arm or seed was added.


### 25 September, 23:54 UTC — both main observation paths running

Live accounting confirms transfer LR pilots **18549413/18549415 completed**.
Their selected integrated-plus-spectral validation scores were 0.656926 (3e-5)
and **0.619043 (1e-4)**. The selector chose 1e-4 and launched transfer main
**18555451** from the immutable selected OM4 weights, not calibration weights.
It is running on gr105, around reconstruction update 200. Scratch main
**18540597** is running on gr101, around joint update 2,486; its latest validation
score is 0.579971 at update 2,400 and its best so far is **0.558058**. These
are different training stages and do not yet establish a matched-budget result.

Both arms have their monthly, annual and checkpoint-state diagnostic descendants
registered with completion dependencies. They remain pending, not completed
evaluations. Training continued during the interruption in assistant responses.


### 26 September, 00:54 UTC — transfer enters joint training

Both main jobs remain running. Scratch is at joint update 3,330, with best
integrated-plus-spectral validation score **0.549892** (latest 0.552740 at
3,300). Transfer completed reconstruction and entered joint training, reaching
update 34. Its early joint score is not a matched-budget comparison with scratch.
No new failed, timed-out or out-of-memory allocations were found. Recursive
submission accounting, including superseded attempts, totals **18.745 GPU-hours**;
`ACCOUNTING_2026-09-26T0054Z.json` preserves the allocation rows. Evaluation
descendants remain pending on training completion.


### 26 September, 06:50 UTC — scratch complete and evaluated

Scratch main **18540597** completed 1,000 reconstruction and 8,000 joint
updates at 06:37 UTC. The selected score is **0.543910**, reached at joint
update 4,900; the final update scores 0.558085. No later validation improvement
was observed, so the conditional 16k extension is not triggered. All nine scratch
evaluation/diagnostic directories now have checked completion markers, including
96 held-out monthly origins, three continuous held-out years, one validation
year, five annual checkpoint diagnostics and dense initializer diagnostics.
The selected checkpoint hash is
`7bb03712e1ee5e8bf46b130895a61b7610498f0d6bfef7d1594df0d4850aa652`.

Transfer main remains running, around joint update 4,764, with best validation
score **0.538717**. At the matched raw 1,000-joint-update checkpoint, transfer
scored 0.573948 versus scratch 0.629016. This is an observation-update comparison,
not a total-compute-matched or replicated estimate. Full comparison awaits
transfer completion and its held-out diagnostics. CPU report-array extraction
job **18573930** reads completed scratch outputs only; no new evaluation or
training protocol is introduced.


### 26 September, 10:58 UTC — InstanceNorm wave complete

Both main runs completed all 8,000 joint updates. Transfer selected update 6,400
(score 0.534191); scratch selected 4,900 (0.543910). All eighteen dependent
evaluation/diagnostic completion markers are present, and the user queue is empty.
All-attempt GPU allocation totals **34.81 hours**, below the 80-hour ceiling.
CPU report extraction jobs 18573930/18575391/18578592 completed; copied report
arrays passed SHA256 verification. The final comparison, compute curves, native-grid
checkpoint maps, profiles and interventions are in
[the completed InstanceNorm report](instance-norm-results-2026-09-26.md).
No 16k extension was needed: scratch showed no new validation minimum after 4,900.
Monitoring for this wave is complete; the old timer remains disabled.


### 26 September — requested regional velocity and spectra figures

Added observed/scratch/OM4-initialized geostrophic U/V and speed maps for the
North Pacific, Gulf Stream and Agulhas metric boxes, plus annual day-30/day-365
velocity comparisons and regional EKE maps. Plots use the saved monthly and
annual EKE spectral curves. CPU extraction **18580257 completed in 23 seconds**
and reproduced all six model/region day-30 EKE curves within numerical tolerance.
Transferred arrays passed SHA256 verification; map pixels retain exact 2×2
representation. No GPU allocation, training or selection changes were required.

### 26 September 2026, three-day follow-up implementation

The bounded goal runs to **29 September 09:30 ET**; see the
[plan](three-day-followup-2026-09-26.md) and
[interim findings](three-day-findings-2026-09-26.md). Existing annual diagnostics
show most of the OM4-initialized model's MSE reduction is reduced temporal-mean
bias; both selected models still lose to training climatology in day-365 SST and
ADT RMSE. This is a diagnostic finding, not new training success.

New single-loop producer `50630959c6a8aed89fcc0152cab28b16f8c377d3` is pushed;
29 targeted tests and repository commit checks pass. Build 18581034 queues its
producer-matched qualifications and dependent mixed-task/resume probe. Original
backbone qualifications were still pending at the latest check; the submission
helper preserves their records and cancels them only if still pending when
superseding them. Production counts/rates remain unfrozen until throughput and
memory are observed. The monitor timer remains disabled. The known preprocessing
ceiling and quarter-degree training remain excluded.

### 26 September 14:28 UTC — mechanism result and qualification repair

The [three-day report](three-day-findings-2026-09-26.md) now contains paired
state/forcing intervention curves. Jobs 18581194/18581195 completed all five
conditions on all three annual origins. Replacing the initial hidden state with
its own training seasonal mean worsens short-lead performance but preserves the
year-end SST gap (about 0.820°C before, 0.825°C after). The long-range benefit is
therefore not explained by case-specific departures of the hidden initialization
from its seasonal mean in these cases. Actual future forcing helps, but the gap
also survives replacing it with training climatology. These are exploratory
sensitivity results, not physical-state identifiability or useful year-ahead
skill versus climatology.

OM4 larger-backbone qualification 18581191 completed 100 updates in 191 training
seconds. Observation qualification 18581192 passed with full gradient reach and
loss 1.2603 → 0.2463. Mixed probe 18581193 failed a tight replay-weight tolerance
while replay losses matched. The failed attempt/checkpoint remain preserved.
Producer `dd3a05b07554ce0c1c9f589643d2afb764ab1440` now checks exact serialization
and restoration separately from measured native GPU arithmetic variation;
CUDA bilinear interpolation backward is documented as potentially nondeterministic.
The revised probe includes actual observation-to-OM4 switches. Eighteen tests and
all commit hooks pass. Build 18582535 submits fresh fitting and mixed qualifications;
no old-producer checkpoint is resumed. Production is still gated.

New campaign accounting at 14:27:49 UTC: **0.36694 allocated GPU-hours**, including
the failed probe and completed diagnostics. All cancelled RTX requests had zero
allocated time. The 120-hour ceiling and 29 September 09:30 ET deadline remain.


### 26 September 15:10 UTC — production and evaluation dependencies

Fresh producer-matched fitting 18582538 and mixed/resume probe 18582539 passed.
The latter verifies exact weight/optimizer/RNG restoration separately from native
GPU numerical variation. Production producer is pinned to
`dd3a05b07554ce0c1c9f589643d2afb764ab1440`; all three main H200 jobs
18583546 / 18583547 / 18583548 have reached finite structured training updates.
The frozen protocol is 16k total updates per arm: scratch 16k observations,
sequential and mixed 8k OM4 + 8k observations. No convergence claim is made yet.

Selected-checkpoint monthly/annual evaluation jobs are 18583774/18583776
(scratch), 18583777/18583779 (sequential), and 18583780/18583781 (mixed).
Each waits for successful corresponding training and independently checks its
completion marker and selected-checkpoint hash.

Fixed-budget evaluation producer `b18821b218f6c9dd00cd1de3fb52babba2b1332d`
passed 16 targeted tests and real-model validation qualification 18584087.
The qualification evaluated nine origins and verified explicit lineage for
five OM4/two observation updates without labeling the raw weights as selected.
The raw 8k-observation monthly/annual jobs are 18584202/18584203 (scratch),
18584205/18584206 (sequential), and 18584208/18584209 (mixed).
Initializer/profile/30-day-map validation diagnostics at 0, 10, 100, 1k, 4k,
8k observation updates (also 16k for scratch) are 18584204, 18584207, 18584216.
Zero denotes random weights before either task, not a pretrained zero-obs model.
All nine jobs depend on successful corresponding training; they add nine
requested GPU-hours. Primary selected evaluations add nine; production caps add
84. Actual campaign use at 15:08:44 UTC was **1.96 allocated GPU-hours**, including
failed/replaced attempts, under the 120-hour total ceiling.

Continuous hidden-state reset diagnostics 18583498/18583499 completed. Resetting
to each model's own training seasonal hidden mean at every forecast step reduced
the year-end SST gap from 0.820 to 0.057 degrees C, but worsened short-lead error
and annual EKE spectral error (scratch 1.96 to 3.20 dex; transfer 1.67 to 2.70 dex).
This supports a hidden-feedback drift explanation while exposing a spectral
tradeoff; it is not an improved model under the selection objective. Full plots,
metrics and intervention limitations are in the findings report. No timer has
been re-enabled; monitoring remains within the three-day wall-time cap.


### 26 September 15:41 UTC — automatic preemption recovery verified

All three production jobs were preempted after roughly 30–35 minutes and Slurm
requeued them on new H200 nodes. They are running with one recorded restart each;
finite updates continued beyond the preserved checkpoint counts. This was recorded
as PREEMPTED, not evidence of low-utilization root cancellation. All evaluation
dependencies remain pending on the same job IDs. No manual restart or producer
change was required.

At this check, scratch had 482 observation updates, sequential had 902 OM4 updates,
and mixed had 621 OM4 + 102 observation updates. The last 40 GPU-utilization samples
(roughly ten minutes) averaged 82.4%, 94.7%, and 76.9%, respectively. Recent throughput
was about 620 scratch, 1,405 OM4-only, and 1,058 early-mixture updates/hour. Scratch's
remaining 15.5k updates project to about 25 hours at that rate, before further
preemption overhead; this is a throughput estimate, not a completion guarantee.
The source-containing arms' rate will slow as observation share grows.

Accounting explicitly includes both allocation records for each requeued job:
**3.3303 allocated GPU-hours** campaign total at 15:41:23 UTC. No completed
production result is available yet, and unequal early exposure is not a fair
performance comparison. Next routine check is around 16:40 UTC.


### 26 September 20:43 UTC — continued training under repeated preemption

All three original main job IDs remain running and resume automatically. Current
counts are scratch 2,717 observation updates; sequential 6,712 OM4 updates; mixed
3,541 OM4 + 1,076 observation updates. Recent integrated-plus-spectral validation
scores are 0.5856 (scratch at 2,700 observations) and 0.6495 (mixed at 1,070
observations); sequential has not reached observations. These unequal exposures
are progress measurements, not a final lift comparison.

There have been seven, six and five preemptions for scratch, sequential and mixed,
respectively. All allocation records are retained in accounting. Total campaign
use is **17.1525 GPU-hours** at 20:42:48 UTC, including repeated allocations.
Healthy uninterrupted rates remain roughly 600 observation updates/hour and
1,400 OM4 updates/hour, but reload/queue overhead makes wall-time completion
slower. The earlier 25-hour scratch estimate was uninterrupted training time,
not a guarantee despite preemptions. The three-day deadline still leaves room.

The preferred RTX partition still has its shared 128-CPU quota fully occupied;
some physical GPUs/CPUs are free, which does not make that quota available. No
migration, seed, optimizer, producer, or scientific-protocol change was made.
All selected/fixed-budget/profile evaluation jobs remain dependent on production
completion. The next hourly check will include the sequential transition to
observation training if it has been reached.


### 26 September 21:54 UTC — sequential task transition verified

Samudra2 sequential completed its 8,000 OM4 updates and reached 77 observation
updates with finite loss and gradients. The unchanged single-loop producer carries
the optimizer across the task boundary; no reconstruction handoff, selected-weight
reload, or new phase job was introduced. This verifies operational transition,
not final transfer benefit. At the 21:44 accounting snapshot, campaign use was
19.8431 GPU-hours including every preempted allocation. Scratch was at 3,187
observations and mixed at 3,886 OM4 + 1,265 observations; all three remained live.


### 26 September 22:48 UTC — early source-task retention evidence

Added raw validation retention evidence to the findings report. Sequential OM4
T/S MSE increases from 0.00473 at the 8k OM4 boundary to 0.66556 after 100
observations, while mixed remains at 0.00591 with 4,287 OM4 + 1,513 observations.
The report labels these unequal-exposure, unselected snapshots and does not infer
long-range observation benefit from source retention. It also distinguishes
initializer/processor changes from a demonstrated physical mechanism.
At the 22:45 accounting snapshot, total use was **22.6397 GPU-hours**, including
all preemptions. All three jobs remained running; held-out production evaluation
is still pending.


### 27 September 00:44 UTC — Torch access unavailable

The hourly SSH check failed with `No route to host` for the Torch login host.
This is a monitoring/connectivity failure, not evidence that training terminated.
No job was cancelled or resubmitted. The authorized Slack blocker alert was sent
to Jesse. Cluster mutations are paused while access is unavailable.

Last successful snapshot was 26 September 23:42:44 UTC: scratch had reached
4,188 observations and was temporarily queued after another preemption;
sequential and mixed were running at 8,000 OM4 + 802 observations and 4,666 OM4
+ 1,778 observations, respectively. Total accounted allocation then was
**25.2136 GPU-hours**. Current counts, allocation use and job state are unverified.
The report deadline remains 29 September 09:30 ET; the goal is not complete.


### 27 September 01:06 UTC — access restored, all jobs progressing

Authenticated Torch access is restored. All three original production jobs are
running and producing finite updates: scratch 4,607 observations; sequential
8,000 OM4 + 1,256 observations; mixed 5,074 OM4 + 2,104 observations. Expected
raw milestone files through those counts are present. All fifteen downstream
evaluation/diagnostic jobs remain pending on their original training dependencies.
No manual restart, producer change, or protocol change was required.

Actual campaign accounting is **28.8247 allocated GPU-hours**, including all
preemptions. The goal resumes with its original 29 September 09:30 ET deadline;
hourly monitoring continues with the old timer disabled.


### 27 September 05:09 UTC — authenticated access failed again

The hourly SSH check terminated with `Too many authentication failures` after
permission-denied messages. Further SSH attempts are paused, and an updated
blocker alert was sent in the existing Slack conversation. No job mutation was
attempted. This does not establish a training failure.

Last successful snapshot, 04:08:34 UTC: scratch had 5,619 observation updates;
sequential 8,000 OM4 + 2,508 observations; mixed 5,714 OM4 + 2,721 observations.
Scratch and sequential were running; mixed was temporarily queued after
preemption. Total allocated campaign use then was **35.8289 GPU-hours**. Current
state and additional allocation use are unverified. The existing three-day
report deadline and scientific protocol remain unchanged.


### 27 September 11:12 UTC — authenticated monitoring restored

Authenticated access is restored again. All three production jobs are running
with fresh finite updates; no manual restart was required. Counts: scratch
8,225 observations, sequential 8,000 OM4 + 4,742 observations, mixed 6,932 OM4
+ 4,474 observations. The separate raw scratch-8k file exists, along with the
expected earlier milestones for every arm. All fifteen dependent evaluation
jobs remain queued. Current allocated campaign use is **52.6992 GPU-hours**,
including 26/22/24 preemptions for scratch/sequential/mixed.

Scratch still has about 7,775 updates remaining. Its cumulative internal elapsed
time is 14.53 hours; projected completion is close to the 27-hour internal cap.
Monitor that boundary explicitly: a partial marker must not be mistaken for
completion, and dependent evaluators must retain their completion/lineage guards.
The original 29 September 09:30 ET report deadline and 120 GPU-hour ceiling remain.


### 27 September 14:18 UTC — report learning curves prepared

Added reproducible in-progress validation curves versus observation exposure and
total updates, plus a separate OM4-retention panel. Raw replayed validation events
are retained; best-so-far curves and current selected counts are explicit. The
figure was rendered and visually checked. The report does not treat unequal
unfinished endpoints as a final performance comparison, or the old/new recipe
comparison as an isolated architecture ablation.

At 14:16:45 UTC, all main jobs were running: scratch 10,093 observations;
sequential 8,000 OM4 + 6,503 observations; mixed 7,548 OM4 + 5,978 observations.
Actual total use was **61.8292 GPU-hours**, including preemptions. Production
held-out results are still pending.


### 27 September 20:15 UTC — source-containing production arms complete

Sequential completed at 18:29 UTC and mixed at 20:03 UTC, both at exactly 8k OM4
+ 8k observation updates. All ten downstream monthly/annual/fixed-budget/state
jobs completed. Selected checkpoint hashes match training completion, best.json,
monthly outputs and annual inputs. The report now contains the two-arm held-out
comparison and raw-final cross-check: mixed retains the source task but has worse
monthly and annual observation error than sequential in this schedule. Scratch
remains running; its comparison and plateau conclusion are not finalized.

The same seven previously implemented state/forcing interventions are submitted
for the new selected models using unchanged diagnostic producer 83119442c3d:
18668686 (scratch, dependent), 18668687 (sequential), 18668688 (mixed). Requested
additional caps total three GPU-hours within the original 120-hour ceiling.
They are exploratory fixed-model diagnostics, not additional training or selection.
CPU compact-array extraction 18666514 completed in 18 seconds for sequential,
reproducing all saved regional day-30 EKE curves; mixed extraction is 18668647.
Transfer of those arrays awaits explicit approval after an automatic review block.

### 27 September 20:27 UTC — production feedback diagnostics complete

Sequential/mixed mechanism jobs 18668687/18668688 completed in 296/307 seconds.
Their input checkpoint hashes match the selected production models, and baseline
annual errors reproduce the main evaluations. Continuous hidden-state seasonal
replacement narrows the year-end SST gap from 2.290 to 0.055°C, but worsens annual
EKE spectral error from 1.231 to 2.923 dex (sequential) and 1.424 to 2.642 dex
(mixed). Initial-only replacement does not eliminate the gap. Full numerical
evidence and interpretation are now included in the report; no model was retrained
or reselected from these diagnostics.

Scratch is running after another automatic preemption/requeue, replaying from
its last saved validation checkpoint: latest update 12,618 at 20:27:23 UTC.
Cumulative internal elapsed time is 21.99 hours; the 27-hour boundary remains
under review. Total allocated campaign use is 76.6519 GPU-hours including all
recorded attempts. Report-array transfer approval remains pending.

### 27 September 21:33 UTC — runtime recovery prepared, not launched

Scratch is live at 13,056 observation updates, with 22.75 cumulative internal
hours; its selected score remains 0.540782 at update 7,900. Repeated preemption
and checkpoint replay make the 27-hour internal stop likely before 16k.
Prepared a narrowly scoped runtime-recovery launcher and 14 passing focused
tests (including the existing joint-training tests). It runs against the
**unchanged original qualified training overlay**, restores through the existing
strict constructor, and gives the runner at most three extra cumulative hours.
It does not alter optimizer state, task ordering, update target, science arguments,
selection, the absolute deadline, or old checkpoint manifests. A separate hashed
RUNTIME_RECOVERY record discloses the override; the original partial marker,
manifest and resume checkpoint are archived before continuation. Existing
fixed-checkpoint evaluation contracts therefore remain exact. This is operational
recovery within the 120 GPU-hour campaign ceiling, not a new scientific arm.

No recovery job has been submitted. Before launch, verify the original job is
terminal with a real time-limit partial marker, refresh total GPU accounting,
and rewire dependent evaluations to wait for actual recovered completion.
The completion marker must still come from finishing the original 16k loop.

### 28 September 00:05 UTC — protect evaluation dependencies at the runtime cap

Scratch reached 14,180 updates (24.72 cumulative internal hours), still selecting
update 7,900. The runtime-only launcher is committed as 474922e44 and uploaded
under a content-hash directory; its module entry point was checked in the original
dd3a05b07554 overlay. No runtime recovery has been launched.

Slurm rejected holds on all six unstarted scratch evaluations with `Unspecified
error`; subsequent queue inspection confirmed they remained pending on the
original training job. Cancelled those six pending jobs instead: 18583774,
18583776, 18584202, 18584203, 18584204 and 18668686. Verified that all disappeared
from the queue. Their submission records remain intact. Replacement evaluations
must be submitted with new names after full completion or with a dependency on
the recovery job. Training was not cancelled. The cluster-side
SCRATCH_EVALUATION_HOLD.json preserves original dependencies and the failed hold
attempt; its filename does not imply that a hold succeeded.

### 28 September 03:06 UTC — scratch runtime recovery submitted

Original scratch job 18583546 terminated successfully at the operational cap,
with a genuine TRAIN_PARTIAL marker at 15,504 observation updates; it did not
complete the scientific budget. Fresh accounting was 82.4706 allocated GPU-hours.
Submitted runtime recovery **18687482** (currently queued for priority), keeping
the original dd3a05b07554 training overlay and both qualification contracts.
Launcher 474922e44 is separately pinned by SHA-256; the cumulative internal cap
is now 30 hours, with the original absolute deadline and 16k target unchanged.
Its Slurm allocation cap is four hours. The launcher archives the old resume
checkpoint and partial marker before strict restoration; successful completion
still must be produced by the unmodified training loop.

Replaced the six cancelled, never-started evaluations with afterok dependencies
on 18687482: selected annual 18687491; raw-8k annual 18687492; raw-8k monthly
18687493; mechanisms 18687494; selected monthly 18687495; initializer diagnostics
18687496. Module arguments and evaluator producers are unchanged. Their original
accounting records show zero allocated seconds. Cluster records
SCRATCH_RUNTIME_RECOVERY_SUBMITTED.json and SCRATCH_EVALUATION_REPLACEMENTS.json
preserve the chain. No queued phase is yet claimed running or scientifically
complete. Remaining requested recovery/evaluation caps fit within 120 GPU-hours.

### 28 September 03:29 UTC — recovery restored and producing updates

First recovery submission 18687482 failed before model execution because torchrun
parsed the launcher's `--run` as its own abbreviated option (13 allocated GPU
seconds). Its six afterok evaluations were cancelled by invalid dependency without
running. Added the conventional `--` argument separator; verified the corrected
entry point locally through torchrun. No launcher/training source change was needed.

Replacement recovery **18687867** is running on gh130 and emitted finite updates
15,529–15,531. It resumed the existing W&B run. The live manifest SHA remains
identical to the original; RUNTIME_RECOVERY.json records the 27→30-hour operational
override, original partial step 15,504, and archived resume hash. Both qualifications
were accepted by the unchanged strict original constructor.

Fresh dependent evaluations: selected annual **18688308**, raw-8k annual
**18688309**, raw-8k monthly **18688310**, mechanisms **18688311**, selected monthly
**18688312**, initializer diagnostics **18688313**. All wait for 18687867. The
failed and cancelled submission records remain available and are charged in
campaign accounting where allocation occurred.

### 28 September 04:32 UTC — monitoring blocked by authentication again

The hourly check ended with permission-denied messages and `Too many
authentication failures`. Further SSH attempts are paused under the Torch access
policy, and Jesse was alerted through the authorized Slack DM. This is an access
failure, not evidence that the running recovery or dependent evaluations failed.
Last verified at 03:29 UTC: recovery 18687867 was producing finite updates at
15,531 and its six replacement evaluations depended on completion. No job was
mutated in response to the access failure. Current completion and additional GPU
allocation are unverified. The original report deadline remains unchanged.

### 28 September 11:27 UTC — all production training and evaluations verified complete

Authenticated Torch access is restored. Recovery 18687867 completed at 04:16 UTC
with exactly 16,000 observation updates and a real TRAIN_COMPLETE marker. All six
replacement evaluations 18688308–18688313 completed successfully. The selected
checkpoint hash agrees across the completion marker, best.json, monthly outputs,
annual inputs and mechanism diagnostics. The archived partial record remains;
the live partial marker was removed after completion. Runtime recovery metadata
is included with scratch evidence. Campaign accounting totals **83.7039 GPU-hours**,
including failed attempts, preemptions, and all numerical evaluations.

Scratch's best score remained 0.540782 at update 7,900 through the full 16k budget.
Its monthly test composite is 0.596874, versus sequential 0.567688 and mixed
0.591432. Selected year-end SST RMSE is 2.2711 / 2.1854 / 4.4749°C respectively.
The report now distinguishes a modest sequential monthly benefit from a weak,
checkpoint-dependent year-end SST benefit. Scratch also has better annual EKE
spectral agreement despite worse velocity RMSE. Updated three-arm learning,
initializer-profile and intervention figures include scratch and its 16k endpoint.

CPU compact-array extraction for scratch is job 18708203. Final map assembly
still awaits the separately requested transfer approval; no model-output transfer
has bypassed that review block. The scientific protocol, one seed and original
report deadline are unchanged.

### 28 September — report and map collection completed

The user explicitly approved direct authenticated SSH transfer of the compact
report arrays. Copied 40 NPZ files (137,480,082 bytes total) and their manifests;
verified every byte count and SHA-256 locally. Scratch extraction job 18708203
completed successfully. All three extraction manifests confirm reproduction of
the original monthly regional EKE spectral curves.

Added the complete map collection and plotting source: early/later initializer
SST, 550 m temperature, velocity-designated and SSH channels; corresponding
30-day SST/ADT rollouts; raw scratch 8k versus 16k; selected annual forecasts
against observations for all three test origins; and regional velocity, EKE and
exact-bin spectra. Thirty-seven map sheets passed native-cell geometry checks
(1:1 global and 2:2 regional). Representative maps and spectral figures were
visually inspected; all report artifact links and repository size limits pass.
The full provenance audit is committed beside the figures. No raw checkpoints
or full evaluation stores were transferred.

The completed report distinguishes the modest sequential monthly performance
benefit, the 8k–16k scratch selection plateau, the lack of a robust annual SST
benefit, and mixed-task retention without better observation forecasts. Maps and
spectra clarify that scratch's stronger small-scale variability can coexist with
worse pointwise velocity error; unconstrained initializer slots remain diagnostic.
No new training arm, seed, task embedding or quarter-degree experiment was added.
All campaign training/evaluations are complete. The original monitor timer is
verified disabled and inactive. Final recorded GPU use is 83.7039 hours; this
report finishes before the original 29 September 09:30 ET deadline.

### 28 September — new missingness and conditioned-mixture goal authorized

Created an active goal for initialization-gap diagnostics, smaller missingness-aware
models, and mixed pretraining with an observation-only finish and explicit task
conditioning. The preceding campaign remains complete. The new
[wave plan](missingness-wave-2026-09-28.md) defines six controlled arms, one seed,
unchanged observation selection and a 100 allocated GPU-hour operational ceiling.
Long-rollout training and OM4-supervision ablations are deferred by user instruction.
Authenticated Torch master works; required credentials/data/scratch checks pass,
queue is empty, and RTX partition nodes currently report allocated. Home quota
query returned permission denied, so it is not evidence of available scratch quota.
No new jobs have been submitted yet. Implementation and qualification are next.

Implementation now includes normalized-zero input fill, valid-only surface copy,
structured persistent masking, a full-grid observed-only completion auxiliary,
small initializer selection, explicit identity-initialized task input adapters,
and exact-count mixed schedules with an observation-only tail. Legacy defaults
remain load-compatible. Monthly/annual/state evaluators read the new model/fill
contract rather than silently reconstructing the old wide initializer. Gap
interventions preserve known observations and distinguish initial-state-only from
history-wide perturbation. Scratch quota was checked with myquota: 4.18/5.00 TB
used, about 0.82 TB nominal remaining; the new wave is estimated below 160 GB.

At 18:43 UTC, implementation and submission helper are pushed; training producer
is `053e34947932fdb56c6baf2c717a23f2df3cc733`. All 54 targeted local tests pass,
including a regression test that finish-only checkpoint selection cannot return
an earlier mixed checkpoint. Code-layer build 18723915 completed. Qualification
DAG: legacy fitting/resume 18723921/18723930, masked 18723931/18723932, conditioned
18723933/18723934. Legacy fitting completed with loss 1.296 to 0.254 and gradients
reaching initializer, dynamics and forcing adapter; its resume probe is running.
The other variants are queued. Production is not submitted until every variant
passes actual fitting and resume checks. H200 comment routing, requeue and GPU
constraint were verified; RTX's scheduler estimate was the following day, so H200
is being used for qualification and frozen diagnostics.

CPU climatology job 18723916 failed before data processing because an unnecessary
module load failed despite Apptainer already being on PATH. Corrected launcher
18724258 completed in 27 seconds with unchanged scientific producer; both attempts
are retained. The training-only climatology uses 1975-01-03 through 2013-07-27;
its SHA256 is `30cc359dba69c859ab96109aeeca90912d43e89c2e43294df737e75cdb784ba5`.
Frozen scratch/sequential/mixed gap diagnostics are queued as
18724314/18724315/18724316 after successful climatology completion. No diagnostic
result is claimed yet. Actual allocated GPU time at this snapshot is 0.0964 hours,
including qualification; CPU failures consume no GPU-hours. Accounting uses
Slurm duplicate-attempt records, not requested time limits.

At 18:56 UTC all three fitting and training/resume variants have passed. Masked
and conditioned fitting records explicitly confirm completion-head gradients.
Resume serialization and restoration are exact; replay numerical differences are
bounded against repeated native GPU updates rather than claimed bitwise exact.
No production model has started. A separate 60-OM4/60-observation throughput probe
18724971 will measure warmed-data update rates before fixing production limits;
it is operational qualification, not a scientific comparison arm.

Frozen diagnostic attempt 18724314 failed before inference because torchrun
intercepted `--run` without the argument separator. Queued siblings 18724315 and
18724316 were canceled before running. Corrected attempts 18724936/18724937/
18724938 preserve scientific producer, checkpoint paths and output contracts;
original submission and failed-attempt records remain intact. The first two
corrected attempts are running. Evaluation-only producer
`9999caf5937a8d0983dc0b6b6883011dc6a1a450` adds the fixed validation completion
and polar-support audit (two targeted scoring tests pass). Build 18724617 completed;
GPU audit smoke 18724939 is queued. Training remains pinned to `053e34947` and its
completed qualification contracts; adding an evaluator does not invalidate them.

At 19:18 UTC all submitted new-wave jobs are terminal and the queue is empty.
Actual accumulated GPU use is **0.7333 hours**, including every allocated failed
attempt, qualification, diagnostic, and throughput run. All nine unchanged
frozen-model/origin forecasts reproduce their original exports exactly. The
[gap report](initialization-gap-diagnostics-2026-09-28.md) now includes verified
full-grid initial and forecast maps and all five interventions. Repairing initial
missing surfaces improves sequential/mixed day-365 SST RMSE by about 5%, but
annual artifacts persist; history-wide replacement worsens short-lead SST.
Completion GPU smoke passed and retains absent-support scores as null.

Throughput job 18724971 completed all 120 updates. Last-20-task mean times were
2.320 seconds/OM4 update and 4.140 seconds/observation update. The original 16k
counts project 94.2 training GPU-hours before overhead/evaluation/recovery, so
production is not yet submitted. A [concrete budget decision](throughput-budget-decision-2026-09-28.md)
proposes matched 12k-total counts within the existing 100-hour cap, or retaining
16k with a 120-hour cap. Automatic approval review rejected editing the current
launcher/plan's counts and job limits without explicit user approval; those files
remain unchanged. Approval is pending in the thread. This is a production blocker,
not a Torch failure or scientific qualification failure.

The user clarified that feasible GPU spend and sensible update counts should be
chosen within the wall-time budget; the 100-hour ceiling was only an assistant
planning assumption. The approval blocker is resolved. Retain all original
16k-total comparisons and 2k observation-only finish, increasing scratch allocations
to 20 hours and the other arms to 16 hours. The operational planning ceiling is
120 GPU-hours, with the three-day deadline controlling the work. No scientific
counts, batch sizes, losses, qualification contracts or training producer changed.
Fresh test-only scheduling estimates H200 at 28 September 15:27 ET versus RTX
at 29 September 13:15 ET, so production will use the qualified H200 hardware.

Production is now submitted with the original counts and qualified training
producer `053e34947932fdb56c6baf2c717a23f2df3cc733`. Launcher/report revision is
`64a4264da0c557e9b087e82aac3c0b054131dc2d`.

| Arm | Job | Requested limit |
| --- | --- | --- |
| Small legacy scratch | 18726195 | 20 hours |
| Small masked scratch | 18726197 | 20 hours |
| Small masked sequential | 18726199 | 16 hours |
| Small masked mixed-finish | 18726202 | 16 hours |
| Small conditioned mixed-finish | 18726203 | 16 hours |
| Small conditioned mixed | 18726205 | 16 hours |

All six were pending for priority at the post-submit check. H200 routing, GPU
constraint, requeue, preemption comments and exact time limits were verified for
every job. The helper checked all three successful fitting/resume contracts before
submission. No production result is claimed yet. Draft PR 892 now links the frozen
gap report first in its table of contents and distinguishes those completed
sensitivity results from the new training wave. The prior approval question is
resolved; no blocker alert was needed after the user's clarification.

Evaluation launcher `f800bb151` submitted 32 dependent jobs with evaluator producer
`9999caf5937a8d0983dc0b6b6883011dc6a1a450`. Every submission has an `afterok`
dependency on its own production arm; monthly/annual evaluators additionally
require TRAIN_COMPLETE and selected/fixed-checkpoint lineage. The matrix includes
selected monthly and annual reports, matched-total endpoint monthly reports,
scratch raw-8k monthly reports, and validation-only completion/state diagnostics
at early milestones, selected endpoints and around the observation-only finish.
The first and last evaluator jobs are 18727181 and 18727734; full immutable records
are under EVALUATION_DAG.json and each name-submission.json in the cluster root.
These jobs are submitted dependencies, not completed scientific results. The
local dry-run verified all 32 command/dependency mappings before submission.

At 19:37 UTC all six production jobs are RUNNING. Both scratch arms have emitted
real joint-training events (steps 5 and 8); the four OM4-containing arms are still
warming their verified data caches. No job has a completion or partial marker.
Actual accumulated allocated GPU time is 1.0000 hours at this snapshot. The
transient QOSMaxGRESPerUser pending reasons cleared without intervention; no
replacement, protocol change or user action was needed.

### Sequential startup recovery

By 19:48 UTC the other five arms were updating normally, but sequential
18726199 stalled in CPU input preparation on gh114. Its recent OM4 updates took
42–160 seconds versus about 2.3 seconds in the mixed arms. Read-only py-spy showed
the main thread in NumPy atmospheric normalization in Samples.load; GPU telemetry
was repeatedly 0%. The job recorded 625 seconds kernel CPU versus 107 seconds user
CPU, host memory-pressure avg60 about 14%, and no cgroup OOM/high-limit events.
This is evidence of a host-side performance problem, not a scientific failure or
proof of a specific kernel root cause. Scratch startup slowness settled naturally.

Archived the sequential logs, telemetry, manifest and checkpoints under
`masked-sequential/attempt-records/18726199-before-node-relocation`. The archived
snapshot had observed 13 updates; its latest durable checkpoint was step zero.
That exact checkpoint was copied to joint-last.pt for deterministic restart;
SHA256 `05548e668c9895f35e0ee6de42d385563a181d0c361621ad4b66f219d33f76b3`.
Torch rejected node exclusion and releasing an explicitly held requeue, so the
held job and its five dependent evaluations were canceled and replaced through
ordinary scheduling, without forcing a node. New training job **18729753** keeps
the same output directory, producer, model/optimizer/RNG checkpoint, seed, data,
update counts and runtime limit. No new qualification or scientific protocol was
needed. Evaluations **18729754–18729758** now depend on that replacement; actual
scheduler dependencies were verified. Initial DAG copies and all old submission
records are retained alongside RECOVERY_1.json. The failed release is explicitly
corrected there; it was not a successful relocation.

Accounting retains the original job's REQUEUED attempt (1,163 allocated seconds,
0.3231 GPU-hours) plus the zero-allocation canceled held entry. Diagnostic overlap
steps are not double-counted. The replacement was pending at its post-submit check;
recovery performance is not yet verified. The other five training jobs continue.

### First automatic preemption and early completion checks

At 20:06 UTC, the five non-sequential jobs were explicitly PREEMPTED and
requeued automatically with the same IDs and dependencies. This was scheduler
preemption, not an inferred utilization cancellation. Archived their logs and
selection records under `attempt-records/first-preemption`; all five best.pt hashes
match best.json. Durable joint-last checkpoints are present. At 20:18 UTC the
sequential replacement and both conditioned arms have RUNNING allocations on
gh119, still warming caches; the remaining three arms are pending. Restoration
and resumed update throughput still need verification. Actual allocated use at
this snapshot is 3.8903 GPU-hours including failed/requeued attempts.

Early validation-only completion jobs 18731002–18731004 completed on immutable
checkpoints without selecting models or looking at test data. Their completion
signatures/checkpoint hashes were checked and the compact bundle transfer hash
matched. [Early missingness diagnostics](early-missingness-diagnostics-2026-09-28.md)
contains maps and pooled errors: masked scratch at 100 observation updates has
1.259°C block-hidden SST RMSE versus 9.531°C for legacy zero fill and 0.821°C for
monthly observation climatology. This establishes early learned completion, not
natural-gap truth, forecast improvement, or compute-matched transfer benefit.

At 20:29 UTC all six allocations are RUNNING. Fresh events confirm restoration
at updates 251/301 for legacy/masked scratch, 501/401 for conditioned finish/mixed,
and step 1 for the sequential replacement's archived step-zero restart. Their
latest steps are 272/318/553/452/57 respectively; the masked mixed-finish arm is
still loading its cache. Sequential's recent median update time is 5.21 seconds,
far below its failed attempt's stalls, although slower than qualification. The
other co-located mixed arms are also around 5.3 seconds; do not yet extrapolate
this startup snapshot to full-run timing. Actual allocated GPU use is 4.6361 hours.
Next bring-up check should confirm masked mixed-finish restoration and settled
throughput; resume hourly monitoring when stable.

### 20:49 UTC — repeated preemption; regular RTX fallback qualification

All six arms have now emitted fresh post-restart training events. The sequential
replacement and both conditioned arms were explicitly PREEMPTED again after
31–33 minutes and returned to the queue. The other three remained running at the
check. Accounting totals 6.5439 allocated GPU-hours. No completion is claimed.

Test-only scheduling rejects direct regular `h200` requests; enabling normal plus
preemption routing is accepted on `h200_public`, with a 30 September 01:12 ET
estimate. Regular `rtx6000_lzanna` is accepted and estimates 29 September 01:12 ET.
These are scheduler estimates, not reservations. Queued **18733298**, a one-hour
RTX throughput/memory check using exactly the existing qualified training
producer, conditioned architecture, seed and 60 OM4 + 60 observation updates.
Its account, partition, constraint and requeue flag were verified. This is an
operational hardware qualification, not another scientific arm. Production stays
on its current jobs while the fallback is assessed; no duplicate production writer
was launched and no pending job is described as running.

The wave plan now explicitly treats 120 GPU-hours as a planning estimate rather
than a hard user cap, matching the user's clarification. The original scientific
counts and 1 October 18:00 UTC wall deadline remain unchanged. Next ordinary check
is around 21:49 UTC, including RTX qualification state, production restoration,
settled throughput, accumulated allocation and all completion/error markers.

### 21:49–21:55 UTC — RTX qualified; two slow arms migrated

RTX qualification 18733298 finished successfully on gr106 in 15m16s with the
original producer, 60 OM4 + 60 observation updates and a real TRAIN_COMPLETE
marker. The device is NVIDIA RTX PRO 6000 Blackwell Server Edition. Last-20 mean
update times are 2.008 seconds OM4 and 3.438 seconds observation; the full model,
optimizer and data caches fit. No configuration or scientific protocol changed.

At the hourly check, actual allocated use was 11.6703 GPU-hours. All six arms had
made progress, but repeated preemptions continued. Conditioned mixed-finish had
landed on gh114 and again needed roughly 58–76 seconds per OM4 update; masked
scratch was around 7–8 seconds per observation update. The other arms' successful
placements were faster. Move only these two slow arms to regular RTX capacity.

Canceled their original jobs and eleven still-pending dependent evaluations,
verified that both old writers had stopped, and archived the exact resume/model,
manifest, selection metadata, events and Slurm logs before submission. Replacement
**18735489** (masked scratch) and **18735533** (conditioned mixed-finish) retain
all original module arguments, qualified producer, optimizer/RNG state, counts,
seed, deadline and output paths. The original strict resume contract remains in
force; changing GPU family is not a claim of bitwise-identical arithmetic.

RECOVERY_2.json records resume SHA256s and the full old/new job mapping. Current
PRODUCTION_DAG.json and EVALUATION_DAG.json are updated, with pre-migration copies
preserved. All eleven new evaluator afterok dependencies were verified against
the appropriate replacement. Evaluators retain their original producer and H200
routing. Masked scratch is RUNNING on gr106 but restoration is not yet confirmed;
conditioned mixed-finish is PENDING. The four other production jobs were left
untouched. No duplicate production writer was launched.

At 22:05 UTC both RTX replacements have restored successfully: masked scratch
emitted steps 901 through 1,085, and conditioned mixed-finish 1,101 through 1,132.
Masked scratch's recent median update time is now 3.36 seconds, versus 7–8 on its
previous H200 placement. Conditioned mixed-finish is around 5.06 seconds during
initial warmup, versus the prior 58–76-second stalls. All six jobs are RUNNING
and producing finite training updates; total allocated use is 13.1331 GPU-hours.
No full training/evaluation completion is claimed. Return to the ordinary hourly
cadence, next check around 23:05 UTC, including settled RTX throughput and runtime
margin. The unchanged strict constructors accepted both resumes and qualifications.

### 23:05 UTC — third RTX recovery and later completion diagnostics

Actual allocated use reached 18.2592 GPU-hours. The two RTX jobs ran uninterrupted
through the hour, reaching scratch 2,135 observations and conditioned finish
2,268 OM4 + 463 observations. The other mixed-finish arm had made only about 160
net updates amid further preemptions and a gh114 placement, so it was migrated to
regular RTX as job **18743657**. Preserved/stopped the old writer before restart;
RECOVERY_3.json records its resume SHA and replaces its five evaluation jobs with
18743658/659/660/671/675. Scheduler routing and all five afterok dependencies were
verified. It remains PENDING at 23:18; no RTX restoration is claimed yet. The
three other H200 jobs remain unchanged. Source, optimizer/RNG, counts, selection
and deadline are unchanged for the replacement.

Completed additional validation-only diagnostics 18743672/18743674 at matched
250/500/1,000 observation updates. The compact transfer hash and audit signatures
match; native-cell map geometry is verified. Extended the existing early report:
masked scratch's block-hidden SST RMSE reaches 0.718°C versus climatology 0.821°C,
while hidden polar caps and SSH still favor climatology. Naturally missing SST
cells have cold overshoot (minimum −3.981°C in the November map); report this
explicitly rather than treating the removal of warm speckles as physical truth.
The plotting helper now accepts explicit checkpoint panels and audit prefixes;
its default early outputs were verified byte-for-byte unchanged.

The user requested Pushover notification on completion or genuine blockage.
Read the pushover-notify skill and checked that local credentials are present
without sending a notification or exposing credentials. Use that channel for
those outcomes; no blocker currently requires user action.

### 29 September 01:05 UTC — third RTX arm restored; training continues

Regular RTX replacement 18743657 is RUNNING on gr106 and has advanced beyond its
1,500-step resume checkpoint to 2,342 total updates (1,966 OM4 + 376 observations).
All six arms currently have running allocations. Masked scratch reached 4,210
observation updates; conditioned mixed-finish reached 4,334 OM4 + 1,351 observations.
Sequential has reached 5,136 OM4 updates and has not yet begun its observation
phase. H200 preemptions continue with automatic resume; legacy scratch's latest
training event preceded the latest allocation startup. No TRAIN_COMPLETE or
TRAIN_PARTIAL marker exists in any arm, and dependent final evaluations remain
unstarted. Actual accumulated allocated use is 28.2922 GPU-hours including retries
and diagnostic jobs. The RTX qualification and three production migrations retain
the original scientific update counts and pinned training/evaluation producers.

### 02:05 UTC — remaining production routed to regular RTX

The stable RTX jobs sustained about 1,000–1,500 updates/hour. Net H200 progress
fell to a few hundred updates/hour amid repeated preemption/cache startup. Actual
allocated use was 33.3486 GPU-hours. All scientific completion markers remain
absent. To reduce this scheduling overhead, stopped and archived the three
remaining H200 writers and submitted ordinary RTX replacements with identical
qualified arguments and optimizer/RNG checkpoints:

| Arm | Previous job | Current RTX job |
| --- | --- | --- |
| Small legacy scratch | 18726195 | 18758994 |
| Small masked sequential | 18729753 | 18759212 |
| Small conditioned mixed | 18726205 | 18759237 |

RECOVERY_4.json preserves checkpoints, hashes and job mappings. All three RTX
requests have verified account/partition/constraint/requeue settings and are
currently PENDING; no restoration is claimed. Existing RTX jobs 18735489,
18743657 and 18735533 continue. Counts, seed, producer, loss, selection, deadline
and output paths are unchanged. Hardware migration does not imply bitwise-identical
floating-point arithmetic.

A migration-script prefix match also captured the similarly named conditioned
mixed-finish evaluators, canceling five unstarted jobs. Corrected this before any
evaluation ran: RECOVERY_4_EVALUATION_CORRECTION.json replaces those zero-allocation
jobs with 18759295/297/299/302/303 depending on the proper existing training job
18735533. Audited **all 32** current evaluation dependencies using each command's
explicit `--run` path, rather than name prefixes; every scheduler dependency now
matches its actual arm. PRODUCTION_DAG.json and EVALUATION_DAG.json are authoritative.
Future recovery must use explicit run ownership and must not reuse a submission
record pointing to an already-canceled job.

### 03:05 UTC — all six production arms restored on RTX

All six current jobs are RUNNING and emitting finite updates on regular RTX.
The latest three replacements have advanced past their restored checkpoints:
legacy scratch 3,461 observations; sequential 6,399 OM4 / 0 observations;
conditioned mixed 4,015 OM4 / 1,341 observations. The other RTX arms reached
masked scratch 6,306 observations, masked mixed-finish 4,112/1,225, and conditioned
mixed-finish 5,897/2,536. No scientific completion marker is present yet. Actual
accumulated allocated use is 38.3081 GPU-hours. With settled RTX throughput, the
remaining update budgets appear to fit the existing cumulative internal runtime
limits; keep monitoring this estimate rather than reducing counts. All final
evaluations remain correctly dependent on the respective current training jobs.

### 09:38 UTC — first complete arm and verified held-out evaluations

Small conditioned mixed-finish completed exactly 8,000 OM4 + 8,000 observation
updates at 09:29 UTC. The final 2,000 updates were observation-only; its selected
checkpoint is within that finish, at 8,000 OM4 + 7,200 observations, validation
score 0.566018. Actual best.pt SHA matches best.json and TRAIN_COMPLETE. All five
associated evaluation jobs completed successfully, with expected signatures,
96 monthly test origins, three annual origins and nine diagnostic validation
origins. A collector verified training/evaluation lineage and copied a small
metrics bundle with matching transfer SHA; it explicitly labels the five other
arms incomplete rather than presenting a finished comparison.

First-arm held-out composite is 0.590619 selected / 0.600793 at the raw equal-budget
endpoint, normalized against test seasonal climatology using the frozen spectral
key set. Its same-initializer persistence control scores 0.576657, so learned
monthly evolution is not an across-the-board improvement over this control.
Three-case mean day-365 SST RMSE is 0.810858°C and ADT RMSE 0.109203 m; this is a
lead-specific annual diagnostic, not selection or an established benefit relative
to the five still-running current controls. Diagnostic validation reevaluation
is 0.565980 versus selected 0.566018 across GPU families. Actual allocated use at
this snapshot is 77.7517 GPU-hours, including retries and completed evaluations.
No final cross-model conclusion or goal completion is claimed.

### 10:46 UTC — five arms advancing; report arrays verified

All five remaining production jobs were RUNNING with fresh finite events:
legacy scratch 11,757 observations; masked scratch 14,362; masked sequential
8,000 OM4 + 7,178 observations; masked mixed-finish 8,000 + 7,005; conditioned
mixed 7,856 + 7,167. Actual accumulated allocation is 83.3822 GPU-hours. No partial
completion markers were present.

CPU export job 18790475 completed successfully using reporting producer
ffea4ccfd (extractor SHA 902575af692c4416570d3b478fdd50232db0fc2a80452518008ac86a126152f9).
The initial explicit `all` partition request was rejected before allocation;
`cpu_short` was accepted. Export only subsets existing arrays; it neither runs
models nor changes selection. Conditioned mixed-finish compact archive is
35,235,840 bytes, SHA c8d7ef325fdcb501eaf8fa9df8fb566639b50269422a6a51cced7326cc0c681f.
DTN rclone PID 1805000 copied the archive/manifest to
`nyu-osn:emulators/jr7309/outputs/2026-09-28-observation-missingness-report`;
local read-back matched the archive SHA and every member-array SHA.

For selected conditioned mixed-finish, naturally missing wet-cell SST spans
−2.529 to 31.442°C in November 2013 and −2.266 to 32.959°C in July 2014.
These are 3,598 missing cells across the globe, not an Arctic-only statistic.
There is spatial variation rather than a constant warm fill, but cold extremes
remain and no missing-region observational accuracy is inferred. Full comparison
awaits the remaining arms. A reporting summarizer has been exercised on this
completed arm; cross-arm and scratch-8k branches await their completed outputs.

### 11:42 UTC — sequential evaluation verified; three training arms complete

Sequential and conditioned mixed training have now finished their exact 8k/8k
budgets. Sequential's five evaluations are verified; two conditioned mixed monthly
evaluations were still pending completion at collection time. Mixed-finish was
at 15,976 total updates, masked scratch at 15,340, and legacy scratch at 12,760.
Accumulated allocation is 88.1664 GPU-hours, with no TRAIN_PARTIAL markers.
A reporting-only submission used `job_id` rather than the accounting collector's
expected `job`; added the equivalent `job` field, restoring monitoring without
changing jobs or losing allocations.

Verified metrics bundle `report-metrics-1142.json.gz` has SHA
7aaad30c8dfe5cf4570783ea944094e82e45edf48d827fc6930d501b9b0ce09f, matching local read-back.
Sequential held-out selected composite is 0.572607 (raw endpoint 0.571255), versus
0.590619 for conditioned mixed-finish. Its three-case day-365 SST RMSE is 1.088920°C
versus 0.810858°C for conditioned mixed-finish. Monthly and annual rankings differ;
these two arms also differ in both task ordering and conditioning, so this is not
an isolated conditioning effect. Completion block SST RMSE is 0.559°C sequential /
0.594°C conditioned mixed-finish versus 0.821°C climatology; hidden polar-cap SST
is 0.540 / 0.516°C versus 0.545°C climatology. These scores use genuinely observed
artificially hidden labels, not naturally missing cells.

CPU export 18791598 completed, producing lossless sequential and conditioned mixed
arrays. Report renderers were exercised on the available data, including exact
native-cell map checks, full-grid initialized/30-day/annual fields, temperature,
salinity and velocity profiles, and regional spectra. Final six-model figures
and conclusions await the outstanding runs/evaluations.

### 12:41 UTC — five arms trained; evaluation capacity is the remaining queue

Five production arms completed their exact budgets. Legacy scratch remains RUNNING
at 13,844/16,000 observations with fresh finite events. Masked scratch completed at
12:18 UTC, selecting 10,800 observations (validation 0.575680). Masked mixed-finish
completed at 11:43 UTC, selecting 8,000 OM4 + 7,900 observations (0.570041). Actual
allocated use is 89.9739 GPU-hours. No partial training markers are present.

Eight eligible evaluations are PENDING for scheduler priority on H200; the six
legacy evaluations still depend on its live training job. Completed identical
protocol evaluations took 1–6 minutes. Attempted to shorten pending requests to
20 minutes for backfill, but Slurm rejected in-place time-limit updates with
`Unspecified error`; requests remain unchanged. A regular-RTX test-only request
estimated a later start (17:04 cluster-local), so no jobs were canceled or
replaced speculatively. EVALUATION_RUNTIME_REQUEST_REJECTED.json records this
scheduling check. This is a resource wait, not a scientific failure or a user-action
blocker. Continue hourly monitoring within the existing wall-time deadline.

### 13:41 UTC — five complete result sets collected

All 26 evaluations belonging to the five completed arms now have completion
markers. The collector verified their checkpoint hashes, exact budgets, expected
origins and selection/endpoint lineage. Legacy scratch remains RUNNING at
14,943/16,000 observations. Actual allocated use is 91.4525 GPU-hours.
`report-metrics-1341.json.gz` SHA is
bdb84b2e04158b16aac527d765c9f58ff589bf88d3b30464493801f0dff0d156,
matching the local copy. CPU export 18794818 completed; all five available compact
array archives and every member hash now verify locally.

Automatic approval review rejected an additional OSN upload because authorization
for that payload/destination was not established. No retry to OSN was made. The
safe alternative copied the compact report arrays from the DTN directly into the
user's local workspace, without an external-storage upload. This succeeded and
is not a user-action blocker.

Among these five, selected masked scratch has monthly test composite 0.568791,
sequential 0.572607, unconditioned mixed-finish 0.576713, conditioned mixed-finish
0.590619, and conditioned mixed 0.593176. Raw equal-total-budget endpoints instead
favor sequential (0.571255 versus scratch 0.598370). Annual and source-retention
comparisons differ again: conditioned mixed has three-case day-365 SST/ADT RMSE
0.748294°C / 0.107469 m versus scratch 0.887365°C / 0.159390 m. These results do
not establish a uniform pretraining benefit or a need for an observation-only tail.
The full comparison, including the old-fill control, remains pending.

The methods draft now distinguishes 6-step OM4 training from 6/7-step calendar-month
observation training, auxiliary reconstruction using contemporaneous histories
from the forecast path, and internal U/V diagnostics from geostrophic observation
metrics. Model parameter counts were independently reconstructed: 62,875,370
without task adapters and 62,966,546 with them. Qualification JSON was read back,
hash-verified, and included as a compact report artifact.

### 14:40 UTC — all training budgets completed

Legacy scratch completed at 14:37 UTC with exactly 16,000 observation updates,
selecting update 10,900 (validation 0.536989). All six training jobs are now
COMPLETED 0:0 with no partial markers. Allocated use at the monitor snapshot is
92.3983 GPU-hours. Its annual evaluator started; the five other legacy evaluators
remain queued for priority. All other arms' evaluations are complete.

Collector revision 257574bcf additionally checks actual hashes for every diagnostic
checkpoint, exact nine-origin lists, pinned producers, seed and fixed-budget
lineage counts. It passes the five finished result sets; the legacy annual output
is also verified while five remaining evaluations are explicitly marked missing.
`report-metrics-1443.json.gz` SHA is
40592fd1d9788b308afdd330882f8f34ee1fe85ed2c28bf73f583c1f5e4fb44b.
This strengthens the final audit without changing training, selection or evaluation.

### 15:42 UTC — all evaluations complete; final report assembly

All six training jobs and all 32 current evaluations are COMPLETED 0:0. The final
collector verifies every requested checkpoint against its actual file hash,
training/evaluation producers, fixed-budget counts, selection contract and exact
origin lists; all six result sets pass. No partial-budget outputs remain.
Final numerical bundle SHA:
`b64a6e1ab27a04ae2b804ac515b8821dc15b8ae2c6336dfdf8de1bbaedc32305`.
CPU export 18803869 completed; the last archive and every member hash matched the
direct DTN-to-local copy. Final allocated use is **92.726944 GPU-hours** across
142 accounting attempt records, including zero-allocation cancellations and CPU
jobs; no requested limits are counted as consumed time.

The final report includes all six selected and raw matched-budget scores, scratch
8k controls, persistence, component metrics, completion support/bias, source
retention and state interventions. Its 81 figures include 68 native-grid map
panels/collections, every requested checkpoint series, three annual cases and
full 19-level initializer profiles. Map geometry and pixel values are verified;
links, hashes and representative rendered figures were inspected. The old timer
is confirmed disabled and inactive. Report publication, PR contents and the
requested completion notification are the remaining delivery steps.


### 16:06 UTC — report published and completion notification accepted

Published the completed [six-model report](missingness-results-2026-09-29.md),
81 figures and numerical/provenance artifacts. The GitHub report blob matches the
local file (85ecad47a1103a22d9602a549204e083093e4322). Draft PR 892's description
and table of contents now lead with this completed wave; read-back matches the
submitted body. Repository hooks passed. Additional checks verified finite values
in 136 exported wet-cell state arrays and 450 full-state profile records.
All report links and artifact hashes pass; the automatic timer remains disabled.

The requested Pushover completion notice was accepted by the service. Automatic
review rejected a detailed notification; after verifying the active goal's
explicit Pushover instruction, a minimal report-ready notice containing no
experiment results, PR details or GPU accounting was accepted. No duplicate
notification or unauthorized OSN upload was sent. The full authorized experiment
and report scope is complete; no new wave or follow-up training was launched.


## 2026-09-29 — global observation-domain rerun

User authorized a full matched rerun of Small conditioned mixed, using the existing
run as control, and explicitly requested no latitude cutoff in inputs or scoring.
[Protocol and implementation](global-domain-rerun-2026-09-29.md) now remove the polar
cutoff from observation losses and global selection/evaluation, while preserving
model, schedule, normalization, seed and 8k/8k task budgets. Inputs already retain
all available latitudes. Geostrophic diagnostics retain their equatorial validity
exclusion; surface and thermohaline supervision do not exclude the equator.

Producer `79025a163817a577ab81af95b401f5cb0563cd12` passed 31 targeted tests and all
repository hooks. CPU build job 18811814 submitted; GPU DAG submission is gated by
verified overlay construction. Production is gated by fitting and resume probes
with the global-domain flag in both contracts. The primary comparison is the
matched final endpoints, rescoring the historical control globally; selected
checkpoint comparisons will explicitly distinguish their selection domains.
Torch authentication, credentials and data paths passed; scratch usage 4.31/5 TB.
The old timer remains off.

### 18:17 UTC — global rerun DAG queued

CPU overlay build 18811814 completed 0:0. Fit 18811858, resume probe 18811859,
and production 18811860 have verified after-success dependencies. Control global
endpoint monthly/annual evaluators are 18811861/18811862; original-selected control
is 18811863/18811864. New endpoint evaluators 18811865/18811866 and selected
evaluators 18811867/18811868 depend on production. All request one RTX6000 on
LZanna, producer 79025a163817a577ab81af95b401f5cb0563cd12. Fit and independent
control evaluators remain queued on QOSGrpCpuLimit; no GPU allocated, no training
or qualification success claimed, 0 GPU-hours consumed. Report and PR contents
now link the new experiment and clarify the historical input/loss/scoring domains.

### 19:13 UTC — matched ten-channel memory arm queued

User authorized a third arm adding ten initialized/autoregressed latent channels.
It matches the global treatment's 8k OM4/8k observation schedule and all physical
losses; no direct latent targets or penalties. Counts independently constructed:
62,966,546 physical-only versus 63,363,656 with memory (+0.63%). Tests verify
latent carry between steps, physical-loss gradient reach, land masking, physical
score extraction, OM4 loss compatibility and zero-latent checkpoint compatibility.
37 targeted tests and all hooks pass. The actual fitting probe additionally checks
latent initializer and processor gradients; production requires successful fitting
and mixed-task resume qualification under the new producer.

Producer b95179b963d372b72333c7f0e7e521a67b51ff6e, root
`/scratch/jr7309/runs/2026-09-29-observation-global-latent10`. Build 18815268 completed
0:0. Fit 18815275 depends on global reference producer 18811858; resume probe
18815277 and production 18815278 follow after success. Endpoint monthly/annual
18815279/18815280 and selected monthly/annual 18815281/18815282 depend on training.
All GPU jobs are pending: original global reference and control rescoring still
wait on QOSGrpCpuLimit, and the latent arm waits on those qualification dependencies.
No GPU allocated or training success claimed. Old jobs and pinned producers remain
unchanged. Annual exports preserve latent arrays for later diagnostics.

### 20:44 UTC — both global arms training; estimated overnight completion

Both RTX production jobs started at 19:41 UTC and are running: physical-only
18811860 has 1,515/16,000 updates (1,272 OM4 / 243 observations); latent10
18815278 has 1,544/16,000 (1,296 / 248). Both fitting markers verify gradient
reach and both mixed-task probes verify resume equivalence. The latent fitting
marker explicitly confirms initializer and evolution latent-head gradients.
All four historical-control global evaluation completion markers are present.

Measured median update times are 1.845/3.333 seconds for OM4/observations in the
physical arm and 1.809/3.271 seconds in latent10. Accounting for the later schedule's
higher observation fraction gives 10.4–10.6 hours of remaining update compute,
plus validation/checkpoints and final evaluation. Estimate results around
05:00–07:00 ET September 30, contingent on no interruption or evaluation queue
stalls. This is a throughput extrapolation, not a guaranteed scheduler completion.

## 2026-09-30

### 06:15 UTC — monitoring blocked by Torch connectivity

User requested continued monitoring and an updated report after completion. The
previous SSH ControlMaster socket is absent; one fresh BatchMode connection with
15-second timeout failed with `connect to host login.torch.hpc.nyu.edu port 22:
Connection timed out`. No live queue, accounting or new result files could be
read. Following torch-train guidance, did not retry authentication/connectivity.
This establishes loss of monitoring access, not job failure or completion; the
already-submitted training/evaluation dependencies may continue independently.
The September 29 20:44 UTC snapshot remains the last verified progress/ETA.

Sent the authorized blocker DM to Jesse's Slack account U086XUETAJ0; service
returned message link https://openathena.slack.com/archives/D086ET5CVQF/p1790748953259049.
Pending on restored access: check training and all evaluation markers, safely
recover routine failures, verify endpoint/selection lineage and global metric
support, collect three-arm results/maps and publish the updated comparison report.
The timer remains disabled; no replacement timer was created.

### 06:30 UTC — access restored; report assembly underway

Authenticated Torch access restored. Both jobs continued during the monitoring
outage: at 06:18 UTC physical-only was 14,518/16,000 and latent10 14,620/16,000;
subsequent structured events show continued finite-loss updates. No retries or
training changes were needed. Dependent evaluations remain queued for completion.

Control globally rescored monthly/annual results pass actual checkpoint hashes,
exact origins, global-domain flags and fixed 8k/8k lineage checks. CPU export
18859335 completed; the 18,872,320-byte control archive and every member SHA256
match the direct DTN-to-local stream. The documented DTN rclone module was absent,
so an authenticated read-only SSH stream was used; no new external bucket upload.
New-arm CPU exports 18859573/18859588 depend on their endpoint and selected annual
evaluations. Export sources are frozen under a hash-named directory with checksum
verification before execution. Report code is committed through 7229528ae.

Prepared the matched endpoint/selected tables, global component and persistence
comparison, full-grid native-pixel initial/day30/day365 maps, regional spectra,
and separate northern/southern polar SST/ADT errors. The latter independently
reproduce the existing control's global annual scores before splitting latitude
bands. Final conclusions await both new models' completed held-out evaluations.

### 07:45 UTC — global-domain comparison complete

Both new training runs reached exactly 16,000 updates (8k OM4 / 8k observations).
Physical-only 18811860 and memory 18815278, all twelve monthly/annual evaluations,
four qualification probes, two builds and three CPU exports are COMPLETED 0:0.
No retries were required. Final accounting is **24.618611 GPU-hours**, including
qualification and control rescoring; historical control training is not recharged.

The final collector verifies actual checkpoint/producer hashes, qualification and
resume contracts, explicit global scoring, exact cohorts and fixed-budget lineage.
Evidence SHA256: `13890f750ab84dbcb5b2233c806b931e2eea5c50be4757e989f2e97ad706d25c`.
All three compact archives and every member passed full SHA read-back. Native-grid
maps and independent regional/global reductions passed numerical checks.

[Final report](global-domain-results-2026-09-30.md): matched endpoint composites
0.8388 restricted control, 0.5927 global physical-only, 0.6064 global +10 memory.
Global supervision sharply reduces polar errors. Added memory has mixed effects,
not a consistent improvement; cold extremes and spatial ripples remain. Selected
scores and all components, initializer/day30/day365 maps, spectra, annual cases,
persistence controls and model definitions are included. Monitoring for this wave
is complete; the timer remains disabled. No further runs are submitted.

### September 30 — focused global physical-only day-30 report

User authorized a separate report with observed initializer references, saved
checkpoints across training, wider day-30 spectra against OM4/coarse/native
observations, and undetrended annual global means with baselines. No new training.
[Diagnostic plan](global-physical-focus-plan-2026-09-30.md) records the scope.

Native reference extraction on Grace 110827 completed 0:0. Twelve exact five-day
2015 day-30 windows from the original OISST/DUACS stores, their conservative
training-grid remaps, and preceding-December IAP context for the three annual
origins are local. The 139,335,680-byte archive and all members passed full SHA256
read-back; temporal/coarse equivalence is checked against the model targets.

Checkpoint stages have OM4/obs counts 324/50, 2,167/500, 4,949/2,000, 8,000/8,000.
The existing pinned evaluator producer is reused. Initial jobs 18883746–18883751
failed before inference because the submission omitted the torchrun argument
separator. Their logs/submissions are retained. Corrected jobs 18884170–18884175
all completed 0:0; the CPU collector 18884176 follows them. Include both attempts
in final allocated GPU-hour accounting. The final endpoint evaluations are reused.

The first CPU collector was canceled after repeated NPZ member decompression
was diagnosed. Source and partial arrays remain; replacement 18886425 loaded
members once and completed 0:0. All additional inference, including failed
launches, totals **0.245 allocated GPU-hours**; CPU work charged no GPUs.
Compact archive SHA256:
`40f0c8b09e9d22f5b84df753a3a0ab73d5b137c89635a8ef1b359dc20a9ef3a0`.
Native archive SHA256:
`df0759c84c55bf968ced0a69065fdfb68a5fd05495ea9e5bdb7d424059618c81`.
Both archives and every member passed full read-back. Native five-day coarse
references equal the actual model targets (identical finite masks; numerical
agreement within atol 2e-6 / rtol 1e-6).

[Separate physical-only report](global-physical-day30-2026-09-30.md) is complete:
64 figures, observed initializer references, four saved stages, wider SST/SSH/
geostrophic-KE day-30 spectra, and annual global means with persistence and
training climatology. Day-30 SST RMSE improves 1.486→0.640°C, but final SSH is
0.0792 m versus persistence 0.0800 m. SSH/velocity spectra shift strongly toward
observations; annual means expose persistent cold/low-SSH offsets and maps
retain artifacts. No new training or scientific selection changes. Regional
transform tests, reference equivalence and map pixel read-back passed.

A same-analysis OHC representation check on the preceding December references
finds model-layer integration offsets of about −0.21 GJ/m² (0–700 m) and
−0.059 GJ/m² (700–2000 m) before model error. The report distinguishes this
contribution from forecast evolution rather than attributing the entire annual
OHC offset to learned climatology or drift. Annual curves retain the native-layer
observation baseline; this diagnostic does not correct them or change selection.

### September 30 — matched global observation-only run

User authorized a fresh comparator and intermediate checkpoints, with fixed 8k
and 16k observation results added to the focused report. [Protocol and startup](global-scratch-2026-09-30.md).
One seed; same 62,966,546-parameter global physical-only IN architecture, task
adapters and losses; zero OM4 / 16k observation updates. Existing matching global
fitting/resume qualifications and producer 79025a163817a577ab81af95b401f5cb0563cd12
are reused; production starts from random weights and loads no qualification model.
No training implementation changed. Remote dry-run and repository checks passed.

Training **18890321** started immediately on gr102 RTX PRO 6000 Blackwell.
At 19:28 UTC it has 54 finite observation updates, zero OM4, and immutable
0/10/25/50 snapshots. Global observation validation at 50 updates is 1.9116.
This verifies execution, not final scientific performance. W&B is online; source,
global flags, fresh-source manifest and telemetry were checked. Four endpoint
evaluations depend on training succeeding: 8k monthly/annual **18890323/18890324**;
16k monthly/annual **18890360/18890437**. Runtime also enforces completion/counts.

Personal scratch usage is 4.36/5 TB, about 640 GB headroom; reserve roughly 30 GB
for the new outputs. One-GPU caps are 24 training hours plus six evaluation hours;
actual elapsed allocation, including retries, will be charged. Initial accounting
at 19:28 UTC is 368 allocated GPU-seconds (0.1022 GPU-hours), all training.
Cold-cache initial updates were slower than the completed mixed arm; revisit the
15–17-hour estimate after warm-up. Monitor with interruptible waits, no timer.
