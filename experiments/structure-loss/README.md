# Current-preset OM4 structure-loss experiment

This is an experiment/results record, not intended to merge.

Base: Samudra main `f1e713220`. All three arms use the literal current
`samudra_om4/train.yaml` preset, seed 15, 1-degree v3 data, and the same
initialization and training schedule. Differences are confined to the loss:

- Local: MSE weighted by frozen local priorities bounded to [0.95, 1.10],
  with wet-cell mean one separately for every variable/depth.
- Spatial: MSE plus 100 times interior curvature-of-error MSE.
- Gradient: MSE plus 0.1 times horizontal forward-difference L1 error,
  following Cody's `llc_cpu_working` implementation at `2651acfff`.
  This uses MSE alone as the base, and omits the vertical term.

All objectives return per-output-channel values for the standard trainer.
Two output times repeat the same 77-channel mask/weight ordering.
MSE and curvature retain main's full-grid denominator. The gradient term
uses wet starting cells, includes wet-to-land differences, wraps longitude,
and excludes the final latitude difference. Local weights alter loss only,
not input/output normalization. `check_losses.py` verifies independent
loop-based references, exact-target zero loss, land gradients, repeated
output channels, and finite-difference gradients on CPU.

Local scales are regenerated from the training store's own means/stds and
32 sampled adjacent five-day pairs strictly within the training period.
They combine spatial variance and temporal change MSE using their maximum,
with a 0.02 normalized floor and approximately 300 km Gaussian smoothing.
The inverse standard-deviation weights are bounded to [0.5, 2] with a
per-channel mean-one solve, then shrunk 90% toward one. The archive records
selected dates, normalization, weights, masks, and checksum.

The current-model single-example qualification (Torch 20437457, all six
arms COMPLETED 0:0) showed relative state/high-pass MSE versus ordinary MSE:
local mild10 1.059/1.039; curvature100 0.374/0.0123; horizontal gradient
0.655/0.546. Local weighting is stable but has no demonstrated improvement.
These are memorization results, not forecast-skill claims.

## Execution

Set ROOT to a new personal-scratch run directory, DATA_ROOT to the selected
1-degree data root, and SIF to the verified runtime. Stage code under ROOT/code
and write a manifest.sha256 covering that snapshot. Submit prepare.sbatch;
it computes scales and validates six generated configs.

Pilot configs run one short training epoch with normal four-call
backpropagation, two output times, online W&B, and 360-day validation.
Full configs retain 70 epochs. All use checkpointing each epoch and
preemptible resume. Submit train.sbatch with MODE=pilot first. Full jobs
require an explicit PILOTS_VERIFIED file after checking finite training,
validation, online W&B, and checkpoints. No full jobs have been submitted.

2026-10-20: CPU preparation job 20439512 submitted to Torch; dependent
callback 20439565. Durable follow-up c25ded47-860e-4350-a65e-e39bac965379
checks preparation and continues the pilot/full-training sequence.

Existing recent MSE and dynamic-MSE controls differ in data version,
gradient accumulation, forcing configuration, and loader; treat them as
contextual controls, not strictly matched loss-only comparisons.

Preparation 20439512 completed 0:0 in 1m47s. All 77 channel weight means
and all sampled training-date bounds verified. Pilot array 20441707
(local/spatial/gradient, one concurrent RTX6000) submitted with preemptible
routing and requeue enabled. Dependent callback 20441724 and durable
30-minute reminder 388bf327-13c4-4598-a543-751235619040 cover bring-up and
completion. Preparation callback's outbound HTTPS failed; the local
reminder recovered status. Full training remains gated on pilot results.

All three pilots completed 0:0 (7m52s, 7m04s, 8m50s), with finite training
and validation losses, completed 360-day validation, four checkpoint files,
and online W&B sync. Pilot run IDs: local `7qnfv530`, spatial `79ptq2e2`,
gradient `3lwtwa3u`. Six updates only: do not interpret these as forecast
skill comparisons. A separate CPU job 20443913 reloads each latest
checkpoint and checks every model/optimizer tensor is finite.

Full 70-epoch array **20443921** is submitted with `afterok:20443913` and
the PILOTS_VERIFIED marker guard. It allows three concurrent RTX6000 jobs,
one per loss, from fresh matched seed-15 initialization (not pilot resume).
Checkpoint each epoch; 12-hour allocations with preemptible automatic
requeue/resume. Submission verified in scheduler; at this update all three
are waiting for checkpoint verification, not yet training. Callback 20443926
and 45-minute reminder `d28989d0-b669-4bd7-a497-5f8fecbae5b1` cover startup
and completion. Watch sustained loader throughput/GPU utilization and
scratch quota before extending runs; initial pilot batches are not a
reliable full-dataset throughput estimate.

At the first full-run check, verification job 20443913 completed 0:0 and
all 370 saved tensors per pilot were finite. All three full tasks are
RUNNING after ~42 minutes, epoch 1 at batches 502/543/543 of 705, with
finite losses. Online W&B IDs: local `0l15g248`, spatial `cpkbm74n`,
gradient `dc598v1x`. No full-run checkpoint yet at this observation.

The last 200 batches average 4.49/3.89/3.89 seconds per iteration, with
82.6%/79.0%/79.3% spent waiting for data. These are log-derived wait
fractions, not GPU-utilization telemetry. Investigate after first-epoch
checkpoints before accepting this throughput for the full campaign.
The pinned Rust backend explicitly rejects derived anomaly forcing;
switching loaders directly would require changing the specified forcing
or implementing/parity-testing support, so do not silently substitute it.
Next durable check is in 25 minutes, to capture first-epoch validation and
checkpoints and address sustained loader overhead.

All full runs saved epoch-1 checkpoints and reached epoch 2. First fixed
360-day checkpoint scores were local 0.350, spatial 0.429, gradient 0.346;
these early values do not establish final ranking. Input waits persist.
CPU-only job 20449205 benchmarks four/eight workers with concurrent reads
on/off, checking bitwise equality of every returned input/forcing/target
on the same 16 windows. No training settings changed. Callback 20449222,
20-minute reminder 7d8628ad-3ec1-429c-a213-ed078be3c3be; full-run completion
monitor remains active with a two-hour fallback.

Loader job 20449205 completed 0:0, peak RSS 11.1 GB. All four variants
returned bitwise-identical tensors. Steady seconds/sample: 4 workers with
concurrent reads 1.373; 4 without 1.601; 8 without 0.839; 8 with 0.831.
Eight workers thus provide ~1.65x benchmark throughput; this is not yet a
measured end-to-end training gain. Full runs remain on four workers and
are progressing in epoch 2.

Expanded CPU benchmark 20451207 compares 8/16 concurrent workers on 32
training windows, with identical-tensor checks, 16 CPUs and 64 GB RAM.
128 GB requests were rejected by CPU submission policy; the earlier
11.1 GB measured usage supports a 64 GB allocation. Initial expanded job
20451197 was cancelled before execution to correct window indices to stay
within the training split. Callback 20451211 and 20-minute reminder
91db9858-3ace-4ec8-a8f2-c6c5c481351e track the replacement. Choose restart
settings from this evidence, retain scientific configuration, and resume
from saved epoch checkpoints rather than restarting initialization.

Expanded benchmark completed 0:0 in 2m13s with bitwise-identical tensors.
Eight versus sixteen workers: 0.783 versus 0.761 steady seconds/sample,
only ~3% benefit from doubling further. Select eight concurrent workers.
Replacement harness requests 16 CPUs / 128 GB per RTX6000, preserves all
scientific settings and resumes latest model/optimizer checkpoints with
the original run names/W&B IDs. CLI override and scheduler test passed.
Wait for the final epoch-2 checkpoint before interrupting the old array.
Original per-run configs are archived remotely before the trainer can
write its updated configs. Raw CPU benchmark timings are in results/.

After all three epoch-2 checkpoints were fully saved and the last arm
started epoch 3, cancelled original array 20443921 deliberately and
submitted replacement **20453279** with afterany dependency to prevent
concurrent writers. Latest checkpoint remains epoch 2 for all arms;
unsaved epoch-3 work in the faster arms is intentionally replayed.
Only loader worker count changes (4 to 8); resource request becomes
16 CPUs / 128 GB per GPU. Two tasks initially RUNNING, gradient task
queued for QOSMaxGRESPerUser. Original tasks confirmed cancelled by us,
not by an inferred utilization-policy action.

Completion callback **20453294**, 30-minute reminder
**61b9f274-aca8-48dd-8720-5a23421ba2a5** replaces the original full-run
monitor. Verify checkpoint/optimizer restoration, original W&B IDs, and
measured throughput after startup. All earlier stage registrations retired.

Observed local/spatial resume: checkpoint loaded, Start Epoch 3, optimizer
LR 0.0005987923, original W&B IDs resumed online, saved configs confirm
8 workers, and finite epoch-3 per-batch logs. Gradient remains queued at
last scheduler observation. Initial prefetch batches do not establish a
sustained speedup; assess after warmup at the next callback. W&B rejects
replayed step numbers already logged before interruption (local through
1413, spatial through 1585); stdout retains replay metrics. Checkpoint
and final validation evidence remain the comparison source.

At the 30-minute resumed-run check, all three tasks RUNNING in epoch 4;
epoch-3 checkpoint files exist (1,344,736,251 bytes each). Saved config
diffs confirm the only change was loading.num_workers 4 -> 8. Over the
last 200 logged batches, mean iteration seconds local/spatial/gradient
were 1.972/1.968/1.823, versus the earlier 4.494/3.887/3.895; observed
throughput roughly doubled, though these windows are not a controlled
same-node benchmark. Data-wait fractions remain 61.7%/60.2%/57.3%.
Epoch-3 fixed validation scores improved to 0.294/0.364/0.293 respectively.
No final loss ranking is inferred from these early results. Settings remain
fixed; durable follow-up rearmed for one hour, completion callback retained.

At 17:34 UTC all three resumed jobs were found CANCELLED by UID 0 around
17:32, after 2h27-2h29. This was not our cancellation or a model exception.
Accounting provides no administrator explanation or GPU-utilization
telemetry; low utilization is plausible given the measured loader waits,
but not confirmed. All three epoch-8 checkpoints are intact; interrupted
epoch-9 work can be replayed. Best logged scores: local 0.280, spatial
0.337, gradient 0.273. Scratch usage 4.45/5 TB. Do not repeatedly submit
identical inefficient allocations.

CPU job 20462512 tests reusing duplicate frame reads within each four-call
sample, with copies to prevent tensor aliasing. It compares baseline and
candidate twice on the same 32 windows and requires bitwise input/forcing/
target equality. No production code or scientific settings changed yet.
Callback 20462544; 20-minute reminder a019c638-389b-4370-82eb-71da5f58955c
replaces cancelled-array monitoring. Next: inspect benchmark, validate a
material I/O fix, then resume from epoch 8 with appropriate compute.

Dedup benchmark 20462512 completed 0:0 in 6m39; all returned tensors
match bitwise. Baseline steady seconds/sample 1.194/1.205 versus dedup
2.167/1.952: dedup is 1.62-1.81x slower and will NOT be deployed. Quiet
monitor handed back results; completed registration a019c638 retired.
No full-training restart has been submitted. Epoch-8 checkpoints remain
intact on Torch. Asked user to refresh Engaging login before checking an
alternative host; previous failed authentication must not be retried until
refresh is confirmed under engaging-train policy. No results ETA until
compute access/capacity and a qualified resume are established.

Engaging login refreshed successfully. Personal pool has ~487 GB free;
scratch only ~204 GB, so use pool for this campaign's checkpoint outputs.
Matching container is already available in personal scratch. Existing
shared 1-degree data audit matches all 312 selected raw chunks, masks,
coordinates and metadata files byte-for-byte; this is sampled data
verification, not a full-store checksum. Shared means/stds differ, so the
exact original Torch statistics were copied directly into the personal
Engaging run's normalization directory; all 600 extracted files match
source SHA256. Resume configs point explicitly to these statistics.

Prepared engaging.sbatch and prepare_engaging.py retain loss/model/data
and eight loader workers, restore epoch eight, and first complete epoch
nine for qualification with scheduler target70. Full phase remains gated
on ENGAGING_QUALIFIED. Test-only Slurm submission accepted RTX PRO6000,
16 CPUs,128GB,mit_preemptable. No Engaging GPU job has been submitted.

Checkpoint transfer is awaiting user response: automatic approval review
rejected intermediate OSN staging of learned checkpoints without explicit
approval for that destination. No upload/export began. Asked permission
for the concrete user-owned OSN staging prefix, with direct Engaging
transfer as an alternative. Code/runtime preparation and normalization
verification are complete; no ETA or monitoring registration while this
transfer approval remains outstanding.
