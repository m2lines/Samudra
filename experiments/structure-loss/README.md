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
