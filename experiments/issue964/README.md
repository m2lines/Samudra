<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Issue 964: matched 1-degree loss comparison

This experiment/results record is not intended to merge. Results belong on
https://github.com/m2lines/Samudra/issues/964.

The reference is the completed October 24 control,
https://wandb.ai/ocean_emulators/default/runs/6j1z2mwc. Its source overlay is
`4462fa1f4179193820ba1e867cded56de9076ac7`; W&B's generic git field identifies
its container commit instead. This branch starts at that source and applies
PR #965's loss change. Model, training, stepping, and loader code are unchanged.

Three new arms use `legacy_mse`, `mse` (wet-cell denominator), and
`{type: dynamic, metric: legacy_mse, limit: 20}`. All use seed 15, 70 epochs,
batch 4 per GPU, four GPUs, accumulation 2 (global effective batch 32),
LR 0.0006 with cosine scheduling, InstanceNorm, BF16, and two input/two output
states with four autoregressive calls. Rust uses eight reads, two prefetched
batches, and device prefetch. Forcing is `tau_hfds`, without anomalies.

Data is the unfiltered September 2026 `om4_onedeg` release, locally under
`/scratch/jr7309/data/om4-v2026-09/om4_onedeg`. Training ends September 30, 2013;
validation starts October 5, 2013. The original control configuration is saved
in `oct24-control-config.yaml`. Qualification asserts equality after excluding
loss, operational paths/W&B metadata, and preemption recovery. The three new
arms differ only in loss and arm-specific output/configuration names.

Checkpoint cadence remains five epochs; validation uses the control's fixed
360-day normalized area-weighted RMSE every epoch, with EMA and the same
checkpoint selection. Post-training evaluation retains the control's
2014-10-10 through 2022-12-24 period, 25 calls per chunk, final epoch and EMA
checkpoints. Compare fixed metrics, physical-unit per-variable/depth RMSE,
and final versus validation-selected checkpoints; training-loss magnitudes
are not comparable across these objectives. Lower RMSE is better. This is a
single-seed comparison against OM4 model data.

Torch changes the hardware/runtime relative to the original four-GB200 run.
The three fresh arms use the same source/container and resource requests.
Preemption recovery is enabled and resubmissions must retain checkpoints.
The original January official v2 run `3o112ot4` remains historical context;
it differs in normalization layer, precision, forcing, data release, and
validation/checkpoint selection and is not the matched control.

Run IDs, qualification evidence, and results will be added after execution.
