<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# CM4 1-degree Samudra training

Experiment/results record, not intended to merge. Train the current standard
Samudra ConvNeXt U-Net from scratch on the newly processed CM4 control run.

- 70 epochs, four Beta GPUs, maximum 24 hours (96 GPU-hours for training).
- Standard widths `[280, 380, 480, 520]`, instance normalization, bfloat16,
  seed 15, dynamic MSE capped at a 20-fold weight ratio, cosine LR from 0.0006.
- Batch size 4 per GPU, no gradient accumulation (effective batch 16).
- Two input and two output states; four autoregressive calls per training window.
- All 19 levels of temperature, salinity, and both horizontal velocities plus SSH
  (77 prognostic channels); wind stresses, heat flux, and `wfo` as forcing.
- Rust loading: eight read threads per rank, two prefetched batches, GPU prefetch.
- Full-record means/stds, as explicitly selected for this CM4 dataset.
- Annual validation rollout each epoch; select checkpoints by fixed normalized
  rollout RMSE using the current EMA-validation defaults. Save regular checkpoints
  every five epochs and retain latest, best-validation, and EMA checkpoints.
- W&B online in `ocean_emulators/default`.

Disjoint interval-end timestamps on CM4's no-leap calendar:

| Split | Start | End, inclusive | Five-day records | Years |
| --- | --- | --- | ---: | ---: |
| Train | 0151-01-06 | 0231-01-01 | 5,840 | 80 |
| Validation | 0231-01-06 | 0251-01-01 | 1,460 | 20 |
| Held-out long-term testing | 0251-01-06 | 0351-01-01 | 7,300 | 100 |

Each split constructs its own windows, so no window crosses split boundaries.
The 100-year test period is reserved for later evaluation; no post-training
test sweep is attached to this training job. Qualification inspects test-period
timestamps only; it does not read physical test fields or evaluate forecasts there.

The dataset is `s3://m2lines-pubs/Samudra/v2026-09/cm4_onedeg/`, copied to
`/projects/ny/lz1955/multiscale/jrusak/data/cm4_onedeg`. It retains `wfo`, uses
corrected geographic velocity/stress rotation, no spatial filter, and one-time,
all-depth chunks compressed with Zstd and byte shuffle. The data-store transfer
record reports 131,466 objects and 140,827,879,002 bytes, all compared by full
download read-back on 2026-10-16.

`qualify.py` checks real CM4 values against the Xarray reader at both ends of
the train/validation splits, no-leap dates, a GPU optimizer step, ordinary
validation, and an annual rollout. Qualification uses one `beta_test` GPU for at
most 30 minutes. Full training is submitted only after qualification passes. The launcher
uses an archived source overlay on a pinned, previously verified ARM64 container;
the native extension is copied from a verified Beta run with matching Rust source.
The run directory retains the exact source archive/hash, extension hash, expanded
config, qualification report, allocation, logs, checkpoints, and completion hook.

Run root: `/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017`.

Training job **214219 completed all 70 epochs** in 5h21m31s on four GPUs
(about 21.43 GPU-hours). [W&B](https://wandb.ai/ocean_emulators/default/runs/4yoy3b6s).
The best configured 360-day validation rollout score was **0.294515 at epoch 40**;
the final score was 0.312688. See the [completion report](results/COMPLETION.md)
for the curve, checkpoints, and a diagnosed ordinary-validation logging defect.
The [100-year held-out evaluation](results/EVAL-100YR.md) is submitted as
**Beta job 214347**, using the selected epoch-40 EMA state. Results are pending.

Qualification job **214211** completed successfully.
See [the launch record](results/LAUNCH.md) for verified checks and provenance.
