<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Current Samudra stochastic-depth comparison

Experiment/results record; not intended to merge.

Combines main `0b5320248`, PR #914 `ae4122eb1`, and PR #896 `de3c05ecd`.
Matched seed-15 runs differ only in constant ConvNeXt stochastic depth (0 or 0.1), names, and output paths.
Uses the current `samudra_om4` preset: instance normalization, bfloat16, dynamic MSE capped at weight ratio 20, 70 epochs, learning rate 0.0006, four autoregressive calls, two output steps, and 360-day rollout validation each epoch for checkpoint selection.

Both use the staged 1-degree OM4 dataset at `/mnt/home/jrusak/data/om4/v2026-09/om4_onedeg` (4745 × 180 × 360), existing means/stds, Rust loading (8 concurrent reads per rank), and user-approved `tau_hfds` boundary variables because the native loader rejects derived heat-flux anomalies.

Each production beta job requests one full host: four GPUs, 144 CPUs, all schedulable memory, 24-hour limit. Batch 4 per GPU and gradient accumulation 2 give effective batch 32. Training logs go to W&B `ocean_emulators/default`.

Post-training evaluation selects the last periodic checkpoint (epoch 70) plus final EMA, with saved predictions and OM4 metrics over 2014-10-10 through 2022-12-24. Observation metrics were omitted in the initial evaluation and subsequently backfilled; see the corrected observational report below. Evaluation logs and predictions are retained under each training output's `evals` directory.

Pinned ARM64 image: `ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4`. Native extension copied from the successful beta job 210434; Rust source and Python reader are unchanged from that job's `fd5b9f7ad` revision.

Validation: 30 focused block/post-training-evaluation tests passed; both TrainConfig and EvalConfig pairs parse successfully. Both jobs completed successfully on 2026-10-07. See [results and full comparison](results/RESULTS.md). This single-seed pair showed higher fixed rollout and time-mean errors with stochastic depth 0.1 despite lower dynamically weighted validation loss.

[Observational results](observations/RESULTS.md) now include DUACS, OISST, IAP, and matched OM4 metrics for both final checkpoints, with the original wet masks restored for scoring.

[Observation figures and spectra](observations/viz/RESULTS.md) include the complete implemented visualization suite and exported spectral/time-series values.
