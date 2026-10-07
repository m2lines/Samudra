<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Results: current Samudra, stochastic depth 0 versus 0.1

Both 70-epoch runs and all four post-training checkpoint evaluations completed successfully on 2026-10-07. **This single-seed comparison does not support enabling stochastic depth 0.1 for this setup:** the dynamic validation loss is lower, but fixed 360-day normalized rollout RMSE is 4.19% higher and long-rollout normalized time-mean RMSE is 28.48% higher for the final raw checkpoint (26.88% higher for final EMA).

Positive percentage changes below mean higher error. The dynamic loss uses evolving per-channel weights and is not a fixed metric for comparing forecast quality across runs.

| Metric | Control | SD 0.1 | Change |
|---|---:|---:|---:|
| Epoch-70 dynamic validation loss | 0.70669 | 0.623189 | -11.82% |
| Epoch-70 EMA 360-day normalized rollout RMSE | 0.285537 | 0.297493 | +4.19% |
| Epoch-70 long-rollout normalized time-mean RMSE | 0.111534 | 0.143301 | +28.48% |
| Final EMA long-rollout normalized time-mean RMSE | 0.113139 | 0.143549 | +26.88% |

The 360-day metric is the spatial area-weighted normalized RMSE averaged equally over prognostic channels and target times; validation uses EMA weights. The evaluation time-mean metric is the spatial RMSE between generated and target time-mean maps, normalized per channel and averaged over the 77 prognostic channels. **It is not trajectory RMSE over the full eight-year period.** These two metrics measure different errors and should not be compared numerically against each other.

## Physical 360-day rollout errors at epoch 70 (EMA)

| Metric | Control | SD 0.1 | Change |
|---|---:|---:|---:|
| thetao (mean over 19 depths) | 0.263628 | 0.448941 | +70.29% |
| so (mean over 19 depths) | 0.0605947 | 0.102139 | +68.56% |
| uo (mean over 19 depths) | 0.0267959 | 0.0261617 | -2.37% |
| vo (mean over 19 depths) | 0.02551 | 0.0245138 | -3.91% |
| zos | 0.0365998 | 0.0596941 | +63.10% |

Temperature, salinity, and SSH worsen, while both velocity components improve modestly. The depth means above are arithmetic means of per-depth RMSE, not volume-weighted scores. Units follow the OM4 fields: temperature in °C, salinity in practical salinity units, velocity in m/s, SSH in m.

## Long-rollout time-mean errors: surface fields, raw epoch 70

| Metric | Control | SD 0.1 | Change |
|---|---:|---:|---:|
| thetao_0 | 0.329262 | 1.15082 | +249.51% |
| so_0 | 0.110552 | 0.252609 | +128.50% |
| uo_0 | 0.0177395 | 0.01932 | +8.91% |
| vo_0 | 0.0116952 | 0.012255 | +4.79% |
| zos | 0.0224242 | 0.0631262 | +181.51% |

Across all 77 prognostic channels, SD 0.1 improves time-mean-map RMSE in only 8/77 channels for raw epoch 70 and 13/77 for final EMA. These counts exclude derived ocean heat content. All 469 scalar evaluation metrics for each of the four checkpoints are finite. The complete scalar comparison, including all depths, is in [comparison.csv](comparison.csv); evaluation summaries and selected training scalars are alongside it. Copied summary paths are relative to each training output directory; metric values are unchanged.

## Execution and provenance

| Run | Beta job | W&B | Host | Slurm elapsed | Exit |
|---|---:|---|---|---|---|
| Control | 210664 | [uaei2vbr](https://wandb.ai/ocean_emulators/default/runs/uaei2vbr) | b1-31-s1-dgx-01-c12 | 3:40:32 | COMPLETED, 0:0 |
| SD 0.1 | 210665 | [km9xi8su](https://wandb.ai/ocean_emulators/default/runs/km9xi8su) | b4-18-s1-dgx-04-c08 | 3:50:22 | COMPLETED, 0:0 |

Source commit: `a20c44484` (main `0b5320248` + PR #914 `ae4122eb1` + PR #896 `de3c05ecd` + experiment configs). SHA256 of the staged source archive: `086fb156245f4ceb9d32f289fc58de9bdcc2364dbfcecf105d05b62394e5ff55`.

Same seed 15, same 1° OM4 data and normalization stores, Rust loader, approved `tau_hfds` boundary inputs, instance normalization, bfloat16, dynamic MSE limit 20, 70 epochs, learning rate 0.0006, 4 autoregressive calls and 2 output steps. Each used a full 4-GPU beta host, batch 4 per GPU, accumulation 2, effective batch 32. See the parent directory for exact configs and pinned image. These are fresh runs, not checkpoint resumes.

Runtime root: `/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006`. Under `control/output/current-1deg-control-ga2-210664` and `sd01/output/current-1deg-sd01-ga2-210665`, `evals/summary.json` records both `saved_nets/ckpt_70.pt` and `saved_nets/ema_ckpt.pt`. Each `evals/{epoch_0070,ema_latest}/predictions.zarr` contains 598 prediction times on a 180 × 360 grid, with 19 depths for thetao/so/uo/vo and surface zos. All 120 expected prognostic chunks per store exist (480/480 across all four). Control epoch-70 saved time coordinates were decoded as 2014-10-20 12:00 through 2022-12-24 12:00; the inference source starts 2014-10-10 and supplies the two initial input states.

Evaluation uses one continuous held-out trajectory, with no observational metrics. The configured last periodic checkpoint and final EMA were evaluated; best-validation checkpoints were not swept. No extra training or checkpoint selection was done after inspecting the comparison.

This is one matched seed on 1° OM4 with heat-flux anomalies omitted in both arms. It establishes the observed outcome for this configuration, not a universal conclusion about stochastic depth or the earlier 2° short-run experiments in #896. Different hosts also mean elapsed-time differences should not be treated as a controlled throughput benchmark.
