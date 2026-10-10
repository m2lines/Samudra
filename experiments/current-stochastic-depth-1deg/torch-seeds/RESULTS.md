<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Three paired seeds: current Samudra with stochastic depth 0 versus 0.1

**The two additional seeds reproduce the original result: keep stochastic depth at 0 for this configuration.** Across seeds 15, 16 and 17, SD 0.1 increases all five primary observation RMSEs for both raw epoch 70 and final EMA: **30/30 paired comparisons worsen**. SST RMSE increases by 54–64%. Fixed 360-day rollout error and long-rollout time-mean error also worsen in every seed, despite lower dynamic validation loss.

All four new Torch runs completed 70 epochs, their eight checkpoint evaluations, and separate CPU observation scoring on 2026-10-10. No new visualization was run. These are three matched pairs, with a hardware/backend difference between the original Beta pair and the two new Torch pairs; they are not six independent replications of each metric or evidence about every stochastic-depth configuration.

## Observation metrics

Entries are percentage increases in RMSE, `100 × (SD/control − 1)`; positive means worse. “Raw” is epoch 70; “EMA” is the final EMA checkpoint.

| Observation RMSE | Seed 15 raw | Seed 15 EMA | Seed 16 raw | Seed 16 EMA | Seed 17 raw | Seed 17 EMA |
|---|---:|---:|---:|---:|---:|---:|
| Surface geostrophic velocity | +0.31% | +0.37% | +0.13% | +0.18% | +0.12% | +0.16% |
| Instantaneous surface EKE | +0.16% | +0.14% | +0.17% | +0.19% | +0.12% | +0.15% |
| SST | +58.13% | +63.57% | +61.95% | +62.11% | +54.48% | +57.15% |
| OHC per area, 0–700 m | +2.09% | +2.70% | +4.71% | +5.49% | +3.74% | +3.52% |
| OHC per area, 700–2000 m | +1.17% | +1.33% | +3.21% | +2.90% | +3.21% | +3.46% |

The velocity/EKE increases are small in practical magnitude; the SST degradation is much larger. Absolute SST RMSE:

| Seed / platform | Raw control → SD (°C) | EMA control → SD (°C) |
|---|---:|---:|
| 15 / Beta | 0.8922 → 1.4109 | 0.8875 → 1.4517 |
| 16 / Torch | 0.8793 → 1.4241 | 0.8835 → 1.4322 |
| 17 / Torch | 0.9086 → 1.4035 | 0.9153 → 1.4383 |

[All absolute primary scores and paired changes](results/observation-paired-comparison.csv) and [all model observation rows across seeds](results/observations-all-seeds.csv) include annual scores, temporal uncertainty, residual-variance diagnostics, and coverage/geometry diagnostics. The combined table has 612 rows (12 model/checkpoint combinations × 51); the unchanged OM4 baseline remains in each run's individual table and the [original report](../observations/RESULTS.md).

Scoring uses the same published DUACS, OISST and ARGO-IAP products and metric code as seed 15. Velocity, EKE and SST use complete years 2015–2022; OHC uses 2015–2021 because the last month/year is incomplete after temporal alignment. CSV confidence intervals use calendar-year block bootstrap, **not uncertainty across training seeds**. We do not infer significance of the small velocity/EKE changes from their consistent signs alone.

## Fixed OM4 metrics versus dynamic loss

| Metric, SD change | Seed 15 | Seed 16 | Seed 17 |
|---|---:|---:|---:|
| Epoch-70 dynamic validation loss | −11.82% | −15.17% | −15.52% |
| Epoch-70 EMA 360-day normalized rollout RMSE | +4.19% | +2.52% | +3.10% |
| Raw long-rollout normalized time-mean RMSE | +28.48% | +32.95% | +23.18% |
| EMA long-rollout normalized time-mean RMSE | +26.88% | +32.10% | +25.07% |

The dynamic loss has evolving per-channel weights and is not a fixed quality metric across runs. The 360-day score averages spatially weighted normalized errors over prognostic channels and target times. The long-rollout score compares generated and target **time-mean maps**, normalized per channel; it is not eight-year trajectory RMSE. See [all paired evaluation scalars](results/eval-paired-comparison.csv), the per-run `eval-summary.json` files and `training-scalars.json` files under [results](results).

## Execution and reproducibility

| New run | Training job | W&B | Training + evaluation elapsed | CPU observation job / elapsed |
|---|---:|---|---|---|
| Control seed 16 | 19521808 | [rzp661ov](https://wandb.ai/ocean_emulators/default/runs/rzp661ov) | 3h38m13s | 19527566 / 28m45s |
| SD 0.1 seed 16 | 19521809 | [pndzc5mx](https://wandb.ai/ocean_emulators/default/runs/pndzc5mx) | 3h46m39s | 19530759 / 31m58s |
| Control seed 17 | 19521810 | [dkya8d2t](https://wandb.ai/ocean_emulators/default/runs/dkya8d2t) | 3h42m10s | 19536765 / 32m05s |
| SD 0.1 seed 17 | 19521811 | [hlznlfbf](https://wandb.ai/ocean_emulators/default/runs/hlznlfbf) | 3h46m09s | 19548461 / 26m13s |

All four training jobs exited `COMPLETED 0:0` on gr106. Each requested one full eight-GPU NVIDIA RTX PRO 6000 Blackwell host, 128 CPUs and 512 GiB RAM. Batch size 2 per rank × 8 ranks × gradient accumulation 2 gives effective batch 32. Rust loader: eight concurrent reads per rank, two prefetched batches, device prefetch enabled. Both arms use 1° September-2026 OM4, `tau_hfds`, bf16, instance normalization, 70 epochs, and the original merged PR #914/#896 model settings. Within each seed, saved configs differ only in run name, stochastic-depth rate and the corresponding eval-config path. Both arms have constant SD schedules (0 or 0.1).

**Torch backend workaround:** unmodified runs repeatedly failed with CUDA/cuDNN errors in 1×1 convolutions. Benchmarking changes did not fix this; captured operands passed isolated single-GPU tests, and disabling device prefetch only delayed failure. A narrow runtime wrapper routes 1×1 `Conv2d` through PyTorch's native backend with cuDNN disabled for that operation only. It preserves parameter shapes and convolution semantics and applies in spawned evaluation workers too. A 30-minute diagnostic passed epochs 2–10 and their validations before reaching its time limit during epoch 11. The four reported runs then **started fresh from epoch 1** with identical wrappers; no diagnostic checkpoint or metric is included in their results. `CUDA_LAUNCH_BLOCKING=1` was retained throughout. This is an empirically successful workaround, not proof of the underlying library defect.

The wrapper is [sd_pointwise_native.py](diagnostics/sd_pointwise_native.py). Its [exact executed bytes](diagnostics/sd_pointwise_native.executed.py.txt), before formatting and a type-checker annotation for this report, have SHA256 `e8bde8aaf5efa40bccb97dd722842af1e533ec894727b0f45298b3912052baaf` at execution; see [submission script](diagnostics/submit-native1x1.sh). The original model source overlay remains `c3df51b5e0a8e497201377b27423eaf91d712c3e`, SHA256 `bc963b707b76741c5827d88a41b44b27464b7d16c17b2ee324abac8ac6569fd0`. The x86 container digest is `sha256:babf82575aed4e7cf0166313424e0203b5ea16e09cd525a0bad2874b808d676d` (torch `2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2, cuDNN 92100).

Seed 15 used four Beta GPUs with batch 4 per rank and the original cuDNN implementation. Both platforms have effective batch 32, but hardware, rank count, per-rank batching, reduction order, and convolution backend differ. Compare SD against its matched control within each seed; do not attribute cross-platform absolute differences solely to seed, or interpret timings as a controlled throughput benchmark.

Runtime root: `/scratch/jr7309/runs/current-sd-seeds16-17`; outputs are `output/current-1deg-{control,sd01}-seed{16,17}-torch-native1x1`. The exact staged data metadata hash is `b28030e33e0da05dcacc1c91eb5527b53c9f3601a35e00001852983524485d8e`, matching Beta; full staging verification checked 385,213 files with zero differences. Runtime configs are preserved as JSON (safe conversion of YAML path/tuple tags), with hashes of their original YAML files beside them. Per-run source/container manifests and scalars are in `results/`.

Each of eight prediction stores contains 598 forecast times, 19 subsurface levels, the 180×360 grid, and all 120 expected prognostic chunks: **960/960 chunks verified**. All 469 saved evaluation scalar values per checkpoint and all observation `value` entries are finite. [Verification record](results/output-verification.json).

Observation scoring ran separately with **zero requested GPUs**, 32 CPUs, eight Dask threads and 384 GiB requested RAM. It took 26–32 minutes per raw/EMA pair, excluding queue time. [Scoring script](diagnostics/score_torch.py) reapplies the original per-channel wet masks lazily because the pinned original writer zero-filled dry cells; saved prediction stores are unchanged. The same existing OM4 baseline is reused. No observation metrics were retroactively logged to W&B, no unavailable metrics were treated as zero, and no new spectra or visualization are claimed for these seeds.

The new successful GPU runs consumed 119.09 allocated GPU-hours; failed attempts and diagnostics added 8.03, for 127.12 total. [Slurm accounting](results/accounting.psv) excludes the earlier Beta pair and CPU preparation/scoring from that GPU-hour total. [Download checksums](results/download-checksums.json) match all eight copied evaluation/observation files to their remote originals.
