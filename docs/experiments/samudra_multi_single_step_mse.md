<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# SamudraMulti 1-degree single-step MSE baseline

## Scope and gate

This experiment asks whether the current SamudraMulti representation can become a
useful 1-degree single-step baseline before adding more resolutions. Training and
evaluation use plain normalized MSE, one forecast step, the Rust loader, and no
`hfds_anomalies`. The physical patch extent remains fixed at 3 degrees by 5 degrees;
reducing patch size is outside this experiment.

The all-channel validation-MSE gate is:

- at most 0.05: promotion-ready;
- 0.05 to 0.075: diagnostic follow-up;
- above 0.075: do not add resolutions.

Long autoregressive rollouts are deliberately out of scope until single-step quality
is competitive.

## Reproducible runs

| Purpose | Config | Container | Slurm | W&B | Status |
|---|---|---|---:|---|---|
| Full-data SamudraMulti baseline | `configs/samudra_multi_om4/train_1deg_mse_proxy.yaml` | `26.05-9992bf52a3031442e2875a52fd113131c9162abd` | `14291479` | [ty6mwti9](https://wandb.ai/ocean_emulators/default/runs/ty6mwti9) | Completed |
| 512-timestamp SamudraMulti screen | `configs/samudra_multi_om4/train_1deg_mse_fast_proxy.yaml` | `26.05-3904ad07a55c5ea19d21bfe017e06d4b5bb8234f` | `14311686` | [j76loxwm](https://wandb.ai/ocean_emulators/default/runs/j76loxwm) | Completed |
| 512-timestamp v2 control | `configs/samudra_om4_v2_highres/train_1deg_mse_fast_proxy.yaml` | `26.05-3904ad07a55c5ea19d21bfe017e06d4b5bb8234f` | `14311687` | [b5pdwp91](https://wandb.ai/ocean_emulators/default/runs/b5pdwp91) | Completed |

The immutable config checksums are:

| Config | SHA-256 |
|---|---|
| Full-data SamudraMulti | `aee0841655c136d4b228f06c722b96e2596f18d0d84104d38b78c12fd8561742` |
| 512-timestamp SamudraMulti | `ac96148a78e543d92cfd3265a09168bda44f1ca2bb1ff9f9633dd00c2aa8c1db` |
| 512-timestamp v2 | `67949e5d286fed477c430f1c5a5e3b5d6b6af25229cf760872992c5a47af4de5` |

The full baseline config was introduced by commit `9992bf52`. The fast configs were
introduced by `9d5b278e`; pinned x86 image publication is recorded by
[GitHub Actions run 29691663638](https://github.com/m2lines/Samudra/actions/runs/29691663638).

The best finite validation epoch can be reproduced directly from W&B history with:

```bash
uv run python scripts/summarize_mse_runs.py \
  ocean_emulators/default/ty6mwti9 \
  ocean_emulators/default/j76loxwm \
  ocean_emulators/default/b5pdwp91
```

Pass multiple run paths to produce the final comparison table, or add
`--format=json` for machine-readable output. The script rejects dynamic-loss and
multi-step runs so their weighted dashboard values cannot be mistaken for the
plain-MSE comparison used here.

All comparisons use four RTX6000 GPUs and effective global batch 32. Slurm requests
16 CPUs and 128 GiB per four-GPU job, substantially below the generic proportional
Torch guidance. The dependency chain guarantees that these jobs run sequentially.

The first screen submission (`14295585`) failed before data loading because its
submit-time data root repeated the `om4_onedeg_v3` directory already present in the
config's source paths. Its dependent v2 job (`14295587`) was consequently cancelled.
The replacement chain above uses `/scratch/jr7309/data` as the root; all model,
training, image, and resource settings are otherwise unchanged.

## Screening dataset

Training configuration currently accepts a contiguous `TimeConfig` slice rather than
an arbitrary set of timestamps, so a season/decade-stratified index set is not
available without changing the data interface. The deterministic substitute is the
inclusive 2006-10-05 through 2013-10-05 slice. Torch's stored time coordinate confirms
that it contains exactly 512 five-day timestamps and covers every season seven times.
Validation remains the fixed 2013-10-05 through 2014-10-05 interval used by the full
baseline.

## Baseline evidence

The full run completed all 12 epochs with exit code zero. Training took 2:16:50;
the complete Slurm allocation took 2:26:32 including container setup. It trained on
353 rank-local microbatches per epoch. Representative epochs took about 11:10, or
1.90 seconds per microbatch, with logged data wait near 0.001 seconds.

Peak model-process usage was 26.5 GiB per GPU and 4.2 GiB CPU RSS per rank. Slurm
reported 87,701,424 KiB (83.6 GiB) MaxRSS for the batch step and 60,518,732 KiB
(57.7 GiB) for the Apptainer step. Thus the reduced request of 16 CPUs and 128 GiB
was sufficient without OOM, NCCL, or loader failure.

The full run's validation-selected weights are preserved at
`2026-07-19-multi-1deg-1step-mse-rust-gbs32-9992bf52/saved_nets/best_validation_ckpt.pt`
under `/scratch/jr7309/runs`. The file was atomically replaced when held-out
one-step validation improved and ultimately selected epoch 12.

| Artifact | SHA-256 |
|---|---|
| Resolved `config.yaml` | `00b42980cb5b328d15a22e49a453a8b3d4d8123a80d61b69c41eef5aed0374c1` |
| `best_validation_ckpt.pt` | `189cda72ce574f1db0421b26f6a5da1a934ecb85734cbb1fb97da17c4dbfefda` |
| `ckpt.pt` | `00a76668f95544c4a6a8d0d981c7a2a7e9fe0611a6e8103cfca58e096a3f334d` |
| `ckpt_5.pt` | `1bf957d839c5c66ddb724aa03b1f381dd72e8d2b9817d1f4d310ea421366ed48` |
| `ckpt_10.pt` | `12e8279cbbfa327aafef62f215e12ffbd152e6457f41a03601b9d677b3dd9005` |
| `ema_ckpt.pt` | `a7a217a02135845c5f4f83ac560549af9ad6c1af52f2221a20c9d8f36019583e` |

The final held-out one-step results from the validation-selected checkpoint are:

| Variable group | SamudraMulti | v2 reference | Ratio |
|---|---:|---:|---:|
| Temperature | 0.04279 | 0.00149 | 28.7x |
| Salinity | 0.08267 | 0.00138 | 59.9x |
| Zonal velocity | 0.50774 | 0.0361 | 14.1x |
| Meridional velocity | 0.55915 | 0.0565 | 9.9x |
| SSH | 0.03611 | 0.00239 | 15.1x |
| All channels | 0.29469 | 0.0236 | 12.5x |

These are unweighted normalized one-step MSEs. The all-channel value is above 0.075,
so this baseline fails the promotion gate and must not be extended to additional
resolutions.

## Final screen comparison and decision

Both replacement screen jobs completed with exit code zero. Their
validation-selected epoch-12 metrics are:

| Variable group | SamudraMulti screen | v2 screen | Ratio |
|---|---:|---:|---:|
| Temperature | 0.09835 | 0.00692 | 14.2x |
| Salinity | 0.35394 | 0.00700 | 50.6x |
| Zonal velocity | 0.53695 | 0.04658 | 11.5x |
| Meridional velocity | 0.56634 | 0.06476 | 8.7x |
| SSH | 0.08359 | 0.00805 | 10.4x |
| All channels | 0.38508 | 0.04287 | 9.0x |

The screen preserves the decisive model ranking. SamudraMulti is 12.5 times worse
than the supplied v2 reference on the full-data comparison and 9.0 times worse than
the matched v2 screen. The proxy should therefore be used to reject or rank large
differences, not to estimate a full-data metric: reducing the training slice changed
the absolute MSE for both models.

The SamudraMulti cost comparison is:

| Evidence | Full data | 512 timestamps | Saving |
|---|---:|---:|---:|
| Rank-local microbatches/epoch | 353 | 63 | 5.6x fewer |
| Steady-state epoch time | about 11:10 | about 2:00 | 5.6x faster |
| Training time | 2:16:50 | 0:27:27 | 5.0x faster |
| Slurm allocation | 2:26:32 | 0:28:19 | 5.2x faster |
| Four-GPU allocation | 9.77 GPU-hours | 1.89 GPU-hours | 80.7% less |
| Apptainer MaxRSS | 57.7 GiB | 33.0 GiB | 42.8% less |
| Per-GPU peak | 26.5 GiB | 26.5 GiB | unchanged |

The v2 control trained in 0:22:55 (0:23:30 allocation), used 23.1 GiB Apptainer
MaxRSS, and peaked at 3.1 GiB per GPU. Both screens used the requested four GPUs,
16 CPUs, 128 GiB, Rust loader, effective global batch 32, plain MSE, one step, and
no `hfds_anomalies`.

The screen artifacts are pinned by these additional checksums:

| Artifact | SHA-256 |
|---|---|
| SamudraMulti resolved `config.yaml` | `956e463f0876e870b5958b30bfd9e3565337dd85c03d5179d2d39c69c08f68cd` |
| SamudraMulti `best_validation_ckpt.pt` | `10cdb4f9981ad32accb795f15f631ff22d773b9dabe371626a4feedf6e01fc04` |
| v2 resolved `config.yaml` | `797052a4a6ce74dad7309b4d9fcea6e7edf3eaad965ed96029a936ffafc82719` |
| v2 `best_validation_ckpt.pt` | `d0db895d76e00de61466e0777b21a9ecfe1abf0f2248905a53c0e64a15a1ba5b` |

Do not add another resolution: the full-data baseline is above 0.075 and neither
comparison is within roughly two times v2. The next bounded experiment should use
the 512-timestamp SamudraMulti screen with only residual prediction enabled, keeping
the physical patch extent and all other controls fixed. Promote that candidate to a
full-data 1-degree run only if its proxy all-channel MSE is at most twice the matched
v2 screen (approximately 0.08575). If residual prediction remains far above that
threshold, discuss a representation change before spending more compute. No longer
autoregressive rollout is part of this decision.
