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
| Full-data SamudraMulti baseline | `configs/samudra_multi_om4/train_1deg_mse_proxy.yaml` | `26.05-9992bf52a3031442e2875a52fd113131c9162abd` | `14291479` | [ty6mwti9](https://wandb.ai/ocean_emulators/default/runs/ty6mwti9) | Running |
| 512-timestamp SamudraMulti screen | `configs/samudra_multi_om4/train_1deg_mse_fast_proxy.yaml` | `26.05-3904ad07a55c5ea19d21bfe017e06d4b5bb8234f` | `14295585` | Pending | Dependency on full baseline |
| 512-timestamp v2 control | `configs/samudra_om4_v2_highres/train_1deg_mse_fast_proxy.yaml` | `26.05-3904ad07a55c5ea19d21bfe017e06d4b5bb8234f` | `14295587` | Pending | Dependency on fast SamudraMulti |

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
  ocean_emulators/default/ty6mwti9
```

Pass multiple run paths to produce the final comparison table, or add
`--format=json` for machine-readable output. The script rejects dynamic-loss and
multi-step runs so their weighted dashboard values cannot be mistaken for the
plain-MSE comparison used here.

All comparisons use four RTX6000 GPUs and effective global batch 32. Slurm requests
16 CPUs and 128 GiB per four-GPU job, substantially below the generic proportional
Torch guidance. The dependency chain guarantees that these jobs run sequentially.

## Screening dataset

Training configuration currently accepts a contiguous `TimeConfig` slice rather than
an arbitrary set of timestamps, so a season/decade-stratified index set is not
available without changing the data interface. The deterministic substitute is the
inclusive 2006-10-05 through 2013-10-05 slice. Torch's stored time coordinate confirms
that it contains exactly 512 five-day timestamps and covers every season seven times.
Validation remains the fixed 2013-10-05 through 2014-10-05 interval used by the full
baseline.

## Baseline evidence

The full run trains on 353 rank-local microbatches per epoch. Epochs 2 and 3 took
11:09 and 11:10, respectively, or about 1.90 seconds per microbatch. Logged data wait
is approximately 0.001 seconds. Peak model-process usage observed so far is about
26.5 GiB per GPU and 4.1 GiB CPU RSS per rank; final Slurm MaxRSS will be recorded
after job completion.

The full run's validation-selected weights are preserved at
`2026-07-19-multi-1deg-1step-mse-rust-gbs32-9992bf52/saved_nets/best_validation_ckpt.pt`
under `/scratch/jr7309/runs`. The file is atomically replaced only when held-out
one-step validation improves; its final checksum and selected epoch will be recorded
after training finishes.

Best validation results through epoch 4 are:

| Variable group | SamudraMulti | v2 reference | Ratio |
|---|---:|---:|---:|
| Temperature | 0.0764 | 0.00149 | 51.3x |
| Salinity | 0.2521 | 0.00138 | 182.7x |
| Zonal velocity | 0.5274 | 0.0361 | 14.6x |
| Meridional velocity | 0.5636 | 0.0565 | 10.0x |
| SSH | 0.0647 | 0.00239 | 27.1x |
| All channels | 0.3513 | 0.0236 | 14.9x |

These are unweighted normalized one-step MSEs. They are interim until all 12 epochs
finish and the best validation checkpoint is fixed.

## Final screen comparison and decision

Final full-data metrics, fast-screen runtime/resource savings, ranking agreement, and
the next experiment decision will be added after jobs `14291479`, `14295585`, and
`14295587` complete.
