<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# OM4 fully wet cell experiment

This experiment tests whether excluding partially wet cells improves Samudra v2 predictions on fully wet cells. It is an experiment/results record and is not intended to merge.

The baseline uses the [published September 2026 1° OM4](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_onedeg/OM4.zarr/.zmetadata) masks. The treatment uses the same data values and normalization statistics, with stricter per-layer masks. A destination cell is retained only when conservative regridding of the native indicator `wet == 1 and deptho >= layer_bottom` gives a full-area fraction of at least `1 - 1e-6`; its grid coverage must also be within `1e-6` of one. Land contributes zero to this calculation. This excludes horizontal coastal mixtures and vertically truncated bottom cells. It is stricter than thresholding horizontally averaged bathymetry.

The model treats excluded cells as ordinary land: normalized input and target values are zero, predictions are masked, and those cells contribute no training loss. Surface fields use the first-layer mask. Zarr views link unchanged arrays to the original store; no source chunks are rewritten. Depth labels denote nominal layer centers. The geographic-cell-area-weighted retained fractions of originally wet area are 90.3% at the surface, 82.4% at 1,850 m, 4.6% at 5,000 m, and zero at 6,000 m. See [mask manifest](mask_manifest.yaml) for all layers and geometry hashes.

Both arms train from identical seed-15 initialization for 70 epochs, using the [v2 preset](../../src/samudra/configs/samudra_om4_v2/model.yaml) batch-normalized ConvNeXt U-Net in FP32, two input/two output states, four autoregressive training calls, batch 4 per GPU, four GPUs, and no gradient accumulation: effective batch 16. States are five days apart; four calls each predict two new states, giving a 40-day training forecast. Learning rate is 0.0006 with cosine scheduling; dynamic MSE weights track inverse per-channel MSE with a 25-update moving average and a maximum/minimum weight ratio of 20. The train interval is 1975-01-03 through 2013-09-30, and validation is 2013-10-05 through 2014-10-05. Both use the Rust loader and `tau_hfds` forcing (wind stresses and heat flux); this omits the derived heat-flux-anomaly input in the shipped preset, as in the existing EAI Rust-loader runs.

The primary comparison uses final-epoch EMA checkpoints, with EMA decay 0.999, avoiding checkpoint selection on different validation masks. The architecture retains all 77 prognostic channels even though the strict mask leaves the four 6,000 m channels empty. Rollout validation excludes empty channels from its normalized aggregate and still rejects nonfinite predictions on wet cells. Empty channels are unscorable, not zero-error results.

Each model runs four independent 360-day forecasts beginning with the first two available states in 2015, 2017, 2019, and 2021. Each model keeps its own input mask. Scoring uses identical strict masks for both models, and raises on missing/nonfinite predictions within that support. The primary metric averages spatial area-weighted RMSE divided by the original training standard deviation equally over nonempty channels, 72 forecast lead times, and four origins. Lower is better. CSVs also report physical RMSE and mean bias (prediction minus truth) separately for each variable/depth channel. Persistence repeats the latest observed history state at every lead. Paired summaries are provided by variable and origin. This is one matched training seed against OM4 simulation output; it does not establish observational accuracy. Masking also changes the dynamic loss weights and the available boundary context.

## Execution

The experiment runs on EAI Beta, with one four-GPU production node per arm and explicit standard QoS. Persistent data and results live under `/projects/ny/lz1955/multiscale/jrusak`; the run directory is `runs/om4-nonpartial-v2-20261105`. The pinned ARM64 image is recorded in `train.sbatch`. W&B is enabled for both full training runs in `ocean_emulators/default`, group `om4-nonpartial-v2-20261105`.

1. Run `build_masks.py OUTPUT` in a CPU environment with Xarray, Zarr 2, and xESMF/ESMPy. Copy its outputs to the run directory's `geometry/`.
2. Stage this source revision, its archive checksum, and the compatible Rust extension in `source/` and `native/`.
3. Submit `train.sbatch` as a two-task array. Each arm first shares an exclusive qualification gate with any short test job. The gate checks the actual masks, finite full-shape CUDA forward/backward, empty-channel loss handling, and identical initialization hashes. Its fingerprint covers the source archive, configs, and mask manifest.
4. Each arm trains on four GPUs, then uses all four GPUs for the four independent forecast origins and writes scores. After both finish, run `score.py RUN_ROOT compare` to create the paired CSV and comparison figure.

The qualification gate uses a shared file lock and a fingerprinted success marker. Only one job performs the checks; failures stop that job before training and do not create a success marker.

No quality result is available yet. Job receipts and results will be added after execution.
