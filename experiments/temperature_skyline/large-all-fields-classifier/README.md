<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Large all-fields classifier at 25,000 updates

The 216.37M-parameter classifier completed 25,000 updates with the original 207.97M-parameter all-fields regressor's backbone, crop, initialization seed and training recipe. Temperature and salinity validation MSE improved over that regressor, while eastward and northward velocity MSE increased by 4.8% and 28.8%. On one complete held-out ocean state, the classifier improved velocity-vector RMSE over persistence at 16 of 19 depths. Its predicted five-day velocity changes remained small and weakly aligned with OM4 changes. Samudra had lower velocity-vector RMSE at every depth.

The Skyline comparisons use final-update EMA weights. OM4 is the simulation reference. The comparisons cover one training seed, eight sampled validation dates, and one full-grid test date. The categorical objective, head initialization and scalar readout change together, so their individual effects remain unresolved.

## Matched validation

Each validation date contains 64 sampled points per depth, with coverage-sampling importance correction. The table reports normalized MSE averaged equally over the 19 channels of each volumetric field. SSH has one channel. As in the original Samudra loss, land contributes zero while the denominator retains the full grid. Deep channels therefore receive less total weight as their wet area shrinks. Errors are divided by the same OM4 training-set standard deviation for each channel across all models and persistence. Lower values are better. These sampled metrics differ from the area-weighted wet-ocean physical RMSE used in the maps.

| Field | Large classifier | Original large regression | Persistence |
|---|---:|---:|---:|
| Temperature | 0.00021540 | 0.00083058 | 0.00021875 |
| Salinity | 0.00054130 | 0.00068506 | 0.00026116 |
| Eastward velocity u | 0.054388 | 0.051889 | 0.056435 |
| Northward velocity v | 0.091750 | 0.071240 | 0.101650 |
| SSH | 0.00120396 | 0.00223758 | 0.00117253 |

The classifier beats the regressor at 18/19 temperature depths, 15/19 salinity depths, 4/19 u depths and 3/19 v depths. Across all 77 channels, its normalized MSE is 0.0362624 versus 0.0307856 for regression and 0.0391418 for persistence. EMA improves classifier MSE from the raw-weight value of 0.0368641. Hard-bin validation NLL is 2.78919 with EMA and 2.80952 with raw weights. Two-hot training changes the probability target, so these NLL values are diagnostics rather than an established calibration result.

[Field validation curves](report/curves/field-full_grid_nmse.png) include raw and EMA weights from initialization and update 1 onward. Per-depth curves are available for [temperature](report/curves/thetao-full_grid_nmse.png), [salinity](report/curves/so-full_grid_nmse.png), [u](report/curves/uo-full_grid_nmse.png) and [v](report/curves/vo-full_grid_nmse.png). Download [all channel curves](report/curves/validation-channels.csv.gz), [field curves](report/curves/validation-fields.csv), [training history](report/curves/training-progress.csv.gz) and [curve provenance](report/curves/curve-receipt.json.gz).

## Full native ocean state

All 77 channels were predicted on the native 720×1440 grid for July 21, 2016, using July 11 and July 16 inputs. The classifier evaluated exactly 10,669,976 wet depth-point queries, producing 43,370,201 valid scalar values. Each depth-point pass predicts all four volumetric fields together, plus SSH at the surface. The Samudra comparator is the quarter-degree two-input, one-output model from run 211241, epoch 62. It uses the previously saved full-state forecast with internal BF16 execution. Its inputs, target, masks and coordinates match this evaluation exactly. Samudra training budget and epoch-62 checkpoint selection are inherited from the existing benchmark and are not controlled against the Skyline runs.

| Field | Large classifier normalized MSE | Samudra | Persistence |
|---|---:|---:|---:|
| Temperature | 0.00024116 | 0.00055632 | 0.00024888 |
| Salinity | 0.00065713 | 0.00070339 | 0.00034596 |
| u | 0.071680 | 0.021199 | 0.071160 |
| v | 0.105384 | 0.029514 | 0.117366 |
| SSH | 0.00139665 | 0.00101937 | 0.00137149 |

These normalized MSEs use the same full-grid denominator and equal-depth aggregation as the validation table. The large classifier improves every field over the earlier 18.52M-parameter, 17×17-crop classifier on this date. That comparison also changes crop, seed and frame reuse, which prevents attributing the improvement solely to parameter count.

Surface velocity-vector RMSE is the square root of the wet-area-weighted mean of squared u and v errors summed together. Increment RMS uses forecast minus the latest true input. Alignment is the area-weighted dot product between predicted and true increments, divided by their RMS magnitudes. It ranges from −1 to 1, with 1 indicating aligned increments.

| Surface model | Vector RMSE, m/s | Improvement over persistence | Increment RMS, m/s | Increment alignment |
|---|---:|---:|---:|---:|
| Persistence | 0.095004 | 0% | 0 | Undefined |
| Small classifier | 0.094317 | 0.72% | 0.030656 | 0.184 |
| Large classifier | 0.090894 | 4.33% | 0.029900 | 0.292 |
| Samudra | 0.054627 | 42.50% | 0.087173 | 0.824 |

The true surface increment RMS is 0.095004 m/s. The large classifier predicts 31.5% of that amplitude. Its velocity-vector RMSE beats persistence at 16/19 depths, compared with 1/19 for the small classifier and 19/19 for Samudra. The large classifier's increment alignment ranges from 0.118 to 0.311 across depths. [Per-depth increment diagnostics](report/full-state/velocity-increments.csv) and [all 77 channel scores](report/full-state/global-channels.csv) retain the full results.

Global figures display native samples on a 1° lattice, with equal-sized hard pixels. Scores use every native wet cell. Regional figures show contiguous native 64×64 patches at 2.5, 250 and 1850 m, with a to-scale 33×33 context box. Field scales are shared within each row and clip at OM4's 0.5th/99.5th percentiles. Error scales clip at the pooled 99.5th percentile of absolute error.

- [Global depth-wise physical RMSE](report/full-state/global-depth-rmse.png)
- [Global u, upper four depths](report/full-state/global-uo-00.png) and [global v](report/full-state/global-vo-00.png)
- [North Pacific temperature](report/full-state/north-pacific-thetao.png), [Gulf Stream temperature](report/full-state/gulf-stream-thetao.png), [Agulhas temperature](report/full-state/agulhas-thetao.png)
- [All global and regional maps](report/full-state), including salinity and SSH
- [Regional metrics](report/full-state/regional-metrics.csv), [spectra](report/full-state/regional-spectra.csv.gz), [u error spectra](report/full-state/uo-error-spectra.png) and [v error spectra](report/full-state/vo-error-spectra.png)

Spectra use the same fully wet regional tiles for every model. They describe scale-dependent errors on these patches and this date. They do not establish a general spatial-coherence ranking.

## Original regression on regional velocities

The saved original 208M regressor is available on matched native 64×64 surface patches in the Gulf Stream and Agulhas. The table uses the same vector RMSE definition as the global comparison. Both patches contain 4,096 wet cells.

| Region | Large classifier, m/s | Original regression | Samudra | Persistence |
|---|---:|---:|---:|---:|
| Gulf Stream | 0.061402 | 0.053167 | 0.031096 | 0.059335 |
| Agulhas | 0.082388 | 0.073818 | 0.046821 | 0.079829 |

The classifier is 3.48% and 3.21% worse than persistence in these two regions. Regression improves on persistence by 10.40% and 7.53%. Classifier increment alignment is 0.0016 and 0.0440, compared with 0.4725 and 0.3917 for regression. Its global improvement therefore coexists with nearly unaligned changes in these two current systems. [Fields](report/regional-velocity/regional-velocity-fields.png), [errors with RMSE](report/regional-velocity/regional-velocity-errors.png), [component/vector metrics](report/regional-velocity/regional-velocity-metrics.csv) and [cache provenance](report/regional-velocity/regional-velocity-receipt.json.gz) document the comparisons.

## Architecture, matching and execution

The model has 16 transformer blocks, width 1024, eight attention heads and five field-query tokens at the sampled depth. It consumes continuous values from a 33×33 native quarter-degree crop and two input states separated by five days. Four volumetric fields span 19 depths, with SSH supervised only on surface samples. The fully trainable categorical head has 8,192 bins per query and uses channel-specific fixed asinh-spaced centers with training-audited bounds. Training uses adjacent-center two-hot cross entropy and Gaussian-smoothed head initialization with sigma four bins. Scalar forecasts use argmax centers. The classifier has 216,370,176 parameters, including 8,396,800 head parameters. The original regressor has 207,974,401 parameters, including a 1,025-parameter scalar head.

The fresh run preserves the original seed 20261106, four GPUs, microbatch 32, two accumulation steps, effective batch 256 and 128 accepted crops per loaded time window per GPU. The input-only coverage proposal uses 8,192 candidates, eight clusters per depth and a 0.5 base fraction. EMA decay is 0.999 with the inherited early-update adjustment. Learning rate warms up for 200 updates to 0.0002, then follows cosine decay to 0.00002 at update 25,000. Validation caches frames in memory. [Training configuration](train.json), [qualification configuration](qualify.json) and the [original regression report](../output-task-comparison/README.md) define the comparison.

Container reconstruction verified identical non-head initial parameters between the two models. Attention, crop extraction and coverage sampling code match the reference source. At completion, all four sampler RNG states and frame-stream states exactly match the original regression run. Importance means, effective-sample-size fractions and accepted-example counts match at all 1,251 shared logged updates. Together with the unchanged sampler code, these checks support matched exposure. Individual accepted coordinates were not logged throughout all 6.4 million examples. [Reference audit](reference-audit.json.gz), [initialization audit](initialization-audit.json.gz) and [final audit](final-audit.json.gz) retain the evidence.

CPU Torch convolution intermittently crashed during four-process smoothed-head initialization. The prior small classifier used width 512 and one process, while this model uses width 1024 and four processes. The stack trace identified the batched weight convolution. Disabling oneDNN did not resolve it, and the underlying cause remains unknown. The pinned implementation uses NumPy convolution on CPU with the same Gaussian kernel, boundary L2 normalization and RMS matching. Seven tests cover boundaries, short vocabularies, RMS and seeded initialization. At full head size, maximum differences from the old formula were 2.98e-8 in weights and 1.49e-8 in biases. A remote test completed 16 identical initializations across four processes. An inherited single-rank case-diagnostic flag was removed for distributed validation.

Production job 22998951 ran on four RTX PRO 6000 Blackwell GPUs for 17.36 wall hours, consuming 69.43 GPU-hours. Mean utilization after the first two minutes was 95.0–97.3% across ranks. Full-state inference job 23095245 took 6.23 wall hours on four GPUs, consuming 24.90 GPU-hours. Failed and successful GPU qualifications consumed another 0.414 GPU-hours. Total new GPU allocation was 94.75 GPU-hours, plus 54 seconds for the CPU initialization audit. [Execution receipt](execution.json.gz), [submission receipt](submission.json.gz) and [W&B run](https://wandb.ai/ocean_emulators/temperature-skyline/runs/55cc4bc52e27) record the jobs and costs.

Production and inference use source `7cf57f71ef52cc1ca9e715958ca740a8bd2b80d2`. Full-state physical arrays remain at `/scratch/jr7309/skyline-large-classifier-20261112/evaluation/full/arrays.npz`, with SHA-256 `87e551f931368c20a644b6ceefe44b58b753f546d4a195e20d4e3cc82486b99c`. The local downloaded copy matches exactly. [Full-state provenance](full-state-provenance.json.gz) records model, checkpoint, source and mask receipts. This PR is an experiment/results record and is not intended to merge.
