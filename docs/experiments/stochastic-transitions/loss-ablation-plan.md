<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Initializer loss ablation

The first comparison confounds stochastic feature modulation with a change from MSE to fair CRPS. On one held-out OM4 initialization case, B/C velocity reconstruction improves little while A improves substantially. These three fresh runs test whether the loss, stochastic features, or insufficient pressure on the ensemble mean explains that difference.

| Experiment | Architecture | Objective |
| --- | --- | --- |
| deterministic-mae | A: deterministic initializer and processor | MAE |
| stochastic-mse | B: stochastic initializer, deterministic processor | Mean of member squared errors |
| stochastic-crps-mean-mse | B | Fair CRPS + 1.0 × squared error of the ensemble mean |

Objectives apply to all existing supervised components, with unchanged weights: monthly 0.5, native forecast 0.5, initialization reconstruction 0.1. Errors use the existing channel normalization. The auxiliary coefficient is a fixed pilot choice, not a calibrated balance between gradients. Per-member MSE may suppress stochastic spread; that is an expected diagnostic outcome.

Each run starts from the same seed and trains for 2,000 total optimizer updates: 1,000 OM4 and 1,000 observation updates, effective batch eight. Stochastic training uses four members. Use unchanged data, splits, ConvNeXt widths, ERA5 adapter, learning rate, clipping, and EMA. No replay or cooldown. Save checkpoints every 500 updates. Reuse the original A/MSE and B/CRPS checkpoints at the same update counts as controls.

Before production, qualify finite losses and gradients, checkpoint/resume, correct objective configuration, and warm GPU utilization above 70%. Retain immutable source and data receipts. Execute on Torch with the existing resident data cache and known container workaround.

Evaluate raw/EMA checkpoints on the existing nine-origin forecast protocol. Recover initialization SST/SSH reconstruction scores from saved member arrays. Run the fixed held-out OM4 initializer-only comparison for T/S/U/V mean RMSE, member errors and CRPS; retain per-depth scores. Use eight evaluation members. Report the single-case limitation for interior curves, and distinguish observed-input reconstruction from future prediction. Compare curves and spread, not only endpoint RMSE.

## Multi-scale CRPS extension

Add a fourth fresh 2,000-update B-architecture arm with multi-scale fair CRPS, following the residual-band construction in Lang, Leutbecher and Hatfield, [section 2 and section 3.2](https://arxiv.org/html/2607.19161v1#S3.SS2). This isolates spatial scale awareness from the paper's separate almost-fair correction: retain the same fair estimator as the existing B control.

Use Gaussian smoothers with standard deviations 1, 2, 4 and 8 grid cells, truncated at three standard deviations. The five bands are the coarsest field, successive differences between smoothers, and the finest residual. Sum five equally weighted CRPS terms, retaining the existing per-variable normalization and task/component weights. Do not normalize each band by its own variance in this first pilot. Thus constant fields retain the single-scale score.

This is a grid-based adaptation, not a reproduction of the paper's graph smoothers with approximately 100/200/400/800 km widths or its variable-dependent weighting. Longitude is periodic. Latitude filtering renormalizes at the domain boundary. Each smoother uses common finite-target/wet support and area weights for all members and the target; land is excluded rather than treated as a zero-valued target. Filtering excludes land cells but can span narrow land barriers, a limitation of this inexpensive stencil. Keep that convention fixed and inspect coastal errors.

Qualify runtime, memory, gradients and utilization before production. Add member-wise spectra and scale-band errors to the comparison: improved mean RMSE alone would not establish improved spatial structure. Budget is one additional B-like run, approximately four GPU-hours plus qualification/evaluation, subject to measured smoothing overhead.

## Initializer-only extension and utilization policy

Add a fifth fresh 2,000-update arm using B's stochastic initializer and plain fair CRPS. Train only the initializer and shared ERA5 adapter. Freeze the processor and never invoke it during training. OM4 supervises all 77 channels of the last initialized history state; observations supervise only available latest-history SST/SSH. No instantaneous interior target is inferred from monthly IAP. Retain the existing initialization coefficient 0.1, seed, effective batch, member count, data schedule and EMA. Evaluate initialization only; a frozen untrained processor cannot provide a meaningful forecast comparison. The primary control is the original joint B/CRPS run at matched updates.

The user relaxed the warm GPU utilization requirement to strictly above 50% for these short runs. Preserve measured utilization and original qualification outcomes; record the updated acceptance policy rather than relabeling an old >70% check as passed. A-mae's measured 64.9% meets the revised policy without a microbatch change.
