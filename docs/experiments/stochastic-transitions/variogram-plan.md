<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Variogram initializer experiments

Train two models from scratch for 2,000 mixed optimizer updates each. Add a spatial variogram score to the OM4 initializer velocity loss. Compare with the completed runs in [shared-noise results](shared-noise-results.md), using their original random seeds and data sequence.

| New run | Existing control | Initializer |
|---|---|---|
| B-learned-variogram | B-learned-control | Learned coarse noise; widths 64/128/256; one ConvNeXt block per level |
| B-shared-deep-wide-variogram | B-shared-deep-wide | Coarse and grid noise enter the same network as conditioning; widths 96/192/384; three ConvNeXt blocks per level |

Both retain the deterministic processor, effective batch eight, four training members, seed 1729, learning rate 0.0001, EMA half-life 100 updates, and alternating observation/OM4 training. Each sees 1,000 updates of each source, or 8,000 examples of each source. There is no replay or cooldown. Checkpoints are saved every 500 updates. Qualification uses the first 24 updates of each fresh run, then training resumes to the total of 2,000.

Before release, compare the fresh step-zero model and EMA tensors exactly with each control's saved step-zero checkpoint. The training contract must match except for the code-snapshot identifier and the added variogram objective. Observation, forecast, and monthly-interior losses remain unchanged.

## Added objective

For each spatial pair, let `t_m = abs(x_m(a) - x_m(b))` be member m's velocity increment and `t_y = abs(y(a) - y(b))` the target increment, using the same channel normalization. The auxiliary score is:

```
(mean(t_m) - t_y)^2 - sample_variance(t_m) / M
```

The variance correction makes this an unbiased estimator of the squared error in the ensemble's expected absolute increment, assuming independent members conditional on the inputs. Individual estimates can be negative. Each member receives independent noise; “shared” refers to processing coarse noise, fine noise, and conditioning together within that member. Existing pointwise fair CRPS is retained.

Use all u/v channels at 19 depths, normalized by the existing OM4 channel statistics. Compare pairs separated by 1, 2, 4, or 8 grid cells in east, north, northeast, and northwest directions. Longitude wraps; latitude does not. Every cell along a pair's discrete straight path must be wet and target-finite. Weight pairs by mean endpoint area, then average supported examples/channels within each direction, directions within each lag, and finally lags. Entirely unsupported batches fail the finite-loss check. These are computational-grid separations, not equal physical distances across latitude.

Set auxiliary weight to 1 inside the existing initializer reconstruction loss, which has outer weight 0.1. Thus the added contribution to an OM4 update is `0.1 * variogram_score`. Log the unweighted score, plug-in term, variance correction, weighted contribution, and base initializer CRPS separately. The score rewards target-like increment magnitudes; it does not impose a general smoothness penalty or establish correct current placement.

This adapts the variogram scoring approach of [Scheuerer and Hamill (2015)](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf), using order one and an explicit finite-ensemble correction.

## Evaluation and compute

Use the same held-out dates and eight evaluation members as the controls. At 500, 1,000, and 2,000 updates, evaluate initializer predictions using the raw or EMA version of the entire model. Evaluate final 5/15/30-day forecasts with the unchanged scoring procedure.

Compare velocity ensemble-mean RMSE, pointwise fair CRPS, member maps, spatial variogram scores, spectra, vorticity/divergence, and fixed-noise cross-date similarity. Reuse regions, depths, spectral bins, and the cosine similarity of fixed-noise member perturbations across dates from [the shared-noise spatial diagnostic script](../../../scripts/diagnose_stochastic_shared_spatial.py). Also check SSH and other fields for regressions. Report member realism separately from pointwise skill. The main question is whether held-out spatial scores and member structure improve while preserving pointwise skill; report the tradeoff rather than declaring a winner from a preset aggregate. This is a single-seed screen; lower training variogram loss alone is insufficient evidence of improvement.

Qualify both on Torch RTX GPUs, requiring finite losses/gradients, matched initial weights, and greater than 50% mean GPU utilization, sampled with nvidia-smi once per second during updates 5–24 on each allocated GPU. Inspect loss magnitude, memory, and update time before releasing production. Use one GPU per run. Record per-run production wall caps after qualification, allowing for startup and evaluation. Qualification updates count toward the 2,000-update total; their allocation time is charged once. Keep actual qualification cost plus production allocation caps and the prior campaign and failed first qualification's 21.24 GPU-hours below 40 GPU-hours in total. A changed source or allocation requires a new recorded release check.

The first qualification failed on the first OM4 update because the new score required full-grid area weights while native training supplies latitude-only weights. Both failed allocations used 0.3131 GPU-hours in total. The corrected score expands latitude-only weights without copying, with regression tests for exact score/gradient equivalence and the native training path. Recovery restarts both runs from their original random seed; failed artifacts remain preserved.
