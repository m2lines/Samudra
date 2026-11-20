<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Constant learning-rate continuations

All three learning-rate continuations worsened final validation MSE and hard-bin NLL relative to the starting 25k classifier. Restarting at the original peak rate was the least damaging option. Doubling and quadrupling that rate produced large, fluctuating errors, especially with EMA weights. The sweep gives no evidence that a larger constant learning rate improves the trained classifier over these 3,000 additional updates.

These runs branch from the available final checkpoint after its cosine cooldown. Their constant rates of 0.0002, 0.0004 and 0.0008 are 1×, 2× and 4× the original peak rate, and 10×, 20× and 40× its final rate. They therefore test an abrupt post-cooldown restart. A continuation at the final low rate and branches from a pre-cooldown checkpoint remain untested.

## Fixed 3,000-update endpoint

The primary comparison uses EMA weights at global update 28,000. Each arm adds 768,000 examples to the original 6.4 million, reaching 7,168,000 cumulative examples. Normalized MSE divides physical errors by the same OM4 training-set standard deviation for each channel, retains the full-grid denominator with zero contribution from land, and averages equally across all 77 channels. Coverage-sampling importance correction estimates these full-grid quantities from the sampled validation points. Hard-bin NLL uses the original single-bin validation target and the same full-grid/channel aggregation. Lower values are better. Two-hot cross entropy remains the training objective, so the NLL comparison is a diagnostic of the categorical probabilities and does not establish calibration.

| Model or continuation | EMA normalized MSE | Change from starting classifier | EMA hard-bin NLL | Raw normalized MSE | Raw hard-bin NLL |
|---|---:|---:|---:|---:|---:|
| Starting classifier, 25k | 0.0362624 | — | 2.78919 | 0.0368641 | 2.80952 |
| 1× peak LR, 28k | 0.0375032 | +3.42% | 2.86426 | 0.0379907 | 2.97344 |
| 2× peak LR, 28k | 0.0926767 | +155.57% | 2.92787 | 0.0420289 | 3.11444 |
| 4× peak LR, 28k | 0.0452857 | +24.88% | 3.02283 | 0.0444399 | 3.14793 |
| Original regression, 25k | 0.0307856 | — | — | — | — |
| Persistence | 0.0391418 | — | — | — | — |

The 1× arm briefly reduced EMA normalized MSE by 0.245% after 50 additional updates, then deteriorated. Every sampled post-start EMA checkpoint had higher NLL than the starting model. The 2× and 4× arms peaked at EMA normalized MSE 0.13879 and 0.22540 after 2,000 additional updates. Their later reduction does not recover the starting score. These are single-seed trajectories, with eight fixed validation dates and 64 sampled points per depth per date. The small early improvement in the 1× arm is insufficient to establish a reproducible gain.

## Field and depth results

Each volumetric field below averages equally over its 19 depth channels. SSH has one channel. The full-grid denominator gives deep channels less total loss weight as their wet fraction shrinks. All values are final EMA normalized MSE.

| Field | Starting classifier | 1× peak LR | 2× peak LR | 4× peak LR | Original regression | Persistence |
|---|---:|---:|---:|---:|---:|---:|
| Temperature | 0.00021540 | 0.00214475 | 0.00717748 | 0.01199404 | 0.00083058 | 0.00021875 |
| Salinity | 0.00054130 | 0.00056528 | 0.06473688 | 0.00822354 | 0.00068506 | 0.00026116 |
| Eastward velocity u | 0.054388 | 0.056029 | 0.132139 | 0.059559 | 0.051889 | 0.056435 |
| Northward velocity v | 0.091750 | 0.093180 | 0.171461 | 0.103664 | 0.071240 | 0.101650 |
| SSH | 0.00120396 | 0.00127583 | 0.00133448 | 0.00163276 | 0.00223758 | 0.00117253 |

Every field worsens relative to the starting classifier in every arm. The 2× EMA salinity error at 250 m rises from 0.0023813 to 1.04031. Other large final increases include u at 4000 m and 22.5 m, and v at 10 m and 3100 m. In the 4× arm, temperature errors at 1850 and 2400 m rise to about 0.0444–0.0446. Raw and EMA weights can yield very different scalar errors, as the endpoint table shows. These aggregate depth results do not locate the responsible date-cell cases or establish the cause of the spikes.

[Field MSE curves](report/fields-full_grid_nmse.png) and [field NLL curves](report/fields-full_grid_nll.png) show all three rates, raw and EMA weights, and horizontal 25k classifier references. Regression references appear on MSE plots. The horizontal axis measures additional updates after 25k. The original regression is the matched-backbone, matched-recipe run from the [large-classifier comparison](../large-all-fields-classifier/README.md). It remains at 25k updates, so its compute budget differs from these 28k continuations.

| Field | Per-depth MSE curves | Per-depth hard NLL curves |
|---|---|---|
| Temperature | [MSE](report/thetao-full_grid_nmse.png) | [NLL](report/thetao-full_grid_nll.png) |
| Salinity | [MSE](report/so-full_grid_nmse.png) | [NLL](report/so-full_grid_nll.png) |
| u | [MSE](report/uo-full_grid_nmse.png) | [NLL](report/uo-full_grid_nll.png) |
| v | [MSE](report/vo-full_grid_nmse.png) | [NLL](report/vo-full_grid_nll.png) |
| SSH | [MSE](report/zos-full_grid_nmse.png) | [NLL](report/zos-full_grid_nll.png) |

Download [all channel curves](report/validation-channels.csv.gz), [field curves](report/validation-fields.csv.gz), [fixed-endpoint comparisons](report/fixed-endpoint-summary.csv.gz), [training history](report/training-progress.csv.gz) and [rendering provenance](report/render-receipt.json.gz). The CSVs retain initialization at global update 25,000, physical RMSE, normalized MSE, NLL and persistence. Physical RMSE in these tables also retains the full-grid denominator.

## State preservation and execution

All three branches restore the same raw weights, optimizer moments, EMA state, per-rank NumPy and Torch RNG states, and frame-stream state. The model has 216.37 million parameters, a 33×33 native quarter-degree crop, continuous inputs, an 8,192-bin categorical head, two-hot training loss and argmax-center scalar forecasts. Each run uses four RTX PRO 6000 Blackwell GPUs, microbatch 32, two accumulation steps, effective batch 256, and 128 accepted crops per loaded time window per GPU. Coverage sampling and the original data stream are retained.

Validation runs before the first branch update and after 50, 100, 250, 500, 1,000, 2,000 and 3,000 additional updates. Per-arm best tracking starts again at the branch boundary, while optimizer and EMA update counts continue from 25,000. The final checkpoint is the primary result regardless of intermediate validation minima. Each arm's gzip-compressed configuration is stored here as `1x-train.json.gz`, `2x-train.json.gz` or `4x-train.json.gz`.

The explicit branch loader verifies the parent checkpoint checksum, source receipt, world size and scientific configuration. Ordinary preemption resume retains exact configuration and source checks. All three initial raw-weight and EMA validation MSE/NLL values match the parent exactly for every channel. Restored parameter hashes match the parent, and all four final RNG/frame-stream states match across the three production arms. Every run finishes at optimizer/EMA update 28,000 with agreeing replicas. [Qualification audit](audit-qualification.json.gz) and [final audit](audit-train.json.gz) preserve these checks.

| Allocation | Job | Wall seconds | GPU-hours |
|---|---:|---:|---:|
| Three four-update qualifications | 23208189 | 422 | 0.469 |
| 1× production | 23216154 | 7,691 | 8.546 |
| 2× production | 23216161 | 7,776 | 8.640 |
| 4× production | 23216163 | 7,738 | 8.598 |
| Total | | | 26.252 |

The qualification and final CPU audits took 18 and 17 seconds respectively on four CPUs. All jobs completed successfully. No full-grid inference was performed for this sweep. The source is pinned to `d06db8c7d`, with the immutable parent checkpoint and archive hashes recorded in the [execution receipt](execution.json.gz). Final checkpoints remain under `/scratch/jr7309/skyline-lr-sweep-20261115/{1x,2x,4x}-train/latest.pt`. Twelve targeted CPU tests passed, covering branch provenance rejection, preserved checkpoint state, constant/default schedules and the existing soft-prior implementation. This PR is an experiment/results record and is not intended to merge.
