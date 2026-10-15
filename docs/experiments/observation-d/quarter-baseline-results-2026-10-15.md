<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Quarter-degree baseline comparison

This compares fresh U-global and observation-only models at native quarter-degree resolution. Both completed 6,000 optimizer updates. The earlier-data variants are held at the user's request until these 365-day rollouts are reviewed.

**Obs-only has lower short-lead error; U-global has lower 365-day error under both checkpoint policies, mainly through better OHC.** Neither beats seasonal climatology on the overall year-long blend. These are native-resolution results from one seed and three annual starts; the matched 1° comparison remains pending.

## Models and checkpoint policies

All four table labels below refer to two trained models, each evaluated at two checkpoints. Both use the same 62,967,680-parameter initializer/evolution architecture, a global 720×1440 grid, 77 physical state channels, seed 1729, global batch eight and fixed learning rate 1e-4. Neither starts from the earlier D checkpoint. Training details and data provenance are in the [experiment plan](quarter-global-plan-2026-10-10.md).

| Literal result label | Meaning | OM4 / observation updates |
|---|---|---:|
| U-global selected | Checkpoint at update 4,900, chosen by the frozen nine-origin RMSE validation blend | 2,787 / 2,113 |
| U-global fixed6000 | Same training run at its final update; no checkpoint selection | 3,000 / 3,000 |
| Observation-only selected | Observation-only checkpoint at update 5,300, chosen by the same validation blend | 0 / 5,300 |
| Observation-only fixed6000 | Same observation-only run at its final update | 0 / 6,000 |

“Initialized persistence” holds that particular checkpoint's inferred initial physical state fixed throughout its rollout. Its state can differ across models and checkpoints. Training seasonal climatology is a separate control. The comparison matches total updates, not observation exposure: observation-only gets twice as many observation updates at the fixed endpoint. This is one seed, so differences do not establish reproducible effects across training randomness.

## Evaluation

The monthly test has 96 independently initialized starts and scores 5-, 15- and 30-day leads. Its five-component blend equally averages seasonal-climatology-normalized RMSE for SST, geostrophic velocity, EKE and OHC in the 0–700 m and 700–2000 m bands.

The annual test has three January starts (2015, 2018 and 2021), each evolved continuously for 73 five-day steps with one initialization and prescribed ERA5 forcing. At each lead, pool equal-origin MSE before taking square roots. Its four-component blend covers SST, geostrophic velocity and the two OHC bands, with lead-specific climatology denominators; annual temporal-anomaly EKE is reported separately. These annual scores are distinct from the monthly five-component blend. No spectra enter checkpoint selection or either blend.

Native results retain IAP's observation grid for subsurface/OHC comparisons. The matched 1° comparison conservatively coarsens surface predictions and uses the frozen shared supports and controls; native and common-grid score values are not interchangeable. Annual maps show five-day-bin means, with two pixels per model grid cell in each dimension.

## Results

**Observation-only has lower short-lead error, but U-global has lower year-long error under both checkpoint policies.** In this three-start annual cohort, U-global improves over its own day-365 persistence; observation-only does worse than its own persistence. All four evolved day-365 blends are worse than training seasonal climatology. The matched 1° comparison is still being collected; these native results alone do not establish a benefit from quarter-degree resolution.

| Model/checkpoint | Monthly test blend | Annual day 30 | Annual day 365 | Own persistence day 365 |
|---|---:|---:|---:|---:|
| U-global selected | 0.8741 | 0.9649 | 1.1108 | 1.2529 |
| U-global fixed6000 | 0.9126 | 1.0011 | 1.1290 | 1.1904 |
| Observation-only selected | 0.8509 | 0.9428 | 1.2941 | 1.2136 |
| Observation-only fixed6000 | 0.8872 | 0.9881 | 1.3387 | 1.2279 |
| Training seasonal climatology | 1.0000 | 1.0000 | 1.0000 | — |

The monthly and annual columns use their respective cohorts and definitions above. Among selected checkpoints, observation-only reduces the monthly blend by 2.7%, whereas U-global reduces the annual day-365 blend by 14.2%. This supports a possible long-horizon benefit of OM4 training in this setup; it does not isolate its mechanism or establish significance across seeds.

![Annual forecasts versus own persistence and climatology](artifacts/quarter-baseline-results-2026-10-15/native-annual/rmse-comparison.png)

## What drives the annual difference?

Day-365 pooled physical RMSE for the selected checkpoints:

| Model | SST (°C) | Geostrophic velocity (m/s) | OHC 0–700 m (GJ/m²) | OHC 700–2000 m (GJ/m²) |
|---|---:|---:|---:|---:|
| U-global selected | 0.947 | 0.1877 | 0.898 | 0.476 |
| Observation-only selected | 0.749 | 0.1933 | 1.142 | 0.687 |
| Training seasonal climatology | 1.023 | 0.1796 | 0.787 | 0.358 |

Observation-only still has better year-end SST. Its larger OHC errors outweigh that advantage in the four-component blend. Thus the annual difference is not simply a loss of surface-temperature skill. Both models have worse year-end velocity and OHC RMSE than seasonal climatology.

The per-origin scores and component errors are available in the [annual CSV](artifacts/quarter-baseline-results-2026-10-15/native-annual/rmse-scores.csv). [Monthly results and frozen denominators](artifacts/quarter-baseline-results-2026-10-15/native-monthly/results.json.gz) preserve the independent 96-start calculation.

## Day-365 maps

The following maps use the 2021 start and validation-selected checkpoints. Each includes observations, both evolved models, and each model's own initialized persistence, with shared color limits.

![Year-end SST anomalies](artifacts/quarter-baseline-results-2026-10-15/native-annual/maps/native-selected/day365-2021-01-01-anomalies-sst.png)

![Year-end ADT anomalies](artifacts/quarter-baseline-results-2026-10-15/native-annual/maps/native-selected/day365-2021-01-01-anomalies-adt.png)

Additional maps: [selected checkpoints](artifacts/quarter-baseline-results-2026-10-15/native-annual/maps/native-selected/) and [fixed-6k checkpoints](artifacts/quarter-baseline-results-2026-10-15/native-annual/maps/native-fixed6000/). Anomaly maps cover all three starts; absolute-field maps cover the 2021 start. ADT maps are diagnostic: ADT itself is not a component of the reported RMSE blend.

## Initialized and evolved temperature profiles

![Temperature profiles at initialization, day 30 and day 365](artifacts/quarter-baseline-results-2026-10-15/native-annual/temperature-profiles.png)

Profiles are wet-area means at each of the 19 model depths, averaged equally across the three origins. Observation-only has nearly depth-independent deep temperatures below the IAP-supervised range. These are diagnostics of the retained model state, not evidence of observed accuracy at those depths. The deep extrapolation should not be conflated with the scored 700–2000 m OHC band. Means can also conceal substantial regional errors; the [profile CSV](artifacts/quarter-baseline-results-2026-10-15/native-annual/temperature-profiles.csv) includes individual-origin means and RMS.

## Verification and status

Both training runs and all eight evaluation modes completed. Each model's evaluation used the full 96-month and three-year-start cohorts, with selected/final checkpoint hashes and all 434 output files independently verified. Training consumed 339.7467 GPU-hours; training qualification/retries added 28.0061. The annual memory check and both full evaluations added 8.3967 GPU-hours, for 376.1494 GPU-hours so far. Score/map collection uses CPU jobs; no diagnostic weights or single-origin scores enter these results.

The [native annual provenance](artifacts/quarter-baseline-results-2026-10-15/native-annual/README.md) records source, data, checkpoint, artifact and pixel checks. The early-data jobs remain held pending discussion of these rollouts. The matched 1° report is pending; no early-data jobs have been released.

