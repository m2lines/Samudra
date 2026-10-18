<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Residual replay: 6k versus 8k observation-update maps

These shared-scale maps compare the continuing mixed-training residual model at 6,000 and 8,000 observation updates. They do **not** compare the separate observation-only cooldown fork. See the [main metrics report](scratch-replay-residual-8k.md) for task counts and aggregate rollout results.

For this 2018 origin, day-30 SST has mixed changes: the single draw has lower RMSE at 8k, while the ensemble-mean RMSE rises slightly and its bias becomes more negative. The day-365 SSH error pattern becomes stronger in several regions. Native salinity members remain conspicuously grainy at initialization and day 365, with a more negative mean bias. These examples do not show a general removal of grain. They are individual-origin diagnostics, not aggregate skill estimates.

Each cell occupies exactly one pixel in the saved map, without interpolation or smoothing. Open images at native size if the browser scales this table. Gray denotes land or missing data. Physical, error, and spread scales are shared across both checkpoints and all selected leads **within each channel and source**. Reference percentiles define physical ranges; pooled absolute errors define error ranges. Legends and JSON metadata record clipping and full ranges. Error is prediction minus reference; spread is pointwise sample standard deviation, not error.

The mean uses eight members. Draw 1 uses an independent sample at each lead and is not a coherent trajectory. Annual observed SST/SSH are five-day verification bins; native OM4 fields use their output cadence. Native day 0 tests surface-conditioned reconstruction against OM4, not an observed interior gold standard.

## Observed SST, day 30

| Residual 6k | Residual 8k |
| --- | --- |
| ![Residual6k thetao_0 mean](scratch-replay-residual-8k-assets/annual/Residual6k-thetao_0-day30-mean.png) | ![Residual8k thetao_0 mean](scratch-replay-residual-8k-assets/annual/Residual8k-thetao_0-day30-mean.png) |
| ![Residual6k thetao_0 mean-error](scratch-replay-residual-8k-assets/annual/Residual6k-thetao_0-day30-mean-error.png) | ![Residual8k thetao_0 mean-error](scratch-replay-residual-8k-assets/annual/Residual8k-thetao_0-day30-mean-error.png) |
| ![Residual6k thetao_0 draw1-error](scratch-replay-residual-8k-assets/annual/Residual6k-thetao_0-day30-draw1-error.png) | ![Residual8k thetao_0 draw1-error](scratch-replay-residual-8k-assets/annual/Residual8k-thetao_0-day30-draw1-error.png) |

[Reference](scratch-replay-residual-8k-assets/annual/Residual8k-thetao_0-day30-reference.png), [8k single draw field](scratch-replay-residual-8k-assets/annual/Residual8k-thetao_0-day30-draw1.png), [8k spread](scratch-replay-residual-8k-assets/annual/Residual8k-thetao_0-day30-spread.png).

## Observed SSH, day 365

| Residual 6k | Residual 8k |
| --- | --- |
| ![Residual6k zos mean](scratch-replay-residual-8k-assets/annual/Residual6k-zos-day365-mean.png) | ![Residual8k zos mean](scratch-replay-residual-8k-assets/annual/Residual8k-zos-day365-mean.png) |
| ![Residual6k zos mean-error](scratch-replay-residual-8k-assets/annual/Residual6k-zos-day365-mean-error.png) | ![Residual8k zos mean-error](scratch-replay-residual-8k-assets/annual/Residual8k-zos-day365-mean-error.png) |
| ![Residual6k zos draw1-error](scratch-replay-residual-8k-assets/annual/Residual6k-zos-day365-draw1-error.png) | ![Residual8k zos draw1-error](scratch-replay-residual-8k-assets/annual/Residual8k-zos-day365-draw1-error.png) |

[Reference](scratch-replay-residual-8k-assets/annual/Residual8k-zos-day365-reference.png), [8k single draw field](scratch-replay-residual-8k-assets/annual/Residual8k-zos-day365-draw1.png), [8k spread](scratch-replay-residual-8k-assets/annual/Residual8k-zos-day365-spread.png).

## OM4 salinity reconstruction, day 0, level 9 (550 m)

| Residual 6k | Residual 8k |
| --- | --- |
| ![Residual6k so_9 mean](scratch-replay-residual-8k-assets/native/Residual6k-so_9-day0-mean.png) | ![Residual8k so_9 mean](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day0-mean.png) |
| ![Residual6k so_9 mean-error](scratch-replay-residual-8k-assets/native/Residual6k-so_9-day0-mean-error.png) | ![Residual8k so_9 mean-error](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day0-mean-error.png) |
| ![Residual6k so_9 draw1-error](scratch-replay-residual-8k-assets/native/Residual6k-so_9-day0-draw1-error.png) | ![Residual8k so_9 draw1-error](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day0-draw1-error.png) |

[Reference](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day0-reference.png), [8k single draw field](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day0-draw1.png), [8k spread](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day0-spread.png).

## OM4 salinity, day 365, level 9 (550 m)

| Residual 6k | Residual 8k |
| --- | --- |
| ![Residual6k so_9 mean](scratch-replay-residual-8k-assets/native/Residual6k-so_9-day365-mean.png) | ![Residual8k so_9 mean](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day365-mean.png) |
| ![Residual6k so_9 mean-error](scratch-replay-residual-8k-assets/native/Residual6k-so_9-day365-mean-error.png) | ![Residual8k so_9 mean-error](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day365-mean-error.png) |
| ![Residual6k so_9 draw1-error](scratch-replay-residual-8k-assets/native/Residual6k-so_9-day365-draw1-error.png) | ![Residual8k so_9 draw1-error](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day365-draw1-error.png) |

[Reference](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day365-reference.png), [8k single draw field](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day365-draw1.png), [8k spread](scratch-replay-residual-8k-assets/native/Residual8k-so_9-day365-spread.png).

## Single-map error checks

These are unweighted RMSEs over finite ocean cells for the pictured origin, not the climatology-normalized blended metric. They illustrate why changes in appearance should be checked against error and bias rather than interpreted as a universal improvement.

| Map | 6k mean RMSE | 8k mean RMSE | 6k draw-1 RMSE | 8k draw-1 RMSE |
| --- | ---: | ---: | ---: | ---: |
| Observed SST, day 30 (°C) | 0.57722 | 0.59501 | 0.73122 | 0.71299 |
| Observed SSH, day 365 (m) | 0.12326 | 0.14285 | 0.12936 | 0.14887 |
| OM4 salinity level 9, day 0 | 0.09239 | 0.09855 | 0.13489 | 0.13668 |
| OM4 salinity level 9, day 365 | 0.08845 | 0.09702 | 0.13456 | 0.13921 |

## Full panel collections

The [annual directory](scratch-replay-residual-8k-assets/annual) contains SSH/SST reference, ensemble mean, draw 1, both error panels, and spread at days 30 and 365. The [native directory](scratch-replay-residual-8k-assets/native) contains the same panels for SSH/SST and level-9 temperature/salinity at days 0, 30, and 365.

[Annual metadata](scratch-replay-residual-8k-assets/annual/map-ranges.json) and [native metadata](scratch-replay-residual-8k-assets/native/map-ranges.json) contain shared scale limits, finite/nonfinite counts, clipping fractions, unclipped ranges, and unweighted map error/bias diagnostics. Filenames identify checkpoint, variable, lead, and panel. All PNGs remain below the repository asset-size limit.
