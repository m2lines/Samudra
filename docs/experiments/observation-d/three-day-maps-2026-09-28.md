<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Samudra2 checkpoint and rollout maps

Companion to the [completed three-arm comparison](three-day-findings-2026-09-26.md).
Model definitions, training details and aggregate results are in that report.
These diagnostic views do not select checkpoints or change the scoring protocol.

## Initializer and day-30 maps across observation training

Columns compare **Samudra2 scratch**, **Samudra2 sequential** and **Samudra2 mixed**
on the same November 2013 validation origin. Early panels contain 0, 10 and 100
observation updates; later panels contain 1,000, 4,000 and 8,000. Where later panels
are split, links 1/2/3 follow that order. Zero is the common random model before
both tasks, not a pretrained model at zero observation updates. OM4 exposure
therefore differs at the same observation count. All are raw milestone weights.

Full-grid maps include polar regions outside the 60°S–60°N scoring domain.
Initializer velocity and deep-state channels are unconstrained internal variables;
their nominal physical units do not establish physical reconstruction accuracy.
Surface-derived geostrophic velocity is shown separately below.

| Field | Early checkpoints | Later checkpoints |
| --- | --- | --- |
| Initializer SST | [view](artifacts/2026-09-26-three-day/production-initial-sst-early.png) | [view](artifacts/2026-09-26-three-day/production-initial-sst-later.png) |
| Initializer temperature at 550 m | [view](artifacts/2026-09-26-three-day/production-initial-t550-early.png) | [view](artifacts/2026-09-26-three-day/production-initial-t550-later.png) |
| Initializer eastward velocity slot | [view](artifacts/2026-09-26-three-day/production-initial-u-early.png) | [1](artifacts/2026-09-26-three-day/production-initial-u-later-part1.png), [2](artifacts/2026-09-26-three-day/production-initial-u-later-part2.png), [3](artifacts/2026-09-26-three-day/production-initial-u-later-part3.png) |
| Initializer SSH slot | [view](artifacts/2026-09-26-three-day/production-initial-adt-early.png) | [view](artifacts/2026-09-26-three-day/production-initial-adt-later.png) |
| Day-30 SST | [view](artifacts/2026-09-26-three-day/production-day30-sst-early.png) | [1](artifacts/2026-09-26-three-day/production-day30-sst-later-part1.png), [2](artifacts/2026-09-26-three-day/production-day30-sst-later-part2.png), [3](artifacts/2026-09-26-three-day/production-day30-sst-later-part3.png) |
| Day-30 ADT | [view](artifacts/2026-09-26-three-day/production-day30-adt-early.png) | [1](artifacts/2026-09-26-three-day/production-day30-adt-later-part1.png), [2](artifacts/2026-09-26-three-day/production-day30-adt-later-part2.png), [3](artifacts/2026-09-26-three-day/production-day30-adt-later-part3.png) |

### Scratch extension: raw 8k versus raw 16k

The selected scratch checkpoint is still 7,900; these raw endpoint maps illustrate
what continued training changes without implying improvement in selection score.

- Initializer SST: [view](artifacts/2026-09-26-three-day/production-scratch-initial-sst-8k-16k.png)
- Initializer velocity slot: [view](artifacts/2026-09-26-three-day/production-scratch-initial-u-8k-16k.png)
- Day-30 SST: [view](artifacts/2026-09-26-three-day/production-scratch-day30-sst-8k-16k.png)
- Day-30 ADT: [view](artifacts/2026-09-26-three-day/production-scratch-day30-adt-8k-16k.png)

## Continuous annual forecasts at selected checkpoints

Each figure compares observations with all three selected models at days 30 and
365. Future ERA5 forcing is prescribed. The three origins are the fixed annual
evaluation cohort; these examples do not replace their aggregate errors.

| Initialization | SST | ADT |
| --- | --- | --- |
| 2015-01-01 | [view](artifacts/2026-09-26-three-day/production-annual-sst-2015-01-01.png) | [1](artifacts/2026-09-26-three-day/production-annual-adt-2015-01-01-part1.png), [2](artifacts/2026-09-26-three-day/production-annual-adt-2015-01-01-part2.png) |
| 2018-01-01 | [view](artifacts/2026-09-26-three-day/production-annual-sst-2018-01-01.png) | [1](artifacts/2026-09-26-three-day/production-annual-adt-2018-01-01-part1.png), [2](artifacts/2026-09-26-three-day/production-annual-adt-2018-01-01-part2.png) |
| 2021-01-01 | [view](artifacts/2026-09-26-three-day/production-annual-sst-2021-01-01.png) | [1](artifacts/2026-09-26-three-day/production-annual-adt-2021-01-01-part1.png), [2](artifacts/2026-09-26-three-day/production-annual-adt-2021-01-01-part2.png) |

## Regional velocity and spectra

Rows use the metric regions: North Pacific (170–200°E, 25–45°N), Gulf Stream
(300–320°E, 25–45°N), and Agulhas (40–60°E, 50–30°S). Velocity is computed from
forecast ADT, with the same fixed observed derivative/support masks as evaluation.
It is not the initializer's internal velocity slots. Gray denotes unavailable
common support. Reference maps use observed velocity on that support.

- Eastward velocity: July 2022, day 30: [view](artifacts/2026-09-26-three-day/production-regional-velocity-0.png)
- Northward velocity: July 2022, day 30: [view](artifacts/2026-09-26-three-day/production-regional-velocity-1.png)
- Speed: 2015 continuous rollout, day 30: [view](artifacts/2026-09-26-three-day/production-regional-speed-day30.png)
- Speed: 2015 continuous rollout, day 365: [view](artifacts/2026-09-26-three-day/production-regional-speed-day365.png)
- Mean day-30 EKE over 96 monthly origins: [view](artifacts/2026-09-26-three-day/production-regional-monthly-eke.png)
- Mean EKE over the 2015 continuous year: [view](artifacts/2026-09-26-three-day/production-regional-annual-eke.png)
- Monthly EKE spectra, days 5/15/30: [view](artifacts/2026-09-26-three-day/production-monthly-eke-spectra.png)
- Annual EKE spectra, mean of three years: [view](artifacts/2026-09-26-three-day/production-annual-eke-spectra.png)

EKE maps average instantaneous EKE; spectra average the corresponding spectral
powers, so a spectrum of the displayed mean map would not reproduce the metric.
Monthly anomalies use each fixed-lead cohort mean; annual anomalies use each
continuous sequence's own mean. Spectral agreement does not require features to
be correctly located, and is not a fraction of physical kinetic energy.

[Plotting source](artifacts/2026-09-26-three-day/plot_final_maps.py.txt) and
[provenance audit](artifacts/2026-09-26-three-day/production-map-audit.json.gz)
retain input checksums, extraction provenance and cell-geometry checks.
