<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Continuous rollout: day 5 and day 360

Final epoch-70 EMA checkpoints, identical initial state on 2015-10-25.
Top row: day 5 (2015-10-30). Bottom row: day 360 (2016-10-19).
Columns: truth, ungated control, native feature-conditioned gate, control minus truth,
gate minus truth. The model feeds back its own predictions throughout 36 two-output
calls; the 360-day panel is not a fresh one-step forecast.

Fields share color limits across models within each row (pooled 1st–99th percentiles).
Errors use a shared symmetric 99th-percentile absolute-error limit within each row.
Limits change between lead times; compare color bars before comparing error magnitude.
No spatial smoothing. Global figures are JPEG previews; regional figures are PNG.
Regional crops use the same fixed equatorial-Pacific indices as the earlier examples.
These are illustrative fields from one origin, not the multi-date aggregate evaluation.

| Global area-weighted RMSE | Day 5 control | Day 5 gate | Day 360 control | Day 360 gate |
|---|---:|---:|---:|---:|
| SSH (m) | 0.02552 | 0.01988 | 0.04595 | 0.05824 |
| Surface temperature (°C) | 0.31212 | 0.21252 | 0.55362 | 0.74117 |
| Surface salinity (psu) | 0.11139 | 0.08506 | 0.24117 | 0.30946 |
| Surface U (m/s) | 0.04124 | 0.04086 | 0.06858 | 0.07234 |
| Surface V (m/s) | 0.03845 | 0.03695 | 0.06706 | 0.06972 |
| Temperature at 1050 m (°C) | 0.06027 | 0.02883 | 0.17215 | 0.18817 |
| Salinity at 1050 m (psu) | 0.01595 | 0.00421 | 0.02973 | 0.02935 |

The gate reduces short-range texture errors, especially in deep T/S. That benefit
is not a general improvement in 360-day endpoint skill: most displayed global fields
are worse, while deep salinity and some regional fields remain better.
Full-precision global and regional metrics are in [map-metrics.json](map-metrics.json).

## Surface temperature

![Surface temperature, global](thetao_0-global.jpg)

![Surface temperature, equatorial Pacific](thetao_0-regional.png)

## Salinity at 1050 m

![Salinity at 1050 m, global](so_11-global.jpg)

![Salinity at 1050 m, equatorial Pacific](so_11-regional.png)

## Temperature at 1050 m

![Temperature at 1050 m, global](thetao_11-global.jpg)

![Temperature at 1050 m, equatorial Pacific](thetao_11-regional.png)

## SSH

![SSH, global](zos-global.jpg)

## Surface salinity

![Surface salinity, global](so_0-global.jpg)

## Surface U

![Surface U, global](uo_0-global.jpg)

## Surface V

![Surface V, global](vo_0-global.jpg)

