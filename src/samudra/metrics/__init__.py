# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observation-based evaluation metrics for finished rollouts.

Unlike `samudra.aggregator`, which accumulates statistics incrementally as a
rollout streams past, these metrics operate on a *finished* rollout that has
been written to disk. Several of them (calendar-year block bootstrap, detrended
and deseasonalized residual variance) have no incremental form at all: they need
the whole multi-year time series in hand.

The pieces:

- `kernels`: pure functions over `xr.DataArray`. No I/O, no plotting, no state.
- `observations`: loaders for the DUACS / OISST / ARGO-IAP products.
- `report`: the driver, producing one tidy `pandas.DataFrame` of metric rows.
- `config`: `ObsMetricsConfig`, embedded in `EvalConfig`.

`samudra.eval` calls `report` for the scalars it logs to W&B; `samudra.viz`
calls the same `kernels` when it draws the corresponding figures, so a number on
a figure cannot drift from the scalar logged next to it.
"""
