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
- `spectra`: the same, for the transforms that answer *at which scales* a model
  is wrong rather than by how much.
- `observations`: loaders for the DUACS / OISST / ARGO-IAP products.
- `comparisons`: model and observation fields aligned and put on one grid, which
  is everything scoring and drawing have in common.
- `report`: the driver, producing one tidy `pandas.DataFrame` of metric rows.
- `run`: one observation pass, returning the products, the prepared rollouts,
  the frame and the W&B scalars.

`ObsMetricsConfig` lives in `samudra.config` with the other task configs.

`samudra.eval` takes the scalars from `run` and logs them; `samudra.viz` takes
the same pass and draws it. Both reduce through `kernels`, so a number on a
figure cannot drift from the scalar logged beside it: they are one calculation.

One rule is easy to get wrong and worth stating up front: **reduce before you
regrid, whenever the reduction is nonlinear.** Horizontal interpolation is a
weighted average, so it commutes with differences but not with variance, energy,
or anything else quadratic. Interpolating first blends neighbouring cells and
destroys the fine-scale structure those quantities are largely made of, biasing
them low. Residual variance is therefore computed on the model's native grid and
only the resulting variance map is regridded.

The bias is invisible when the observation grid is as fine as the model's, which
is what makes it dangerous: it hid in SST (quarter degree to quarter degree, a
0.18% error) while costing 6% in OHC (quarter to half degree), and it scales
with the resolution ratio -- so it distorts comparisons between models of
different resolution rather than applying a constant offset.
"""
