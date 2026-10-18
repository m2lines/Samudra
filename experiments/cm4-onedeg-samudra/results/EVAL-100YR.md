<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Held-out 100-year evaluation

Run the selected epoch-40 EMA model continuously over the reserved CM4 control
period, **0251-01-06 through 0351-01-01**, using the same 1-degree grid, 19 depths,
77 prognostics, and `tauuo`, `tauvo`, `hfds`, `wfo` forcing as training. Initialize
with two held-out states; subsequent ocean states are autoregressive predictions.
There are no periodic truth-state resets. Reuse the explicitly selected full-record
normalization. This is CM4-reference evaluation, with no observational scoring.

The input period contains 7,300 five-day records. The first two states provide
context; the remaining **7,298 predictions**, from **0251-01-16 through 0351-01-01**,
are produced by **3,649 two-output model calls**. Chunking into groups of 25 calls
only bounds memory and write size; model state carries between chunks. The standard
rollout reader uses Xarray/Dask for inference, even with Rust configured for source
construction and training.

The best-validation checkpoint contains both raw weights and EMA state.
`prepare_eval_ema.py` exports the EMA parameters using the project's `EMATracker`,
checks every parameter and exported tensor, records checkpoint hashes, and
reproduces the selected 360-day validation score before evaluating held-out data.
The final epoch-70 EMA checkpoint is not used for this run.

The full prediction store is retained in project storage, approximately 146 GB
uncompressed. The job requests one production Beta node (four GPUs, as required by
Beta policy) for at most **two hours**, or eight allocated GPU-hours. The continuous
rollout is sequential on one GPU; it is not split into four independent forecasts.

`run_eval_100yr.py` uses the standard evaluator and adds a full saved-field read-back
audit: exact no-leap target times, shapes, float32 physical fields, finite wet cells,
and NaN dry cells. It also records a correctly labeled prognostic-only mean of the
normalized **time-mean-state** RMSEs. Existing all-field aggregates are preserved but
remain affected by the derived-heat-content issue documented in the training report;
they must not be used as prognostic skill scores. The added metric describes mean
state error, not the average error at each forecast time.

Run root:
`/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017/eval-100yr-best-ema-20261018`.
The run retains the exact source/config, selected EMA export, validation recheck,
prediction Zarr, W&B metrics/maps/time series, allocation, logs, and completion audit.
