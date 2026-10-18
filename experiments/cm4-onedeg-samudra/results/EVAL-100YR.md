<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Held-out 100-year evaluation

**Beta job 214347** was submitted and released on 2026-10-18. Its executed
source is [e5a9fd80a](https://github.com/m2lines/Samudra/commit/e5a9fd80a602b89fefe7e0836c7f285689c0dfd1).
That attempt failed at its first write after 2m31s: full-source `time_bnds` had
7,300 rows while the prediction batch had 50. EMA validation reproduced
0.2945153798867946 exactly, and W&B initialized
[run bktc4eon](https://wandb.ai/ocean_emulators/default/runs/bktc4eon), but no
prediction store was created. All first-attempt artifacts are retained.

The shared writer now selects time-dependent auxiliary coordinates at each
forecast batch's actual timestamps. A separate code PR carries that fix; no
model/forcing/split changes are involved. Regression checks now include both
no-leap and Julian multi-append cases (18 focused tests passed). The replacement
uses a separate `-attempt2` directory and a 1h55m cap, keeping both allocations
within the original two-hour/eight-GPU-hour maximum.

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
`/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017/eval-100yr-best-ema-20261018-attempt2`.
The run retains the exact source/config, selected EMA export, validation recheck,
prediction Zarr, W&B metrics/maps/time series, allocation, logs, and completion audit.
