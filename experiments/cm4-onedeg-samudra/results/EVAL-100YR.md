<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Held-out 100-year evaluation

**Completed:** Beta job **214351** and all job steps exited `0:0` in **26m10s**.
[W&B run 8sui8ikd](https://wandb.ai/ocean_emulators/default/runs/8sui8ikd)
finished and synced five files and 234 media files. Across all three attempts,
allocation time was **30m22s**, or **2.0244 allocated GPU-hours**, below the
original two-hour/eight-GPU-hour cap.

The selected epoch-40 EMA model completed the continuous held-out century without
truth-state resets. Full read-back of all five saved physical fields passed in
199.2 seconds: **7,298 forecast dates**, from **0251-01-16 through 0351-01-01**,
matched the source exactly; depth coordinates, shapes, and float32 types matched;
all ocean values were finite and all dry cells were NaN. Final `time_bnds` has
shape `(7298, 2)`.

The normalized RMSE of the **time-mean states**, averaged over the 77 prognostic
channels, was **0.10901514**. This measures mean-state bias, not average
per-timestep forecast error, and is not comparable to the 360-day rollout score.
The existing aggregate including derived heat content remains unsuitable for
prognostic skill interpretation.

**Physical limitation:** finite forecasts over a century do not establish
physical realism. Salinity attained negative values and a broad range; temperature
also reached -4.31. These extrema alone do not establish their frequency,
location, timing, or change relative to the CM4 reference. Those require separate
diagnostics; no clipping or other corrections were applied.

| Saved field | Minimum | Maximum | Shape |
| --- | ---: | ---: | --- |
| Temperature (`thetao`) | -4.3130 | 34.7686 | `(7298, 19, 180, 360)` |
| Salinity (`so`) | -14.2768 | 69.2833 | `(7298, 19, 180, 360)` |
| Eastward velocity (`uo`) | -2.2374 | 2.2225 | `(7298, 19, 180, 360)` |
| Northward velocity (`vo`) | -2.0453 | 1.7369 | `(7298, 19, 180, 360)` |
| Sea-surface height (`zos`) | -1.9369 | 1.6613 | `(7298, 180, 360)` |

Physical fields are uncompressed, with 25-time/all-depth/spatial chunks. The
prediction store has **145,696,755,420 apparent bytes** (`du -sb`, including store
metadata and directory sizes). The final store and completion audit are retained
under the run root below, in
`output/cm4-100yr-best-epoch40-ema-attempt3/predictions.zarr` and the adjacent
`COMPLETE.json`.

## Attempts and verification

**Beta job 214347** was submitted and released on 2026-10-18. Its executed
source is [e5a9fd80a](https://github.com/m2lines/Samudra/commit/e5a9fd80a602b89fefe7e0836c7f285689c0dfd1).
That attempt failed at its first write after 2m31s: full-source `time_bnds` had
7,300 rows while the prediction batch had 50. EMA validation reproduced
0.2945153798867946 exactly, and W&B initialized
[run bktc4eon](https://wandb.ai/ocean_emulators/default/runs/bktc4eon), but no
prediction store was created. All first-attempt artifacts are retained.

The shared writer now selects time-dependent auxiliary coordinates at each
forecast batch's actual timestamps. [Code PR #935](https://github.com/m2lines/Samudra/pull/935) carries that fix; no
model/forcing/split changes are involved. Regression checks now include both
no-leap and Julian multi-append cases (18 focused tests passed). The replacement
uses separate attempt directories; failed artifacts are retained.

**Replacement job 214349** was submitted and released with executed source
[bc8845fc4](https://github.com/m2lines/Samudra/commit/bc8845fc46adda836f81637a8ad3de5735fb53ce).
It failed before forecasting after 1m41s: the validation score was
0.2945658350901645, 0.01713% above the stored score and beyond the initial
0.01% tolerance. The exported checkpoint SHA256 was identical to attempt 1
(`451bddb092737f1649c3afda640e0a79c408970527064ba2a368835d596ddf49`),
and every tensor had passed exact source/export checks. The numerical cause
was not established; no forecast or W&B rollout was started.

The third attempt retains the exact tensor checks and permits 0.1% relative
variation in the bfloat16 validation score. It uses `-attempt3` and a 1h50m limit,
so the two failed allocations plus this limit remain under the original
two-hour/eight-GPU-hour cap. Model weights, forcing, data, and forecast protocol
are unchanged. **Beta job 214351** was submitted and released with source
[ee03af282](https://github.com/m2lines/Samudra/commit/ee03af282).
Completion was verified against scheduler accounting, W&B sync logs, and the
saved-field audit; all attempt callbacks are retired.

Bring-up verified: the exact EMA export SHA256 again matches attempt 1, and the
validation recheck reproduced 0.2945153798867946 exactly. Online W&B is
[run 8sui8ikd](https://wandb.ai/ocean_emulators/default/runs/8sui8ikd).
After multiple appends, prediction metadata showed 200 forecast times with
matching `time_bnds` shape `(200, 2)`, `thetao` shape `(200, 19, 180, 360)`,
and `zos` shape `(200, 180, 360)`. Thus the first-write failure is resolved in
this real run; the final full-store audit also passed, as reported above.

## Scientific setup

The evaluation ran the selected epoch-40 EMA model continuously over the reserved CM4 control
period, **0251-01-06 through 0351-01-01**, using the same 1-degree grid, 19 depths,
77 prognostics, and `tauuo`, `tauvo`, `hfds`, `wfo` forcing as training. Two held-out states initialized the forecast; subsequent ocean states
were autoregressive predictions.
There were no periodic truth-state resets. It used the explicitly selected
full-record normalization. This is CM4-reference evaluation, with no observational scoring.

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
`/projects/ny/lz1955/multiscale/jrusak/runs/cm4-onedeg-samudra-20261017/eval-100yr-best-ema-20261018-attempt3`.
The run retains the exact source/config, selected EMA export, validation recheck,
prediction Zarr, W&B metrics/maps/time series, allocation, logs, and completion audit.
