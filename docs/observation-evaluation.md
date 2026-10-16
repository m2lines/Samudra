<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Observation evaluation: scores, maps, and spectra

The [Observation Evaluation Summary](https://docs.google.com/document/d/1cWzIrcajfW-gvoMzV3PU0JDAouVt5xerRmEiBIUbV1s/edit)
defines more than the nine headline scores in `observation_metrics.csv`.
The current implementation has two entry points: scoring and visualization.
**Run the five observation visualization steps below to obtain all currently
implemented observation diagnostics, including spectra.** This also computes
the headline CSV; running the scoring command first is optional.

This guide describes the implementation audited on 2026-10-08. The coverage
section records where it differs from the document; neither command currently
exports the entire specification as one machine-readable metric table.

## Inputs and scope

Use an existing multi-year `predictions.zarr` from evaluation with `save_zarr: true`.
No model checkpoint or GPU is needed for the commands below. Training alone does
not produce this store: first [run evaluation](run.md#evaluating-the-model), or
configure a post-training checkpoint sweep.

Use matching data, grid, variables, forecast dates, and observation settings.
The examples use the bundled OM4 presets and their public DUACS, OISST, and IAP
products. For other data, make matching evaluation and visualization configs.
The observation path currently requires a rectilinear grid.

Dry cells must be missing values, not finite zeros. The writer fix in
[PR #917](https://github.com/m2lines/Samudra/pull/917) preserves NaNs in new stores.
For older zero-filled outputs, restore the original per-channel wet masks before
either scoring or visualization. Do not infer land from zero values: valid ocean
predictions can be zero. The observation steps read the original run data and do
not inherit the masking performed for legacy visualization steps.

## Headline scores only

Use the evaluation config matching the saved rollout, with its `observations`
block enabled:

```bash
uv run python -m samudra.metrics path/to/eval.yaml \
  --experiment.data_root /path/to/om4/data \
  --experiment.base_output_dir /path/to/training-output/evals \
  --experiment.name epoch_0070 \
  --data.loading.type cpu
```

This reads `/path/to/training-output/evals/epoch_0070/predictions.zarr` and writes
`observation_metrics.csv` alongside it. Repeat with `ema_latest` for that output.
The module command also works on versions predating the `samudra metrics`
shortcut added by #917.

The CSV includes five primary RMSEs, their annual detail and uncertainty,
four residual-variance map scores, and coverage/bathymetry diagnostics.
The configured OM4 baseline is scored alongside the model. This CLI logs locally;
it does not upload to W&B. Standalone evaluation's `Eval.run()` can log these
scores to W&B when its evaluation config enables W&B.

## Scores and all implemented observation figures

From a repository checkout, run:

```bash
uv run samudra viz samudra_om4/viz.yaml \
  --data_root /path/to/om4/data \
  --base_output_dir /path/to/reports \
  --name observation-suite \
  --groundtruth_time_range.start 2014-10-20 \
  --groundtruth_time_range.end 2022-12-24 \
  --runs '[{"name":"last_checkpoint","location":"/path/to/training-output/evals/epoch_0070/predictions.zarr"},{"name":"ema","location":"/path/to/training-output/evals/ema_latest/predictions.zarr"}]' \
  --steps '["obs_rmse_maps","obs_annual_rmse","obs_variance_maps","obs_timeseries","obs_spectra"]'
```

Set the ground-truth range to the **saved forecast timestamps**, not the initial
input dates. All listed runs must match the ground-truth time length and grid
for the current visualization setup. Relative run locations resolve under
`data_root`; absolute paths avoid ambiguity. The bundled preset includes the
observation products and basin-mask location; visualization still prepares OM4
ground truth and profiles even when only observation steps are selected.

Results go to `/path/to/reports/observation-suite/Figures_observations/`:

| Step | Output |
| --- | --- |
| `obs_rmse_maps` | Five RMSE maps: velocity vector, instantaneous EKE, SST, and two OHC layers |
| `obs_annual_rmse` | One annual-RMSE figure with uncertainty |
| `obs_variance_maps` | Two residual-variance map figures: SST and upper-700 m OHC |
| `obs_timeseries` | Six figures: raw and detrended/deseasonalized global series for SST, EKE, and upper-700 m OHC; trend and residual variance are annotations |
| `obs_spectra` | Six figures: EKE and SSTA spatial spectra, KE temporal spectra, and each one's interannual bands |

There are normally 20 PDFs plus `observation_metrics.csv`. The CSV is recomputed
from the listed runs, not read from earlier evaluation output. Spectral
log10-RMSE scores appear in pooled-spectrum legends, **not in this CSV or W&B**.
Spatial regions are North Pacific, Gulf Stream, and Agulhas; temporal regions
are Global, Gulf Stream, and Agulhas. A region that is too small or not fully
observed yields no spectrum; inspect the plots and logs rather than treating
an unavailable comparison as zero error.

Visualization ignores `observations.baselines`: only entries in `runs` appear
in its figures and CSV. To include OM4, add a run pointing to an OM4 store sliced
to the same forecast timestamps and grid. Do not point that run directly at the
full historical store: visualization requires its time length to match the
forecast. The scoring-only command automatically selects OM4 through its data
source instead.

These commands can run locally or in a separate CPU allocation after the GPU
rollout finishes. Observation products total roughly 65 GB; allow for network
I/O and memory, or configure local copies. Visualization does additional
reductions and profile preparation, so headline-scoring runtime is not a timing
estimate for the full suite. Neither command submits a CPU job automatically.

## Coverage of the metrics document

Read the document's updated primary-score definitions together with its review
comments: it still contains older 2022-only tables and wording. The current
implementation deliberately differs from some reference conventions.

| Requested diagnostic | Current coverage |
| --- | --- |
| Primary velocity, instantaneous EKE, SST, and exact-layer OHC RMSE | CSV and RMSE maps; equal-year annual MSE aggregation, annual standard deviation, and calendar-year bootstrap intervals |
| 2022 snapshots | Surface annual CSV rows include 2022 when coverage is complete. A separate 2022 map set is not emitted by the default suite. OHC 2022 is excluded for rollouts ending December 24 because the final month is incomplete. |
| Pooled and interannual EKE/SSTA spatial and KE temporal spectra | Six PDFs; pooled log-spectrum errors remain figure annotations |
| SST/EKE/upper-700 m OHC trends and residual time-series variance | Figures and annotations; no corresponding CSV rows |
| SST and upper-700 m OHC residual-variance map RMSE and pattern correlation | Full-overlap CSV scores and maps; the document's annual residual-variance RMS and bootstrap intervals are not implemented |
| Mean EKE map, monthly total OHC, annual OHC and OHC-bias maps | Not emitted by these five observation steps. The OHC time series is an area-weighted mean in J/m², not integrated total heat in J. |

OHC uses exact layer overlaps for [0, 700) m and [700, 2000) m with the document's
ρ=1035 kg/m³ and Cp=3850 J/(kg K). For the usual rollout ending December 24,
primary surface scores cover 2015–2022 but OHC covers 2015–2021. Inspect the CSV's
period and paired-cell coverage before comparing runs.

SSTA spectra use the shared pentad seasonal-removal kernel, rather than literal
calendar-day groups in the reference, to avoid leap-year sampling artifacts.
EKE and residual variance are computed on native grids before interpolation.
These choices and their effects were reviewed in #829 and #834. Headline scores
still use pairwise finite masks, which can differ across models; they are not
unconditionally comparable across resolutions.

## Why there are two entry points

[PR #829](https://github.com/m2lines/Samudra/pull/829) implemented the nine headline
scores and the metrics-only rerun command. It explicitly left figures to a
follow-up. These metrics operate on a completed rollout rather than incremental
training aggregators.

[PR #834](https://github.com/m2lines/Samudra/pull/834) added the five visualization
steps and spectral kernels, sharing field preparation and metric kernels with
the scorer. Its stated goals included keeping figure files out of W&B. Its
**Known gaps** section explicitly deferred spectral rows in CSV/W&B and
chaining visualization after evaluation. The
[review discussion](https://github.com/m2lines/Samudra/pull/834#issuecomment-5244215512)
also explains the separation between pure kernels, product-aware comparison
builders, and reporting/visualization.

The missing scalar exports are an unfinished reporting integration, not a
scientific reason to exclude spectra from evaluation. A unified machine-readable
suite needs shared computation of spectral curves/scores and time-series
statistics, CSV coverage for those results, and the missing document diagnostics.
A CPU command could then select scores, figures, or both from that same result.
The existing two commands do not yet provide that complete contract.
