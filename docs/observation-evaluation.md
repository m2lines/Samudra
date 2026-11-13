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
the metrics CSV; running the scoring command first is optional.

The coverage section records remaining differences from the document; neither
command exports the entire specification as one machine-readable metric table.

## Inputs and scope

Use an existing multi-year `predictions.zarr` from evaluation with `save_zarr: true`.
No model checkpoint or GPU is needed for the commands below. Training alone does
not produce this store: first [run evaluation](run.md#evaluating-the-model), or
configure a post-training checkpoint sweep.

Use matching data, grid, variables, forecast dates, and observation settings.
The examples use the bundled OM4 presets and their public DUACS, OISST, and IAP
products. For other data, make matching evaluation and visualization configs.
The observation path currently requires a rectilinear grid.

New evaluation stores preserve dry cells as NaNs. The default OM4 input range is
2014-10-05 through 2022-12-29, producing forecasts from 2014-10-15 through
2022-12-29 with two input and two output pentads per model call. Both endpoints
are chosen so the final output block is complete; changing only the old end
date would leave an incomplete block and still omit December 29.

## Scores without figures

Use the evaluation config matching the saved rollout, with its `observations`
block enabled:

```bash
uv run samudra metrics path/to/eval.yaml \
  --experiment.data_root /path/to/om4/data \
  --experiment.base_output_dir /path/to/training-output/evals \
  --experiment.name epoch_0070 \
  --data.loading.type cpu
```

This reads `/path/to/training-output/evals/epoch_0070/predictions.zarr` and writes
`observation_metrics.csv` alongside it. Repeat with `ema_latest` for that output.

The CSV includes five primary RMSEs, their annual detail and uncertainty,
four residual-variance map scores, coverage/bathymetry diagnostics, and nine regional pooled spectral log10-RMSE
scores per model (three diagnostic families × three regions).
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
  --groundtruth_time_range.start 2014-10-15 \
  --groundtruth_time_range.end 2022-12-29 \
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
log10-RMSE scores appear in pooled-spectrum legends and the CSV, with explicit
region, units (dex), and scoring period. Evaluation also exports these scalar
keys to W&B when enabled. Unavailable comparisons are NaN, never zero.
Spatial regions are North Pacific, Gulf Stream, and Agulhas; temporal regions
are Global, Gulf Stream, and Agulhas. A region that is too small or not fully
observed yields no spectrum; inspect the plots and logs rather than treating
an unavailable comparison as zero error.

Visualization honors `observations.baselines`, just like scoring. With the
bundled `baselines: [om4]`, the matching OM4 ground truth is included automatically
in the figures and CSV over the forecast timestamps. Do not add an `om4` entry
to `runs` when this baseline is enabled; that name is reserved.

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
| 2022 snapshots | Surface annual CSV rows include 2022 when coverage is complete. A separate 2022 map set is not emitted by the default suite. The default rollout includes the final December pentad, allowing OHC 2022 to be scored. |
| Pooled and interannual EKE/SSTA spatial and KE temporal spectra | Six PDFs plus regional pooled log-spectrum errors in CSV and W&B |
| SST/EKE/upper-700 m OHC trends and residual time-series variance | Figures and annotations; no corresponding CSV rows |
| SST and upper-700 m OHC residual-variance map RMSE and pattern correlation | Full-overlap CSV scores and maps; the document's annual residual-variance RMS and bootstrap intervals are not implemented |
| Mean EKE map, monthly total OHC, annual OHC and OHC-bias maps | Not emitted by these five observation steps. The OHC time series is an area-weighted mean in J/m², not integrated total heat in J. |

OHC uses exact layer overlaps for [0, 700) m and [700, 2000) m with the document's
ρ=1035 kg/m³ and Cp=3850 J/(kg K). With the default rollout through December 29,
primary surface and OHC scores cover 2015–2022 when product coverage is complete. Inspect the CSV's
period and paired-cell coverage before comparing runs.

SSTA spectra use the shared pentad seasonal-removal kernel, rather than literal
calendar-day groups in the reference, to avoid leap-year sampling artifacts.
EKE and residual variance are computed on native grids before interpolation.
These choices and their effects were reviewed in #829 and #834. Headline scores
still use pairwise finite masks, which can differ across models; they are not
unconditionally comparable across resolutions.
