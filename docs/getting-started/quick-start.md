<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Quick Start

## Run in Google Colab

The [Colab quickstart](https://colab.research.google.com/github/m2lines/Samudra/blob/main/notebooks/quickstart.ipynb)
installs the latest stable Samudra release from PyPI and walks through a
complete, YAML-configured training run on a free-tier GPU. It writes and
displays an editable configuration, trains and validates Samudra 2 on public
2° OM4 data with live batch progress, saves a checkpoint, and plots one
held-out sea-surface-height prediction.

PyPI installation, public S3 streaming, and a Colab GPU provide a
browser-based workflow from configuration through visualization. The same
editable YAML is a starting point for longer runs, more variables, and
higher-resolution experiments.

## Training a Model

Training is configured via YAML files. To launch a training run with the default Samudra configuration:

```bash
uv run samudra train samudra_om4/train.yaml
```

The samudra-multi model supports multi-scale training across different resolutions:

```bash
uv run samudra train samudra_multi_om4/train.yaml
```

### Validation and checkpoint selection

The `samudra_om4/train.yaml` preset runs a 360-day autoregressive validation
rollout every epoch, within the existing one-year validation split. With two
roughly five-day outputs per model call, this is about 35 calls. Validation
runs on rank 0 and streams one model call's targets at a time.

The best-validation checkpoint minimizes
`rollout_val/360d/normalized_rmse/channel_mean`: spatial, area-weighted RMSE
on wet cells in normalized units, averaged equally over prognostic channels
and forecast times. The square root is taken before averaging over channels
and time. Non-finite rollout scores cannot win checkpoint selection.
W&B receives this scalar alongside the existing physical-unit rollout metrics
and single-step validation metrics.

```yaml
rollout_validation:
  days: [360]
  frequency: 1
  steps_forward: 1
checkpoint_validation_metric: rollout_rmse
```

For multiple horizons, checkpoint selection uses the longest horizon. If
`frequency` is increased, only rollout epochs can replace the best-validation
checkpoint; latest and periodic checkpoints continue to be saved. A requested
day horizon must fit the configured validation split.

To use single-step checkpoint selection and disable rollout validation,
set `checkpoint_validation_metric: one_step_loss` and
`rollout_validation: null`. Rollout selection currently requires a single data
source. When resuming, changing the selection metric or configured horizon resets
the saved best score so incompatible scores are not compared. The checkpoint's
`best_val_loss` (and search summary's `best_validation_loss`) stores the selected
score; `validation_loss` continues to report single-step loss.

### End-of-training evaluation

The current `samudra_om4/train.yaml` preset evaluates the last saved epoch and
the final EMA checkpoint after training. It uses `samudra_om4/eval.yaml`, which
saves the long rollout and scores it against DUACS, OISST, and IAP observations,
alongside a date-matched OM4 baseline.

```yaml
post_train_eval:
  eval_config_path: samudra_om4/eval.yaml
  last_n_checkpoints: 1
```

Each checkpoint writes `predictions.zarr` and `observation_metrics.csv` under
the training output's `evals/<checkpoint>/` directory. `evals/summary.json`
contains both rollout and `obs/*` scalar metrics. When W&B is enabled in the
**evaluation config**, it also receives the scalars and `obs/metrics_table`.
The bundled evaluation config keeps W&B disabled; local metrics are still saved.

Observation scoring reads the public observation stores (roughly 65 GB) and
adds CPU time after the rollout. Budget for this phase in the job wall time.
You can point `observations` at local copies instead. Data or scoring errors
fail the evaluation rather than silently omitting metrics.

Set `post_train_eval: null` to skip the entire phase. To keep the rollout but
omit observations, use a custom evaluation config with `observations: null`.
The evaluation config must match any changes to the training model, input/output
steps, variables, or dataset paths; it inherits the training data root, not all
training overrides. Its inference period must be present in that dataset.

New prediction stores preserve dry cells as NaNs so observation scoring excludes
land and absent depth levels. Older stores written with zero-filled dry cells
need the original per-channel source masks applied before scoring; a zero value
alone cannot distinguish dry cells from valid ocean predictions.

### Data Paths

Training configs reference OM4 ocean model data stored in Zarr format. The bundled
`samudra_om4/train.yaml` includes `data/om4.yaml`; point it at your own data. In a
checkout, edit the preset in place:

```yaml
# src/samudra/configs/data/om4.yaml
data:
  path: "s3://<your-bucket>/path/to/OM4.zarr"  # Update with your data path
```

Without a checkout, copy a preset out and edit it, or override paths on the command
line (see below). See `src/samudra/configs/data/` for example data configurations at
1°, 1/2°, and 1/4° resolutions.

## Evaluation

Run a long autoregressive rollout against ground-truth data:

```bash
uv run samudra eval samudra_om4/eval.yaml
```

This produces metrics (RMSE, bias, anomaly correlation) and writes predicted fields to a Zarr output file.

## Visualization

Generate maps, time series, and probability density plots from evaluation outputs:

```bash
uv run samudra viz samudra_om4/viz.yaml
```

## Configuration

All commands accept `--help` for available options:

```bash
uv run samudra train --help
uv run samudra eval --help
uv run samudra viz --help
```

You can override any config key from the command line:

```bash
uv run samudra train samudra_om4/train.yaml --epochs 100 --lr 1e-4
```

See [Configuration](../config.md) for details on the configuration system.
