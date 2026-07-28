<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Quick Start

## Training a Model

Training is configured via YAML files. To launch a training run with the default Samudra configuration:

```bash
uv run samudra train samudra_om4/train.yaml
```

The samudra-multi model supports multi-scale training across different resolutions:

```bash
uv run samudra train samudra_multi_om4/train.yaml
```

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
