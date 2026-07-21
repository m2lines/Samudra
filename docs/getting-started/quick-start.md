<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Quick Start

## Run in Google Colab

The [Colab quickstart](https://colab.research.google.com/github/m2lines/Samudra/blob/main/notebooks/quickstart.ipynb)
trains a Samudra 2 model on a public slice of 1° OM4 data using a free-tier
GPU runtime. It requires no local installation, HPC access, or data credentials.
The notebook is an onboarding smoke test rather than a scientifically useful
training run.

## Training a Model

Training is configured via YAML files. To launch a training run with the default Samudra configuration:

```bash
uv run -m samudra.train configs/samudra_om4/train.yaml
```

The samudra-multi model supports multi-scale training across different resolutions:

```bash
uv run -m samudra.train configs/samudra_multi_om4/train.yaml
```

### Data Paths

Training configs reference OM4 ocean model data stored in Zarr format. Update the data paths in your config to point to your data location:

```yaml
# configs/data/om4.yaml
data:
  path: "s3://<your-bucket>/path/to/OM4.zarr"  # Update with your data path
```

See `configs/data/` for example data configurations at 1°, 1/2°, and 1/4° resolutions.

## Evaluation

Run a long autoregressive rollout against ground-truth data:

```bash
uv run -m samudra.eval configs/samudra_om4/eval.yaml
```

This produces metrics (RMSE, bias, anomaly correlation) and writes predicted fields to a Zarr output file.

## Visualization

Generate maps, time series, and probability density plots from evaluation outputs:

```bash
uv run -m samudra.viz configs/samudra_om4/viz.yaml
```

## Configuration

All commands accept `--help` for available options:

```bash
uv run -m samudra.train --help
uv run -m samudra.eval --help
```

You can override any config key from the command line:

```bash
uv run -m samudra.train configs/samudra_om4/train.yaml --epochs 100 --lr 1e-4
```

See [Configuration](../config.md) for details on the configuration system.
