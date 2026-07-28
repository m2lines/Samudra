<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Configuration

## Overview

Configuration is defined by `config.py` and values are stored in YAML files within the `src/samudra/configs/`
directory. Configuration files can include other configuration files using the `!include` directive.

Each configuration file is associated with a Pydantic model — you can generate JSON schemas
for them with `uv run src/samudra/config_schema.py` (which is run automatically in pre-commit).
To associate a configuration file with a Pydantic model, generate the JSON schema (if it doesn't
already exist) and then add this line to the top of the config file:

```yaml
# yaml-language-server: $schema=path/to/schema.json
```

This is what the `config_schema.py` script uses to determine which model to validate against,
and also enables autocomplete/type checking in VS Code via the [YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml).

## Command Line Configuration

The train and eval modules accept the configuration file as a positional argument.
You can override arbitrary keys on the command line — see `--help` for details. When overriding
an object (as opposed to a single scalar value) via the command line, you can either supply JSON
like `--data '{"key": "value"}'` or a YAML file with a leading `@` symbol: `--data @src/samudra/configs/data/file.yaml`.

Training runs create a YAML file in the checkpoint directory with the final configuration used which
you can use to reproduce the run by passing to train e.g. `uv run -m samudra.train path/to/config.yaml`.

## Data locations

Each data source names three stores — `data_location`, `data_means_location`, and
`data_stds_location`. Every one of them is a *location*, which can be written two ways:

- **A relative string** (e.g. `OM4.zarr`). It is resolved against `experiment.data_root`,
  so the same data config works against a local copy or a bucket just by changing the root:
  `--experiment.data_root /scratch/om4` or a structured root in the config.
- **A structured, absolute location** with an explicit `type`. This ignores `data_root`
  and points at exactly one place. Two types are supported:

```yaml
# Read a Zarr store directly from S3 (or an S3-compatible endpoint like OSN).
data_location:
  type: s3
  endpoint_url: "https://nyu1.osn.mghpcc.org"  # omit for AWS S3
  anon: true                                   # read a public bucket without credentials
  bucket: m2lines-pubs
  path: Samudra/v2026-07/om4_twodeg/OM4.zarr

# Read a Zarr/NetCDF store from an absolute local path.
data_location:
  type: local
  path: /scratch/om4/OM4.zarr
```

For a signed request (`anon: false`, the default) credentials come from the environment
(the usual `AWS_*` variables); the config never carries secrets. Set `anon: true` to read a
public bucket — such as the open datasets on OSN — with no credentials at all, which also
avoids a stale `AWS_ACCESS_KEY_ID` being rejected by a non-AWS endpoint.

`src/samudra/configs/data/om4-demo.yaml` is a ready-made source that streams the public 2°
OM4 dataset from OSN over S3 with no local download and no credentials. Because its
locations are absolute, `experiment.data_root` is ignored during resolution but must still
be set to some value (e.g. `--experiment.data_root .`).

## API Reference

::: samudra.config

::: samudra.config_base
