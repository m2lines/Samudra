<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Configuration

## Overview

Configuration is defined by `config.py` and values are stored in YAML files within the `configs/`
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
like `--data '{"key": "value"}'` or a YAML file with a leading `@` symbol: `--data @configs/data/file.yaml`.

Training runs create a YAML file in the checkpoint directory with the final configuration used which
you can use to reproduce the run by passing to train e.g. `uv run -m samudra.train path/to/config.yaml`.

## API Reference

::: samudra.config

::: samudra.config_base
