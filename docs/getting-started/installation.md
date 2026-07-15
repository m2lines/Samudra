<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Installation

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/m2lines/Samudra.git
cd Samudra
uv sync --dev
source .venv/bin/activate
```

## Verify Installation

Print the training CLI help to confirm everything is set up correctly:

```bash
uv run -m samudra.train --help
```
