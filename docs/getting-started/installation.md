# Installation

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Open-Athena/Ocean_Emulator.git
cd Ocean_Emulator
uv sync --dev
source .venv/bin/activate
```

## Verify Installation

Run the test suite to confirm everything is set up correctly:

```bash
uv run pytest -m "not manual and not cuda" -n auto
```
