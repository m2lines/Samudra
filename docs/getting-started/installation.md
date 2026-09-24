<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Installation

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager

## Install from PyPI

Samudra is pure Python, so one wheel covers every platform. PyTorch dispatches
scaled dot product attention to an optimized CUDA kernel when supported:

```bash
# Install with `uv` (recommended)
uv add samudra                    # CPU (default)
uv add "samudra[cuda]"            # adds torchvision
uv add "samudra[tensorstore]"     # native Zarr v2 data reads
uv add samudra --prerelease=allow # latest nightly dev build
# Install with `pip`
pip install samudra               # CPU (default)
pip install "samudra[cuda]"       # adds torchvision
pip install "samudra[tensorstore]" # native Zarr v2 data reads
pip install --pre samudra         # latest nightly dev build
```

The `cuda` extra adds torchvision. Perceiver attention uses PyTorch's native
SDPA dispatcher and does not require a separately compiled attention package.
The CUDA kernels are compiled into the CUDA-enabled PyTorch distribution. On
supported NVIDIA hardware, `auto` selects a fused kernel when its dtype and
tensor shapes are eligible and otherwise falls back safely. Selecting `flash`
forces PyTorch FlashAttention and fails loudly when that kernel is unavailable.

The PhysicsNeMo container inherits its PyTorch and CUDA binaries from NVIDIA.
Its build-time smoke test verifies that this CUDA-enabled PyTorch reports
compiled FlashAttention support, while GPU CI forces that backend through a
forward and backward pass.

The `tensorstore` extra adds the optional native TensorStore backend for Zarr
v2 datasets. Set `data.xarray_backend: tensorstore` in a Samudra config to use
it. The default remains `zarr-python`.

## Development setup

To work on Samudra itself, clone the repository and install dependencies:

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
