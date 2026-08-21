#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Trivial runtime smoke test for the containerized project environment."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata

import torch
import torch.nn.functional as F
import zarr  # type: ignore


def require_import(module_name: str) -> None:
    importlib.import_module(module_name)
    print(f"import {module_name}: OK")


def version(dist_name: str) -> str:
    return metadata.version(dist_name)


def main() -> int:
    require_import("samudra")
    require_import("samudra.models.samudra")
    sample = torch.randn(2, 2)
    result = sample @ sample
    query = torch.randn(1, 1, 2, 8)
    attention = F.scaled_dot_product_attention(query, query, query)
    flash_attention_available = torch.backends.cuda.is_flash_attention_available()
    if torch.version.cuda is not None and not flash_attention_available:
        raise RuntimeError(
            "the container has a CUDA-enabled PyTorch build without FlashAttention"
        )
    print(f"torch: {version('torch')}")
    print(f"torchvision: {version('torchvision')}")
    print(f"torch-cuda: {torch.version.cuda}")
    print(f"torch-flash-attention: {flash_attention_available}")
    print(f"sdpa: shape={tuple(attention.shape)} dtype={attention.dtype}")
    print(f"zarr: {zarr.__version__}")
    print(f"tensor-op: shape={tuple(result.shape)} dtype={result.dtype}")
    print("smoke-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
