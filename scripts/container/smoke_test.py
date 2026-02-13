#!/usr/bin/env python3
"""Trivial runtime smoke test for the containerized project environment."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata

import torch
import zarr


def require_import(module_name: str) -> None:
    importlib.import_module(module_name)
    print(f"import {module_name}: OK")


def version(dist_name: str) -> str:
    return metadata.version(dist_name)


def main() -> int:
    require_import("ocean_emulators")
    require_import("ocean_emulators.models.samudra")
    require_import("flash_attn")
    require_import("flash_perceiver")
    require_import("zarr.testing")

    sample = torch.randn(2, 2)
    result = sample @ sample
    print(f"torch: {version('torch')}")
    print(f"torchvision: {version('torchvision')}")
    print(f"flash-attn: {version('flash-attn')}")
    print(f"flash-perceiver: {version('flash-perceiver')}")
    print(f"zarr: {zarr.__version__}")
    print(f"tensor-op: shape={tuple(result.shape)} dtype={result.dtype}")
    print("smoke-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
