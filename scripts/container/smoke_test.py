#!/usr/bin/env python3
"""Trivial runtime smoke test for the containerized project environment."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr  # type: ignore
from packaging.version import Version


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
    require_import("xarray")

    sample = torch.randn(2, 2)
    result = sample @ sample
    xarray_version = version("xarray")
    if Version(xarray_version) < Version("2026.2"):
        raise RuntimeError(
            f"xarray>=2026.2 is required for current GPU decode path; got {xarray_version}"
        )
    if not hasattr(zarr.config, "enable_gpu"):
        raise RuntimeError(
            "Installed zarr build is missing zarr.config.enable_gpu(); expected "
            "Open-Athena GPU-decode-capable zarr build."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        zarr_path = Path(tmp_dir) / "smoke.zarr"
        ds = xr.Dataset(
            {
                "temperature": (
                    ("time", "x"),
                    np.arange(6, dtype=np.float32).reshape(2, 3),
                )
            }
        )
        ds.to_zarr(zarr_path, mode="w", zarr_format=2)
        reopened = xr.open_dataset(
            zarr_path,
            engine="zarr",
            decode_cf=False,
            create_default_indexes=False,
        )
        if "temperature" not in reopened.data_vars:
            raise RuntimeError(
                "xarray/zarr smoke roundtrip failed to load data variables"
            )

    print(f"torch: {version('torch')}")
    print(f"torchvision: {version('torchvision')}")
    print(f"flash-attn: {version('flash-attn')}")
    print(f"flash-perceiver: {version('flash-perceiver')}")
    print(f"xarray: {xarray_version}")
    print(f"zarr: {zarr.__version__}")
    print(f"tensor-op: shape={tuple(result.shape)} dtype={result.dtype}")
    print("smoke-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
