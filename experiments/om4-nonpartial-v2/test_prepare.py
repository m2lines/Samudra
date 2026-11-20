# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression for consolidated Zarr metadata dropping symlinked field arrays."""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

spec = importlib.util.spec_from_file_location(
    "experiment_prepare", Path(__file__).with_name("prepare.py")
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_view_preserves_symlinked_fields_with_replacement_masks(tmp_path):
    source, masks, view = [tmp_path / name for name in ("source", "masks", "view")]
    original = xr.Dataset(
        {
            "thetao_0": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]]),
            "mask_0": (("y", "x"), np.ones((2, 2), dtype=bool)),
        },
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    strict = original[["mask_0"]].copy(deep=True)
    strict["mask_0"].values[0, 0] = False
    original.to_zarr(source, consolidated=True)
    strict.to_zarr(masks, consolidated=True)
    view.mkdir()
    for entry in source.iterdir():
        if entry.name == ".zmetadata":
            continue
        if entry.name == "mask_0":
            shutil.copytree(masks / entry.name, view / entry.name)
        elif entry.is_dir():
            (view / entry.name).symlink_to(entry, target_is_directory=True)
        else:
            shutil.copy2(entry, view / entry.name)
    module.write_view_metadata(source, masks, view)
    result = xr.open_zarr(view, consolidated=True)
    xr.testing.assert_equal(result.thetao_0, original.thetao_0)
    xr.testing.assert_equal(result.mask_0, strict.mask_0)
    assert (view / "thetao_0").is_symlink()
