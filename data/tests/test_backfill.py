# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.backfill_grid_coords import BACKFILL_COORDS, apply_grid_coords


def _published_like_store(path, nx=4, ny=3, nlev=2, nt=2):
    """A minimal stand-in for a published OM4.zarr: time/x/y dims, split vars."""
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            f"thetao_{i}": (["time", "y", "x"], rng.random((nt, ny, nx)))
            for i in range(nlev)
        }
        | {
            f"mask_{i}": (["time", "y", "x"], np.ones((nt, ny, nx)))
            for i in range(nlev)
        }
        | {"zos": (["time", "y", "x"], rng.random((nt, ny, nx)))},
        coords={
            "time": np.arange(nt),
            "x": np.arange(nx, dtype="float64"),
            "y": np.arange(ny, dtype="float64"),
        },
    )
    ds.to_zarr(path, mode="w", consolidated=True)
    return ds


def _grid_coords(nx=4, ny=3, nlev=2):
    """A precomputed grid-coords dataset aligned to the store's x/y."""
    return xr.Dataset(
        coords={
            "x": np.arange(nx, dtype="float64"),
            "y": np.arange(ny, dtype="float64"),
            "lev": ("lev", np.arange(nlev, dtype="float64")),
            "dz": ("lev", np.ones(nlev)),
            "lon": (["y", "x"], np.zeros((ny, nx))),
            "lat": (["y", "x"], np.zeros((ny, nx))),
            "lon_b": (["y_b", "x_b"], np.zeros((ny + 1, nx + 1))),
            "lat_b": (["y_b", "x_b"], np.zeros((ny + 1, nx + 1))),
            "areacello": (["y", "x"], np.ones((ny, nx))),
            "ocean_fraction": (["lev", "y", "x"], np.ones((nlev, ny, nx))),
        }
    )


def test_apply_grid_coords_is_additive_and_promotes(tmp_path):
    """Backfill adds grid coords without disturbing existing prognostic chunks."""
    store = str(tmp_path / "om4.zarr")
    before = _published_like_store(store)

    apply_grid_coords(store, _grid_coords(), dry_run=False)

    after = xr.open_zarr(store)
    # Every backfill array is a coordinate (not a data variable).
    assert set(BACKFILL_COORDS) <= set(after.coords)
    # Existing data variables are untouched: same set, byte-identical values.
    assert set(after.data_vars) == set(before.data_vars)
    for v in before.data_vars:
        np.testing.assert_array_equal(after[v].values, before[v].values)
    # A per-level coord keeps its depth dimension.
    assert after["ocean_fraction"].dims == ("lev", "y", "x")


def test_apply_grid_coords_dry_run_writes_nothing(tmp_path):
    store = str(tmp_path / "om4.zarr")
    _published_like_store(store)

    apply_grid_coords(store, _grid_coords(), dry_run=True)

    after = xr.open_zarr(store)
    assert not (set(BACKFILL_COORDS) & set(after.variables))


def test_apply_grid_coords_refuses_existing_without_overwrite(tmp_path):
    store = str(tmp_path / "om4.zarr")
    _published_like_store(store)
    apply_grid_coords(store, _grid_coords(), dry_run=False)

    with pytest.raises(ValueError, match="already has coords"):
        apply_grid_coords(store, _grid_coords(), dry_run=False)


def test_apply_grid_coords_rejects_misaligned_grid(tmp_path):
    """A grid whose x/y don't match the store must be refused, not written blindly."""
    store = str(tmp_path / "om4.zarr")
    _published_like_store(store, nx=4)

    shifted = _grid_coords(nx=4)
    shifted = shifted.assign_coords(x=shifted["x"] + 100.0)

    with pytest.raises(ValueError, match="coordinate values differ"):
        apply_grid_coords(store, shifted, dry_run=False)
