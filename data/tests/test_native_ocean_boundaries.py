# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Check numerical boundary behavior in the shared OM4/CM4 ocean loader."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.simulation_preprocessing import gfdl_om4


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_native_ocean_wraps_x_and_extends_y(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    symmetric: bool,
    chunked: bool,
) -> None:
    # Nonconstant edge values distinguish wrapping from extension. Exercise
    # both raw staggering layouts and Dask chunks that split the periodic seam.
    ny, nx = 3, 4
    xh = np.arange(nx) + 0.5
    yh = np.arange(ny) - 1.0
    u = np.broadcast_to([1.0, 3.0, 5.0, 7.0], (1, 2, ny, nx)).copy()
    v = np.broadcast_to(np.array([10.0, 20.0, 30.0])[:, None], (1, 2, ny, nx)).copy()
    if symmetric:
        u = np.concatenate([u[..., -1:], u], axis=-1)
        v = np.concatenate([np.zeros_like(v[..., :1, :]), v], axis=-2)
    source = xr.Dataset(
        {
            "thetao": (("time", "lev", "yh", "xh"), np.ones((1, 2, ny, nx))),
            "so": (("time", "lev", "yh", "xh"), np.full((1, 2, ny, nx), 35.0)),
            "uo": (("time", "lev", "yh", "xq"), u),
            "vo": (("time", "lev", "yq", "xh"), v),
            "tauuo": (("time", "yh", "xq"), 2 * u[:, 0]),
            "tauvo": (("time", "yq", "xh"), 10 * v[:, 0]),
            "hfds": (("time", "yh", "xh"), np.ones((1, ny, nx))),
            "zos": (("time", "yh", "xh"), np.ones((1, ny, nx))),
        },
        coords={
            "time": [0],
            "lev": [2.5, 10.0],
            "ilev": [0.0, 5.0, 15.0],
            "xh": xh,
            "yh": yh,
            "xq": np.arange(nx + symmetric) + (0 if symmetric else 1),
            "yq": np.arange(ny + symmetric) - (1.5 if symmetric else 0.5),
        },
    )
    if chunked:
        source = source.chunk({"xh": 2, "xq": 2, "yh": 2, "yq": 2})
    lon, lat = np.meshgrid(xh, yh)
    static_path = tmp_path / "static.zarr"
    xr.Dataset(
        {
            "geolon": (("yh", "xh"), lon),
            "geolat": (("yh", "xh"), lat),
            "areacello": (("yh", "xh"), np.ones((ny, nx))),
        },
        coords={"xh": xh, "yh": yh},
    ).to_zarr(static_path)
    super_lon, super_lat = np.meshgrid(
        np.linspace(0, nx, 2 * nx + 1), np.linspace(-1.5, 1.5, 2 * ny + 1)
    )
    supergrid_path = tmp_path / "supergrid.zarr"
    xr.Dataset(
        {
            "x": (("nyp", "nxp"), super_lon),
            "y": (("nyp", "nxp"), super_lat),
            "angle_dx": (("nyp", "nxp"), np.zeros_like(super_lon)),
        }
    ).to_zarr(supergrid_path)
    # Only bypass the production schema's fixed 1080x1440 size requirement.
    monkeypatch.setattr(gfdl_om4, "ds_processed_validate", lambda ds: None)
    actual = gfdl_om4.native_ocean_preprocessing(
        source, str(static_path), str(supergrid_path)
    ).compute()

    expected_u = np.broadcast_to([4.0, 2.0, 4.0, 6.0], (1, 2, ny, nx))
    expected_v = np.broadcast_to(np.array([10.0, 15.0, 25.0])[:, None], (1, 2, ny, nx))
    np.testing.assert_array_equal(actual.uo, expected_u)
    np.testing.assert_array_equal(actual.tauuo, 2 * expected_u[:, 0])
    np.testing.assert_array_equal(actual.vo, expected_v)
    np.testing.assert_array_equal(actual.tauvo, 10 * expected_v[:, 0])
