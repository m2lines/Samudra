# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""CM4 time semantics and compact publication contract on small local arrays."""

import datetime
import json
from pathlib import Path
from typing import Any

import cftime
import numpy as np
import ocean_preprocessing.__main__ as main
import pytest
import xarray as xr
from ocean_preprocessing.simulation_preprocessing import gfdl_cm4, gfdl_om4

from tests import requires_xesmf


def _cm4_source() -> xr.Dataset:
    times = [
        cftime.DatetimeNoLeap(151, 1, 6) + datetime.timedelta(days=5 * i)
        for i in range(12)
    ]
    return xr.Dataset(
        {
            "wfo": ("time", np.ones(12)),
            "average_DT": ("time", np.full(12, 5), {"units": "days"}),
        },
        coords={"time": times},
    )


def _prepare_time(
    monkeypatch: pytest.MonkeyPatch, source: xr.Dataset, **kwargs: Any
) -> xr.Dataset:
    monkeypatch.setattr(gfdl_cm4.xr, "open_dataset", lambda *args, **kw: source)
    monkeypatch.setattr(
        gfdl_cm4, "native_ocean_preprocessing", lambda ds, *args, **kw: ds
    )
    return gfdl_cm4.cm4_ocean_preprocessing("source", "grid", "mosaic", **kwargs)


def test_cm4_recovers_interval_bounds_without_relabeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _cm4_source()
    out = _prepare_time(monkeypatch, source)
    xr.testing.assert_equal(out.time, source.time.assign_attrs(bounds="time_bnds"))
    assert out.sizes["time"] == 12
    assert out.time_bnds.values[0, 0] == cftime.DatetimeNoLeap(151, 1, 1)
    np.testing.assert_array_equal(out.time_bnds[:, 1], out.time)
    assert "wfo" in out


@pytest.mark.parametrize("invalid", ["calendar", "cadence", "duration", "wfo"])
def test_cm4_rejects_invalid_source(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    source = _cm4_source()
    if invalid == "calendar":
        source = source.assign_coords(time=xr.date_range("2000", periods=12, freq="5D"))
    elif invalid == "cadence":
        values = source.time.values.copy()
        values[1] += datetime.timedelta(days=1)
        source = source.assign_coords(time=values)
    elif invalid == "duration":
        source["average_DT"][0] = 4
    else:
        source = source.drop_vars("wfo")
    with pytest.raises(
        ValueError,
        match={
            "calendar": "no-leap",
            "cadence": "five days",
            "duration": "five days",
            "wfo": "wfo",
        }[invalid],
    ):
        _prepare_time(monkeypatch, source)


@requires_xesmf
@pytest.mark.parametrize("ny,nx", [(2, 4), (4, 8)])
def test_cm4_publication_rotates_stress_preserves_metadata_and_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ny: int, nx: int
) -> None:
    # Two small output grids exercise the same resolution-independent path as
    # the 1-degree and quarter-degree products, without processing real data.
    lon_b = np.linspace(0, 360, nx + 1)
    lat_b = np.linspace(-80, 80, ny + 1)
    lon, lat = np.meshgrid((lon_b[1:] + lon_b[:-1]) / 2, (lat_b[1:] + lat_b[:-1]) / 2)
    xb, yb = np.meshgrid(lon_b, lat_b)
    times = _cm4_source().time.values
    shape = (12, 2, ny, nx)
    source = xr.Dataset(
        {
            n: (
                ("time", "lev", "y", "x"),
                np.full(shape, v, dtype="float32"),
                {"units": unit},
            )
            for n, v, unit in [
                ("thetao", 10, "degC"),
                ("so", 35, "psu"),
                ("uo", 2, "m s-1"),
                ("vo", 3, "m s-1"),
            ]
        }
        | {
            n: (
                ("time", "y", "x"),
                np.full((12, ny, nx), v, dtype="float32"),
                {"units": unit},
            )
            for n, v, unit in [
                ("tauuo", 2, "N m-2"),
                ("tauvo", 3, "N m-2"),
                ("hfds", 10, "W m-2"),
                ("wfo", 0.001, "kg m-2 s-1"),
                ("zos", 0.1, "m"),
            ]
        },
        coords={
            "time": times,
            "lev": [2.5, 10.0],
            "ilev": [0.0, 5.0, 15.0],
            "dz": ("lev", [5.0, 10.0]),
            "x": lon[0],
            "y": lat[:, 0],
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
            "lon_b": (("y_b", "x_b"), xb),
            "lat_b": (("y_b", "x_b"), yb),
            "areacello": (("y", "x"), np.ones((ny, nx))),
            "wetmask": (("lev", "y", "x"), np.ones((2, ny, nx), dtype=bool)),
            "angle": (("y", "x"), np.full((ny, nx), 90.0)),
            "time_bnds": (
                ("time", "nv"),
                [[t - datetime.timedelta(days=5), t] for t in times],
            ),
        },
        attrs={
            "title": "test CM4 piControl",
            "time_sampling": "five-day mean, timestamp at interval end",
        },
    ).chunk({"time": 3, "lev": 1})
    source.time.attrs["bounds"] = "time_bnds"
    target = xr.Dataset(
        {
            "grid_lont": (("grid_yt", "grid_xt"), lon),
            "grid_latt": (("grid_yt", "grid_xt"), lat),
            "grid_lon": (("grid_y", "grid_x"), xb),
            "grid_lat": (("grid_y", "grid_x"), yb),
        }
    )
    grid_path = tmp_path / "grid.nc"
    target.to_netcdf(grid_path)
    native = source.drop_vars(
        ["lon", "lat", "lon_b", "lat_b", "areacello", "wetmask", "angle", "dz"]
    ).rename({"x": "xh", "y": "yh", "lev": "z_l", "ilev": "z_i"})
    for name in ("uo", "tauuo", "vo", "tauvo"):
        field = native[name]
        dims = tuple(
            ("xq" if dim == "xh" else dim)
            if name in ("uo", "tauuo")
            else ("yq" if dim == "yh" else dim)
            for dim in field.dims
        )
        native[name] = xr.DataArray(field.data, dims=dims, attrs=field.attrs)
    native = native.assign_coords(xq=lon[0], yq=lat[:, 0])
    bounds_args = []
    if nx == 4:
        bounds_path = tmp_path / "time_bnds.nc"
        # Explicit integer days also round-trip with older xarray's NetCDF3 writer.
        native[["time_bnds"]].to_netcdf(
            bounds_path,
            engine="scipy",
            encoding={
                name: {
                    "units": "days since 0151-01-01",
                    "calendar": "noleap",
                    "dtype": "int32",
                }
                for name in ("time", "time_bnds")
            },
        )
        native = native.drop_vars("time_bnds")
        bounds_args = [f"--time_bounds_path={bounds_path}"]
    native_path = tmp_path / "raw.zarr"
    native.to_zarr(native_path, consolidated=True)
    static_path = tmp_path / "native.nc"
    xr.Dataset(
        {
            "geolon": (("yh", "xh"), lon),
            "geolat": (("yh", "xh"), lat),
            "areacello": (("yh", "xh"), np.ones((ny, nx))),
        },
        coords={"xh": lon[0], "yh": lat[:, 0]},
    ).to_netcdf(static_path)
    super_path = tmp_path / "supergrid.nc"
    super_lon, super_lat = np.meshgrid(
        np.linspace(0, 360, 2 * nx + 1), np.linspace(-80, 80, 2 * ny + 1)
    )
    xr.Dataset(
        {
            "x": (("nyp", "nxp"), super_lon),
            "y": (("nyp", "nxp"), super_lat),
            "angle_dx": (("nyp", "nxp"), np.full(super_lon.shape, 90.0)),
        }
    ).to_netcdf(super_path)
    # The production native-grid schema fixes 1080x1440; exercise the actual
    # interpolation and geometry code on this small grid without that size check.
    monkeypatch.setattr(gfdl_om4, "ds_processed_validate", lambda ds: None)
    monkeypatch.setattr(
        main,
        "spatial_filter_by_grid",
        lambda *args: pytest.fail("Unfiltered CM4 must not call the filter"),
    )
    out = tmp_path / "CM4.zarr"
    main.fire.Fire(
        main.CLI,
        command=[
            "cm4",
            str(native_path),
            str(static_path),
            str(super_path),
            str(grid_path),
            f"--output_path={out}",
            "--skip_spatial_filtering",
            "--skip_flattening",
            "--skip_validation",
            *bounds_args,
        ],
    )
    actual = xr.open_zarr(out, chunks=None)
    assert actual.sizes["time"] == 12  # no historical 7,302-state truncation
    assert set(actual.data_vars) == {
        "thetao",
        "so",
        "uo",
        "vo",
        "zos",
        "tauuo",
        "tauvo",
        "hfds",
        "wfo",
    }
    np.testing.assert_allclose(actual.tauuo, -3)
    np.testing.assert_allclose(actual.tauvo, 2)
    np.testing.assert_allclose(actual.uo, -3)
    assert actual.wfo.attrs["units"] == "kg m-2 s-1"
    assert actual.tauuo.attrs["standard_name"] == "surface_downward_eastward_stress"
    assert actual.attrs["title"] == "test CM4 piControl"
    assert actual.attrs["m2lines/preprocessing"]["spatial_filter"] is None
    assert actual.attrs["m2lines/preprocessing"]["source_coverage"]["steps"] == 12
    assert actual.time.encoding["calendar"] == "noleap"
    np.testing.assert_array_equal(actual.time, times)
    np.testing.assert_array_equal(actual.time_bnds[:, 1], times)
    assert {
        "areacello",
        "dz",
        "lev",
        "lat",
        "lon",
        "lat_b",
        "lon_b",
        "ocean_fraction",
        "wetmask",
        "time_bnds",
    } <= set(actual.coords)
    for name in ["thetao", "so", "uo", "vo"]:
        metadata = json.loads((out / name / ".zarray").read_text())
        assert metadata["chunks"] == [1, 2, ny, nx]
    assert json.loads((out / "wfo" / ".zarray").read_text())["chunks"] == [1, ny, nx]
