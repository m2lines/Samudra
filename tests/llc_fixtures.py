# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import xarray as xr


def coarse_llc_dataset(*, n_time: int = 8, ny: int = 3, nx: int = 4) -> xr.Dataset:
    """Synthetic compact LLC data with the live store's dimensions and dtypes."""
    # Checked on EAI on 2026-11-19 against coarse_LLC4320/LLC4320_samudra_1deg.zarr.
    # Live shape is (10311 hours, 51 levels, 180 y, 360 x). Keep all 51 levels;
    # shorten time and space for tests, allowing a larger grid for U-Net pooling.
    levels = 51
    # Live lat/lon are float64 Gaussian latitudes and 0.5, 1.5, ..., 359.5 degrees.
    # These evenly spaced synthetic centers and bounds only preserve the layout.
    lat_b = np.linspace(-90, 90, ny + 1)
    lon_b = np.linspace(0, 360, nx + 1)
    lat = (lat_b[:-1] + lat_b[1:]) / 2
    lon = (lon_b[:-1] + lon_b[1:]) / 2
    # Live areas, fractions and bathymetry are spatially varying float32 arrays.
    # Use simple values with land, partial wetness, and a depth-limited column.
    wet_area = np.full((levels, ny, nx), 10.0, dtype=np.float32)
    wet_area[:, 0, 0] = 0
    wet_area[10:, 1, 2] = 0
    data = xr.Dataset(
        {
            "lat": ("y", lat),
            "lon": ("x", lon),
            "lat_b": ("y_b", lat_b),
            "lon_b": ("x_b", lon_b),
            # The live store also has these auxiliary float32 grid fields.
            "XC": (("y", "x"), np.broadcast_to(lon, (ny, nx)).astype(np.float32)),
            "YC": (
                ("y", "x"),
                np.broadcast_to(lat[:, None], (ny, nx)).astype(np.float32),
            ),
            "dxF": (("y", "x"), np.full((ny, nx), 4.0, dtype=np.float32)),
            "dyF": (("y", "x"), np.full((ny, nx), 3.0, dtype=np.float32)),
            "hFacC_2d": (("y", "x"), wet_area[0] / 12),
            "rA": (("y", "x"), np.full((ny, nx), 12.0, dtype=np.float32)),
            "wet_area": (("k", "y", "x"), wet_area),
            "land_frac": (("k", "y", "x"), 1 - wet_area / 12),
            "Depth": (("y", "x"), np.full((ny, nx), 1000.0, dtype=np.float32)),
            # Live int8 flags use -1=unwritten, 0=original, 1=repaired; here all 0.
            "repaired": (("time", "variable"), np.zeros((n_time, 9), dtype=np.int8)),
        },
        coords={
            # Live decoded times are hourly, 2011-09-13 00:00 to 2012-11-15 14:00.
            # As in the live store, time is the only dimension coordinate;
            # lat/lon remain data variables and k has no stored coordinate.
            "time": np.datetime64("2011-09-13T00:00:00", "ns")
            + np.arange(n_time) * np.timedelta64(1, "h")
        },
        attrs={
            # Retain a small metadata subset; omit producer paths/repair history.
            "velocity_frame": "east/north",
            # Live half-open index splits are [0, 8808] and [8808, 10311].
            "train_steps": [0, n_time - 2],
            "val_steps": [n_time - 2, n_time],
        },
    )
    # Live prognostics/forcings are float32, NaN on dry cells. Synthetic ramps
    # distinguish time, depth, location and variable without real ocean values.
    cube = np.arange(n_time * levels * ny * nx, dtype=np.float32).reshape(
        n_time, levels, ny, nx
    )
    for i, name in enumerate(("U", "V", "Theta", "Salt")):
        data[name] = (
            ("time", "k", "y", "x"),
            np.where(wet_area > 0, cube + i * 100, np.nan),
        )
        # Live statistics are float32 per k, wet-area weighted over 96 train hours.
        # These deterministic test statistics are not estimated from the ramps.
        data[f"{name}_mean"] = ("k", np.arange(levels, dtype=np.float32) / 10)
        data[f"{name}_std"] = ("k", 1 + np.arange(levels, dtype=np.float32) / 100)
    for i, name in enumerate(("Eta", "oceTAUX", "oceTAUY", "oceQnet", "oceFWflx")):
        data[name] = (
            ("time", "y", "x"),
            np.where(wet_area[0] > 0, cube[:, 0] + i * 10, np.nan),
        )
        # Surface statistics use the live singleton 'one' dimension, not scalars.
        data[f"{name}_mean"] = ("one", np.zeros(1, dtype=np.float32))
        data[f"{name}_std"] = ("one", np.ones(1, dtype=np.float32))
    return data


def write_coarse_llc_dataset(
    path: Path, *, n_time: int = 8, ny: int = 3, nx: int = 4
) -> xr.Dataset:
    data = coarse_llc_dataset(n_time=n_time, ny=ny, nx=nx)
    # Match live chunks: one hour with all levels and horizontal cells; static
    # fields/statistics each fit in one chunk. The test filename is data.zarr.
    # Use Xarray's default compression here; live arrays use Blosc Zstd level 3.
    encoding: dict[str, dict[str, object]] = {
        name: {"chunks": (1, *variable.shape[1:])}
        for name, variable in data.data_vars.items()
        if "time" in variable.dims
    }
    # Live time is int64 Gregorian seconds since 2011-09-10, in 1024-hour chunks.
    # The fixture uses the same encoding with one shorter coordinate chunk.
    encoding["time"] = {
        "dtype": "int64",
        "units": "seconds since 2011-09-10",
        "calendar": "gregorian",
        "chunks": (n_time,),
    }
    data.to_zarr(path / "data.zarr", encoding=encoding)
    return data
