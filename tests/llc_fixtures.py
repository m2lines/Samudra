# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import xarray as xr


def coarse_llc_dataset(*, n_time: int = 8) -> xr.Dataset:
    """Small compact fixture matching LLC4320_samudra_1deg.zarr's schema."""
    levels, ny, nx = 51, 3, 4
    wet_area = np.full((levels, ny, nx), 10.0, dtype=np.float32)
    wet_area[:, 0, 0] = 0
    wet_area[10:, 1, 2] = 0
    data = xr.Dataset(
        {
            "lat": ("y", np.array([-60.0, 0.0, 60.0])),
            "lon": ("x", np.array([45.0, 135.0, 225.0, 315.0])),
            "rA": (("y", "x"), np.full((ny, nx), 12.0, dtype=np.float32)),
            "wet_area": (("k", "y", "x"), wet_area),
            "land_frac": (("k", "y", "x"), 1 - wet_area / 12),
            "Depth": (("y", "x"), np.full((ny, nx), 1000.0, dtype=np.float32)),
            "repaired": (("time", "variable"), np.zeros((n_time, 9), dtype=np.int8)),
        },
        coords={
            "time": np.datetime64("2011-09-13T00:00:00", "ns")
            + np.arange(n_time) * np.timedelta64(1, "h")
        },
        attrs={
            "velocity_frame": "east/north",
            "train_steps": [0, n_time - 2],
            "val_steps": [n_time - 2, n_time],
        },
    )
    cube = np.arange(n_time * levels * ny * nx, dtype=np.float32).reshape(
        n_time, levels, ny, nx
    )
    for i, name in enumerate(("U", "V", "Theta", "Salt")):
        data[name] = (
            ("time", "k", "y", "x"),
            np.where(wet_area > 0, cube + i * 100, np.nan),
        )
        data[f"{name}_mean"] = ("k", np.arange(levels, dtype=np.float32) / 10)
        data[f"{name}_std"] = ("k", 1 + np.arange(levels, dtype=np.float32) / 100)
    for i, name in enumerate(("Eta", "oceTAUX", "oceTAUY", "oceQnet", "oceFWflx")):
        data[name] = (
            ("time", "y", "x"),
            np.where(wet_area[0] > 0, cube[:, 0] + i * 10, np.nan),
        )
        data[f"{name}_mean"] = ("one", np.zeros(1, dtype=np.float32))
        data[f"{name}_std"] = ("one", np.ones(1, dtype=np.float32))
    return data


def write_coarse_llc_dataset(path: Path, *, n_time: int = 8) -> xr.Dataset:
    data = coarse_llc_dataset(n_time=n_time)
    encoding = {
        name: {"chunks": (1, *variable.shape[1:])}
        for name, variable in data.data_vars.items()
        if "time" in variable.dims
    }
    data.to_zarr(path / "data.zarr", encoding=encoding)
    return data
