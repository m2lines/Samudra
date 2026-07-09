# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import xarray as xr

from samudra.constants import build_llc_spec


def raw_llc_datasets(n_time: int = 3) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    n_face = 2
    n_lev = len(build_llc_spec().depth_i_levels)
    n_j = 4
    n_i = 5
    times = np.arange(
        np.datetime64("2011-09-10T12:00:00"),
        np.datetime64("2011-09-10T12:00:00") + np.timedelta64(n_time, "D"),
        np.timedelta64(1, "D"),
        dtype="datetime64[ns]",
    )

    tracer = np.arange(
        n_time * n_face * n_lev * n_j * n_i, dtype=np.float32
    ).reshape(n_time, n_face, n_lev, n_j, n_i)
    surface = np.arange(n_time * n_face * n_j * n_i, dtype=np.float32).reshape(
        n_time, n_face, n_j, n_i
    )
    u = np.arange(n_time * n_face * n_lev * n_j * n_i, dtype=np.float32).reshape(
        n_time, n_face, n_lev, n_j, n_i
    )
    v = np.arange(n_time * n_face * n_lev * n_j * n_i, dtype=np.float32).reshape(
        n_time, n_face, n_lev, n_j, n_i
    )
    mask = np.ones((n_face, n_lev, n_j, n_i), dtype=bool)
    mask[:, :, 0, 0] = False

    data = xr.Dataset(
        {
            "Theta": (["time", "face", "k", "j", "i"], tracer),
            "Salt": (["time", "face", "k", "j", "i"], tracer + 100_000),
            "U": (["time", "face", "k", "j", "i_g"], u + 200_000),
            "V": (["time", "face", "k", "j_g", "i"], v + 300_000),
            "Eta": (["time", "face", "j", "i"], surface + 400_000),
            "oceQnet": (["time", "face", "j", "i"], surface + 500_000),
            "oceTAUX": (["time", "face", "j", "i_g"], surface + 600_000),
            "oceTAUY": (["time", "face", "j_g", "i"], surface + 700_000),
            "mask_c": (["face", "k", "j", "i"], mask),
            "mask_w": (["face", "k", "j", "i_g"], mask),
            "rAw": (["face", "j", "i_g"], mask[:, 0]),
        },
        coords={
            "time": times,
            "face": np.arange(n_face),
            "k": np.arange(n_lev),
            "j": np.arange(n_j),
            "i": np.arange(n_i),
            "j_g": np.arange(n_j),
            "i_g": np.arange(n_i),
        },
    )
    means = xr.Dataset(
        {
            **{
                f"{var}_lev_{i}": float(i)
                for var in ["Theta", "Salt", "U", "V"]
                for i in range(n_lev)
            },
            "Eta": 0.0,
            "oceQnet": 0.0,
            "oceTAUX": 0.0,
            "oceTAUY": 0.0,
        }
    )
    stds = xr.Dataset(
        {
            **{
                f"{var}_lev_{i}": 1.0
                for var in ["Theta", "Salt", "U", "V"]
                for i in range(n_lev)
            },
            "Eta": 1.0,
            "oceQnet": 1.0,
            "oceTAUX": 1.0,
            "oceTAUY": 1.0,
        }
    )
    return data, means, stds


def write_raw_llc_datasets(path: Path, *, n_time: int = 3) -> None:
    data, means, stds = raw_llc_datasets(n_time=n_time)
    data.to_zarr(path / "data.zarr")
    means.to_netcdf(path / "means.nc")
    stds.to_netcdf(path / "stds.nc")


def write_raw_llc_zarr_datasets(path: Path, *, n_time: int = 3) -> None:
    data, means, stds = raw_llc_datasets(n_time=n_time)
    data.to_zarr(path / "data.zarr")
    means.to_zarr(path / "means.zarr")
    stds.to_zarr(path / "stds.zarr")
