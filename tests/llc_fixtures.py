# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import xarray as xr

from samudra.constants import build_llc_spec


def raw_llc_datasets() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    n_time = 3
    n_face = 2
    n_lev = len(build_llc_spec().depth_i_levels)
    n_j = 4
    n_i = 5
    times = np.array(
        [
            "2011-09-10T12:00:00",
            "2011-09-11T12:00:00",
            "2011-09-12T12:00:00",
        ],
        dtype="datetime64[ns]",
    )

    theta = np.arange(n_time * n_face * n_lev * n_j * n_i, dtype=np.float32).reshape(
        n_time, n_face, n_lev, n_j, n_i
    )
    qnet = np.arange(n_time * n_face * n_j * n_i, dtype=np.float32).reshape(
        n_time, n_face, n_j, n_i
    )
    mask = np.ones((n_face, n_lev, n_j, n_i), dtype=bool)
    mask[:, :, 0, 0] = False

    data = xr.Dataset(
        {
            "Theta": (["time", "face", "k", "j", "i"], theta),
            "oceQnet": (["time", "face", "j", "i"], qnet),
            "mask_c": (["face", "k", "j", "i"], mask),
            "U": (["time", "face", "k", "j", "i_g"], theta + 10_000),
            "V": (["time", "face", "k", "j_g", "i"], theta + 20_000),
            "oceTAUX": (["time", "face", "j", "i_g"], qnet + 30_000),
            "oceTAUY": (["time", "face", "j_g", "i"], qnet + 40_000),
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
            **{f"Theta_lev_{i}": float(i) for i in range(n_lev)},
            "oceQnet": 0.0,
        }
    )
    stds = xr.Dataset(
        {
            **{f"Theta_lev_{i}": 1.0 for i in range(n_lev)},
            "oceQnet": 1.0,
        }
    )
    return data, means, stds


def write_raw_llc_datasets(path: Path) -> None:
    data, means, stds = raw_llc_datasets()
    data.to_zarr(path / "data.zarr")
    means.to_netcdf(path / "means.nc")
    stds.to_netcdf(path / "stds.nc")
