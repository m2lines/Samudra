# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import xarray as xr

from samudra.constants import build_llc_spec


def raw_llc_datasets(n_time: int = 3) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """A tiny LLC dataset matching the production root's variable schema."""
    n_face = 2
    n_lev = len(build_llc_spec().depth_i_levels)
    n_lev_p1 = n_lev + 1
    n_j = 4
    n_i = 5
    times = np.arange(
        np.datetime64("2011-09-10T12:00:00"),
        np.datetime64("2011-09-10T12:00:00") + np.timedelta64(n_time, "D"),
        np.timedelta64(1, "D"),
        dtype="datetime64[ns]",
    )

    tracer = np.arange(n_time * n_lev * n_face * n_j * n_i, dtype=np.float32).reshape(
        n_time, n_lev, n_face, n_j, n_i
    )
    surface = np.arange(n_time * n_face * n_j * n_i, dtype=np.float32).reshape(
        n_time, n_face, n_j, n_i
    )
    w = np.arange(n_time * n_lev_p1 * n_face * n_j * n_i, dtype=np.float32).reshape(
        n_time, n_lev_p1, n_face, n_j, n_i
    )

    mask_c = np.ones((n_lev, n_face, n_j, n_i), dtype=bool)
    mask_c[:, :, 0, 0] = False
    mask_w = mask_c.copy()
    mask_s = mask_c.copy()
    mask_w[:, :, 1, 1] = False
    mask_s[:, :, 1, 2] = False

    hfac_c = mask_c.astype(np.float32)
    hfac_w = mask_w.astype(np.float32)
    hfac_s = mask_s.astype(np.float32)
    # Exercise the production hFac* fractional-cell dtype and semantics.
    hfac_c[0, 0, 1, 1] = np.float32(0.5)
    hfac_w[0, 0, 1, 2] = np.float32(0.5)
    hfac_s[0, 0, 1, 3] = np.float32(0.5)

    grid = np.arange(n_face * n_j * n_i, dtype=np.float32).reshape(n_face, n_j, n_i)

    def shifted(values: np.ndarray, offset: int) -> np.ndarray:
        return values + np.float32(offset)

    data = xr.Dataset(
        {
            "CS": (["face", "j", "i"], shifted(grid, 1_000)),
            "Depth": (["face", "j", "i"], shifted(grid, 2_000)),
            "Eta": (["time", "face", "j", "i"], shifted(surface, 3_000)),
            "KPPhbl": (["time", "face", "j", "i"], shifted(surface, 4_000)),
            "PHrefC": (["k"], np.arange(n_lev, dtype=np.float32)),
            "PHrefF": (["k_p1"], np.arange(n_lev_p1, dtype=np.float32)),
            "PhiBot": (["time", "face", "j", "i"], shifted(surface, 5_000)),
            "SIarea": (["time", "face", "j", "i"], shifted(surface, 6_000)),
            "SIheff": (["time", "face", "j", "i"], shifted(surface, 7_000)),
            "SIhsalt": (["time", "face", "j", "i"], shifted(surface, 8_000)),
            "SIhsnow": (["time", "face", "j", "i"], shifted(surface, 9_000)),
            "SIuice": (["time", "face", "j", "i_g"], shifted(surface, 10_000)),
            "SIvice": (["time", "face", "j_g", "i"], shifted(surface, 11_000)),
            "SN": (["face", "j", "i"], shifted(grid, 12_000)),
            "SSH": (["time", "face", "j", "i"], shifted(surface, 13_000)),
            "SSH_notides": (
                ["time", "face", "j", "i"],
                shifted(surface, 14_000),
            ),
            "Salt": (
                ["time", "k", "face", "j", "i"],
                shifted(tracer, 100_000),
            ),
            "Theta": (["time", "k", "face", "j", "i"], tracer),
            "U": (["time", "k", "face", "j", "i_g"], shifted(tracer, 200_000)),
            "V": (["time", "k", "face", "j_g", "i"], shifted(tracer, 300_000)),
            "W": (["time", "k_p1", "face", "j", "i"], shifted(w, 400_000)),
            "XC": (["face", "j", "i"], shifted(grid, 15_000)),
            "XG": (["face", "j_g", "i_g"], shifted(grid, 16_000)),
            "YC": (["face", "j", "i"], shifted(grid, 17_000)),
            "YG": (["face", "j_g", "i_g"], shifted(grid, 18_000)),
            "Z": (["k"], -np.arange(n_lev, dtype=np.float32)),
            "Zl": (["k_l"], -np.arange(n_lev, dtype=np.float32)),
            "Zp1": (["k_p1"], -np.arange(n_lev_p1, dtype=np.float32)),
            "Zu": (["k_u"], -np.arange(n_lev, dtype=np.float32)),
            "drC": (["k_p1"], np.arange(n_lev_p1, dtype=np.float32)),
            "drF": (["k"], np.arange(n_lev, dtype=np.float32)),
            "dxC": (["face", "j", "i_g"], shifted(grid, 19_000)),
            "dxF": (["face", "j", "i"], shifted(grid, 20_000)),
            "dxG": (["face", "j_g", "i"], shifted(grid, 21_000)),
            "dxV": (["face", "j_g", "i_g"], shifted(grid, 22_000)),
            "dyC": (["face", "j_g", "i"], shifted(grid, 23_000)),
            "dyF": (["face", "j", "i"], shifted(grid, 24_000)),
            "dyG": (["face", "j", "i_g"], shifted(grid, 25_000)),
            "dyU": (["face", "j_g", "i_g"], shifted(grid, 26_000)),
            "hFacC": (["k", "face", "j", "i"], hfac_c),
            "hFacS": (["k", "face", "j_g", "i"], hfac_s),
            "hFacW": (["k", "face", "j", "i_g"], hfac_w),
            "iter": (["time"], np.arange(n_time, dtype=np.int64)),
            "mask_c": (["k", "face", "j", "i"], mask_c),
            "mask_s": (["k", "face", "j_g", "i"], mask_s),
            "mask_w": (["k", "face", "j", "i_g"], mask_w),
            "oceFWflx": (["time", "face", "j", "i"], shifted(surface, 27_000)),
            "oceQnet": (["time", "face", "j", "i"], shifted(surface, 500_000)),
            "oceQsw": (["time", "face", "j", "i"], shifted(surface, 28_000)),
            "oceSflux": (["time", "face", "j", "i"], shifted(surface, 29_000)),
            "oceTAUX": (
                ["time", "face", "j", "i_g"],
                shifted(surface, 600_000),
            ),
            "oceTAUY": (
                ["time", "face", "j_g", "i"],
                shifted(surface, 700_000),
            ),
            "rA": (["face", "j", "i"], shifted(grid, 30_000)),
            "rAs": (["face", "j_g", "i"], shifted(grid, 31_000)),
            "rAw": (["face", "j", "i_g"], shifted(grid, 32_000)),
            "rAz": (["face", "j_g", "i_g"], shifted(grid, 33_000)),
            "rhoRef": (["k"], shifted(np.arange(n_lev, dtype=np.float32), 1_000)),
        },
        coords={
            "time": times,
            "face": np.arange(n_face, dtype=np.int16),
            "i": np.arange(n_i, dtype=np.int16),
            "i_g": np.arange(n_i, dtype=np.int16),
            "j": np.arange(n_j, dtype=np.int16),
            "j_g": np.arange(n_j, dtype=np.int16),
            "k": np.arange(n_lev, dtype=np.int16),
            "k_l": np.arange(n_lev, dtype=np.int64),
            "k_p1": np.arange(n_lev_p1, dtype=np.int64),
            "k_u": np.arange(n_lev, dtype=np.int64),
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
