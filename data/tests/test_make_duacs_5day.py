# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.make_duacs_5day import CORE_VARS, ICE_VAR, coarsen_5day


def _daily_dataset(n_time: int) -> xr.Dataset:
    time = xr.date_range("2022-01-01", periods=n_time, freq="1D", use_cftime=False)
    rng = np.random.default_rng(0)
    data = {
        v: xr.DataArray(
            rng.standard_normal((n_time, 3, 4)),
            dims=["time", "latitude", "longitude"],
            coords={"time": time},
        )
        for v in CORE_VARS
    }
    # Ice flag: day i flagged (1) iff i is even, else 0.
    flag = (np.arange(n_time) % 2 == 0).astype("float64")
    data[ICE_VAR] = xr.DataArray(
        np.broadcast_to(flag[:, None, None], (n_time, 3, 4)).copy(),
        dims=["time", "latitude", "longitude"],
        coords={"time": time},
    )
    return xr.Dataset(data)


def test_coarsen_is_block_mean_with_trimmed_remainder():
    # 12 daily steps, window 5 -> 2 full blocks; the trailing 2 days are dropped.
    ds = _daily_dataset(12)
    out = coarsen_5day(ds, window=5)

    assert out.sizes["time"] == 2
    # Spatial dims untouched.
    assert out.sizes["latitude"] == 3 and out.sizes["longitude"] == 4
    # Each output block equals the simple mean of its 5 source days.
    for block in range(2):
        src = ds["adt"].isel(time=slice(block * 5, block * 5 + 5)).mean("time")
        np.testing.assert_allclose(out["adt"].isel(time=block).values, src.values)


def test_ice_flag_becomes_window_fraction():
    # Window of days [0..4] -> flags [1,0,1,0,1] -> fraction 3/5.
    ds = _daily_dataset(5)
    out = coarsen_5day(ds, window=5)
    np.testing.assert_allclose(out[ICE_VAR].values, 0.6)
    # Relabelled: no longer advertised as a status flag.
    assert "flag_values" not in out[ICE_VAR].attrs
    assert out[ICE_VAR].attrs["units"] == "1"


def test_missing_core_variable_raises():
    ds = _daily_dataset(5).drop_vars("sla")
    with pytest.raises(KeyError, match="sla"):
        coarsen_5day(ds, window=5)
