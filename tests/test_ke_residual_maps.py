# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Distinguish residual forecast error from mean/season errors and variance error."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def analysis():
    spec = importlib.util.spec_from_file_location(
        "ke_residual_maps",
        Path(__file__).parents[1] / "experiments/current-ke-aux/residual_maps.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field(values):
    # Four complete no-leap calendar cycles, with exactly repeated pentad dates.
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=5 * pentad)
            for year in (2001, 2002, 2003, 2005)
            for pentad in range(73)
        ]
    )
    return xr.DataArray(
        np.asarray(values)[:, None, None],
        dims=("time", "y", "x"),
        coords={"time": dates, "y": [0], "x": [0]},
    )


def test_own_mean_and_seasonal_error_is_excluded():
    module = analysis()
    seasonal = np.tile(np.sin(2 * np.pi * np.arange(73) / 73), 4)
    gold = field(15 + seasonal)
    model = field(100 + 3 * seasonal)
    assert float(((model - gold) ** 2).mean()) > 1000
    result = module.error_moments(module.residual(gold), module.residual(model))
    np.testing.assert_allclose(result["mse"], 0, atol=1e-23)


def test_equal_variance_wrong_anomaly_phase_still_has_error():
    module = analysis()
    variability = np.repeat([1.0, -1.0, -1.0, 1.0], 73)
    gold = module.residual(field(variability))
    model = module.residual(field(-variability))
    result = module.error_moments(gold, model)
    # A variance-map comparison would be perfect. Forecast residual errors
    # instead penalize the reversed anomalies at every matched timestamp.
    np.testing.assert_allclose(result["model_variance"], gold.var("time").values)
    np.testing.assert_allclose(result["mse"], 4 * gold.var("time").values)
    assert result["mse"].item() > 1
