# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import xarray as xr

from samudra.viz.core import postprocess_for_plot


def _groundtruth() -> xr.Dataset:
    shape = (2, 1, 2, 3)
    return xr.Dataset(
        {
            "thetao": (("time", "lev", "lat", "lon"), np.ones(shape)),
            "wetmask": (("lev", "lat", "lon"), np.ones(shape[1:], dtype=bool)),
            "areacello_spherical": (("lat", "lon"), np.ones(shape[2:])),
        },
        coords={
            "time": np.arange(shape[0]),
            "lev": [2.5],
            "lat": [-45.0, 45.0],
            "lon": [0.0, 120.0, 240.0],
        },
    )


def test_postprocess_for_plot_accepts_old_and_current_rollout_grids():
    groundtruth = _groundtruth()
    values = np.ones((2, 1, 2, 3))
    old = xr.Dataset(
        {"thetao": (("time", "lev", "lat", "lon"), values)},
        coords=groundtruth.coords,
    )
    current = xr.Dataset(
        {"thetao": (("time", "lev", "y", "x"), values)},
        coords={
            "time": groundtruth.time,
            "lev": groundtruth.lev,
            "y": np.arange(2),
            "x": np.arange(3),
            "lat": (("y", "x"), np.broadcast_to([[-45.0], [45.0]], (2, 3))),
            "lon": (("y", "x"), np.broadcast_to([[0.0, 120.0, 240.0]], (2, 3))),
        },
    )
    predictions = {
        "old": {"ds_prediction": old},
        "current": {"ds_prediction": current},
    }

    result, predictions = postprocess_for_plot(
        groundtruth,
        xr.DataArray(np.ones((2, 3)), dims=("lat", "lon")),
        np.array([5.0]),
        predictions,
    )

    assert result.thetao.dims == ("time", "lev", "y", "x")
    for prediction in predictions.values():
        assert prediction["ds_prediction"].thetao.dims == (
            "time",
            "lev",
            "y",
            "x",
        )
