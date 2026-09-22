# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from samudra.viz.core import postprocess_for_plot


@pytest.mark.parametrize("writer_layout", [False, True])
def test_plot_mask_preserves_horizontal_shape(writer_layout):
    """Both legacy and writer grids must mask without cross-grid broadcasting."""
    values = np.arange(12, dtype=float).reshape(2, 1, 2, 3)
    mask = np.array([[[True, False, True], [True, True, True]]])
    truth = xr.Dataset(
        {
            "so": (("time", "lev", "lat", "lon"), values),
            "mask": (("lev", "lat", "lon"), mask),
            "areacello": (("lat", "lon"), np.ones((2, 3))),
            "areacello_spherical": (("lat", "lon"), np.ones((2, 3))),
        },
        coords={
            "time": [0, 1],
            "lev": [5.0],
            "lat": [-1.0, 1.0],
            "lon": [0.0, 1.0, 2.0],
        },
    )
    prediction = truth[["so"]].copy(deep=True)
    if writer_layout:
        prediction = prediction.rename({"lat": "y", "lon": "x"})
        prediction = prediction.assign_coords(
            lat=(("y", "x"), np.broadcast_to(np.array([-1.0, 1.0])[:, None], (2, 3))),
            lon=(("y", "x"), np.broadcast_to(np.arange(3), (2, 3))),
        )
    original = prediction.copy(deep=True)
    truth_out, predictions = postprocess_for_plot(
        truth, truth.areacello, np.array([10.0]), {"run": {"ds_prediction": prediction}}
    )
    result = predictions["run"]["ds_prediction"]
    assert result.so.dims == ("time", "lev", "y", "x")
    assert result.so.shape == values.shape
    np.testing.assert_allclose(result.so.values, truth_out.so.values, equal_nan=True)
    if writer_layout:
        assert result.lat_2d.dims == ("y", "x")
        assert result.lon_2d.dims == ("y", "x")
    xr.testing.assert_equal(prediction, original)
