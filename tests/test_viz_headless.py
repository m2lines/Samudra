# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Adversarial checks for headless profile plots and unclipped salinity drift."""

import builtins
import types
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from samudra.viz import core


@given(st.lists(st.floats(-1000, 8000, allow_nan=False), min_size=1))
@settings(max_examples=100)
def test_depth_transform_preserves_layout_and_round_trips(depths):
    depth = np.asarray(depths)
    expected = np.piecewise(
        depth,
        [depth <= 1000, depth > 1000],
        [lambda z: z + 5 * (z - 1000), lambda z: z],
    )
    forward = core._profile_depth_forward(depth)
    np.testing.assert_allclose(forward, expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        core._profile_depth_inverse(forward), depth, rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        core._profile_depth_forward(core._profile_depth_inverse(depth)),
        depth,
        rtol=0,
        atol=1e-11,
    )


def test_temperature_profiles_render_with_gui_imports_blocked(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def reject_gui_import(name, *args, **kwargs):
        if name.startswith(("xarrayutils.plotting", "tkinter")):
            pytest.fail(f"Headless profile plotting imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_gui_import)
    data = xr.DataArray(
        np.arange(146 * 5 * 3 * 2, dtype=float).reshape(146, 5, 3, 2),
        dims=("time", "lev", "y", "x"),
        coords={
            "time": pd.date_range("2000-01-01", periods=146, freq="5D"),
            "lev": [2.5, 500, 1000, 2000, 6000],
            "y": [-60, 0, 60],
            "x": [0, 180],
        },
    )
    truth = xr.Dataset({"thetao": data})
    stub = types.SimpleNamespace(
        data=truth,
        pred_dict={"model": {"name": "model", "ds_prediction": truth}},
        key1="model",
        dataset_name="truth",
        basin_masks=None,
        temp_path=tmp_path,
        _reject_on_curvilinear=lambda *args: None,
    )
    monkeypatch.setattr(core, "remove_climatology", lambda array: array)

    def basin_sections(array, *args):
        return [array.mean("x")] * 6, ["basin"] * 6

    monkeypatch.setattr(core, "get_basin_datasets", basin_sections)
    figures = []
    monkeypatch.setattr(
        plt, "savefig", lambda *args, **kwargs: figures.append(plt.gcf())
    )
    try:
        core.Viz.step_ocean_temperature_profile_plots(cast(core.Viz, stub))
        assert len(figures) == 3
        for figure in figures:
            for axis in figure.axes[:6]:
                assert axis.yaxis_inverted()
                transform = axis.yaxis.get_transform()
                depths = np.array([2.5, 999.9, 1000, 1000.1, 6000])
                np.testing.assert_allclose(
                    transform.inverted().transform(transform.transform(depths)), depths
                )
            figure.canvas.draw()
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "method",
    ["step_global_salinity_timeseries_plots", "step_ocean_salinity_profile_plots"],
)
def test_salinity_axes_include_drift_outside_previous_limits(
    tmp_path, monkeypatch, method
):
    salinity = xr.DataArray(
        np.broadcast_to(
            np.array([34.73, 34.70, 34.67])[:, None, None, None], (3, 2, 2, 2)
        ),
        dims=("time", "lev", "y", "x"),
        coords={
            "time": pd.date_range("2000-01-01", periods=3, freq="5D"),
            "lev": [2.5, 10],
            "y": [-45, 45],
            "x": [0, 180],
        },
    )
    truth = xr.Dataset(
        {
            "so": salinity * 0 + 34.724,
            "areacello_weights": (("y", "x"), np.full((2, 2), 0.25)),
            "areacello": (("y", "x"), np.ones((2, 2))),
            "dz": ("lev", [5.0, 10.0]),
        }
    )
    prediction = truth.assign(so=salinity)
    stub = types.SimpleNamespace(
        data=truth,
        pred_dict={
            "model": {"name": "model", "ds_prediction": prediction, "ls": ["so"]}
        },
        dataset_name="truth",
        clist=["red"],
        timeseries_path=tmp_path,
        salinity_path=tmp_path,
        output_path=tmp_path,
        linear_fit=lambda values: np.polyfit(np.arange(values.size), values, 1),
    )
    stub.salinity_global = types.MethodType(core.Viz.salinity_global, stub)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    try:
        getattr(core.Viz, method)(stub)
        axis = plt.gca()
        lower, upper = axis.get_ylim()
        for line in axis.lines:
            values = np.asarray(line.get_ydata(), dtype=float)
            assert lower <= np.nanmin(values)
            assert upper >= np.nanmax(values)
    finally:
        plt.close("all")
