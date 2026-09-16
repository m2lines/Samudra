# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""OHC figures must integrate physical ocean volume, not normalized weights."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes  # type: ignore

from samudra.constants import build_om4_layout
from samudra.viz import core


@pytest.fixture(params=[1.0, 7.0], ids=["original-area", "seven-times-area"])
def ocean(request, tmp_path, monkeypatch):
    # Five wet columns and one land column. Changing physical area must scale
    # heat content even though the normalized weights are identical.
    area = np.array([[2, 3, 4], [5, 6, 7]]) * 1e10 * request.param
    warming = np.arange(300, dtype=float) + 1
    temperature = np.broadcast_to(warming[:, None, None, None], (300, 3, 2, 3)).copy()
    temperature[:, :, 1, 2] = np.nan
    ds = xr.Dataset(
        {
            "thetao": (("time", "lev", "y", "x"), temperature),
            "areacello": (("y", "x"), area / area.sum()),
            "areacello_spherical": (("y", "x"), area),
            "dz": ("lev", [200.0, 1000.0, 2000.0]),
        },
        coords={
            "time": pd.date_range("2001-01-01", periods=300, freq="5D"),
            "lev": [100.0, 1000.0, 3000.0],
            "y": [-30.0, 30.0],
            "x": [0.0, 120.0, 240.0],
        },
    )
    prediction = ds.copy()
    prediction["thetao"] = 3 * ds.thetao
    viz = object.__new__(core.Viz)
    viz.data_layout = build_om4_layout()
    viz.data = ds
    viz.pred_dict = {"model": {"name": "model", "ds_prediction": prediction}}
    viz.key1 = "model"
    viz.dataset_name = "truth"
    viz.clist = ["red"]
    viz.output_path = str(tmp_path)
    viz.ohc_path = str(tmp_path)
    viz.var_list = {"OHC": "Heat content [ZJ]"}
    viz.basin_masks = xr.Dataset(
        {
            "Atlantic": (("y", "x"), [[1, np.nan, np.nan], [1, np.nan, np.nan]]),
            "Pacific": (("y", "x"), [[np.nan, 1, 1], [np.nan, 1, 1]]),
        },
        coords={"y": ds.y, "x": ds.x},
    )
    # Exercise the real plotting/reduction code without writing 600 dpi figures
    # or downloading coastlines. The figures themselves remain inspectable.
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)
    yield viz, warming, request.param
    plt.close("all")


@pytest.fixture
def heat_before_anomaly(monkeypatch):
    recorded = []
    original = core.remove_climatology

    def record(data):
        recorded.append(data.copy(deep=True))
        return original(data)

    monkeypatch.setattr(core, "remove_climatology", record)
    return recorded


def expected_heat(warming, area_scale, area_units=20, thickness=3200):
    # Each area unit is 10^10 m²; the five wet columns total 20 units.
    # rho * cp * volume * delta_temperature gives joules; 10^21 J = 1 ZJ.
    volume = area_units * 1e10 * area_scale * thickness
    return warming * volume * 1025 * 3850 / 1e21


def test_reference_time_ohc_has_physical_magnitude(ocean):
    viz, warming, scale = ocean
    viz.step_ohc_noanomaly_plots()
    lines = {
        line.get_label(): np.asarray(line.get_ydata(), dtype=float)
        for line in plt.gca().lines
    }
    expected = expected_heat(warming - warming[0], scale)
    np.testing.assert_allclose(lines["truth"], expected)
    np.testing.assert_allclose(lines["model"], 3 * expected)


@pytest.mark.parametrize(
    "step,volumes",
    [
        ("step_depthwise_ohc_plots", [(20, 200), (20, 1000), (20, 2000)]),
        ("step_basin_ohc_plots", [(7, 3200), (13, 3200)]),
        ("step_basin_ohc_upto_700_plots", [(7, 200), (13, 200)]),
    ],
)
def test_ohc_subregions_integrate_known_volumes(
    ocean, heat_before_anomaly, step, volumes
):
    viz, warming, scale = ocean
    getattr(viz, step)()
    assert len(heat_before_anomaly) == 2 * len(volumes)
    for i, (area_units, thickness) in enumerate(volumes):
        expected = expected_heat(warming, scale, area_units, thickness)
        np.testing.assert_allclose(heat_before_anomaly[2 * i], expected)
        np.testing.assert_allclose(heat_before_anomaly[2 * i + 1], 3 * expected)


def test_global_anomaly_keeps_existing_physical_scaling(ocean, heat_before_anomaly):
    viz, warming, scale = ocean
    anomaly = viz.ohc_anomaly_global(viz.data)
    np.testing.assert_allclose(heat_before_anomaly[0], expected_heat(warming, scale))
    assert anomaly.attrs["units"] == "ZJ"


def test_ohc_maps_sum_to_the_physical_global_integral(ocean, heat_before_anomaly):
    viz, warming, scale = ocean
    viz.step_ohc_maps()
    # Both the first/last-year panels and later year-difference maps compute
    # ground-truth and prediction columns separately.
    assert len(heat_before_anomaly) == 4
    column_areas = np.array([[2, 3, 4], [5, 6, 0]])
    expected = expected_heat(warming[:, None, None], scale, column_areas)
    for columns, multiplier in zip(heat_before_anomaly, [1, 3, 1, 3]):
        np.testing.assert_allclose(columns, multiplier * expected)
        np.testing.assert_allclose(
            columns.sum(["y", "x"]), multiplier * expected_heat(warming, scale)
        )
