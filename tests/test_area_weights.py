# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest
import torch
import xarray as xr

from samudra.aggregator.metrics import area_weighted_mean, area_weighted_rmse
from samudra.config import JulianDate, Om4DataSourceConfig, Om4TimeConfig
from samudra.constants import GridType
from samudra.utils.data import CanonicalSource


def _source(area=None, *, grid_type: GridType = "tripolar", as_coord=False):
    # Index axes deliberately differ from the nonseparable true geography.
    data = xr.Dataset(
        {
            "thetao_0": (("time", "y", "x"), np.zeros((2, 2, 3))),
            "hfds": (("time", "y", "x"), np.zeros((2, 2, 3))),
            "mask_0": (("y", "x"), [[True, True, True], [True, True, False]]),
        },
        coords={
            "time": xr.date_range(
                "2000-01-01 12:00:00", periods=2, calendar="julian", use_cftime=True
            ),
            "y": [0.0, 1.0],
            "x": [0.0, 1.0, 2.0],
            "lat": (("y", "x"), [[65.0, 70.0, 65.0], [75.0, 85.0, 75.0]]),
            "lon": (("y", "x"), [[0.0, 120.0, 240.0], [30.0, 180.0, 330.0]]),
        },
    )
    if area is not None:
        data["areacello"] = (
            area if isinstance(area, xr.DataArray) else (("y", "x"), area)
        )
        if as_coord:
            data = data.set_coords("areacello")
    stats = xr.Dataset({"thetao_0": 0.0, "hfds": 0.0})
    config = Om4DataSourceConfig.model_construct(
        prognostic_vars_key="thetao_1", boundary_vars_key="hfds", grid_type=grid_type
    )
    canonical, means, stds, layout = config.canonicalize_datasets(
        data, stats, stats + 1
    )
    return CanonicalSource.from_datasets(
        canonical,
        means,
        stds,
        data_layout=layout,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
    )


@pytest.mark.parametrize("as_coord", [False, True])
@pytest.mark.parametrize("grid_type", ["tripolar", "gaussian", "llc"])
def test_source_areas_drive_masked_mean_and_rmse(as_coord, grid_type):
    source = _source(
        [[1.0, 2.0, 3.0], [4.0, 10.0, np.nan]], as_coord=as_coord, grid_type=grid_type
    )
    # Hand-computed ocean mean = (1*1 + 2*2 + 3*3 + 4*4 + 10*5) / 20 = 4.
    field = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, float("nan")]])
    assert area_weighted_mean(field, source.area_weights).item() == pytest.approx(4.0)
    assert area_weighted_rmse(
        torch.zeros_like(field), field, source.area_weights
    ).item() == pytest.approx(np.sqrt(17.5))
    assert source.area_weights.sum().item() == pytest.approx(1.0)
    assert source.area_weights[1, 2] == 0
    assert "areacello" in source.coordinates()
    sliced = source.slice_time(
        Om4TimeConfig(start=JulianDate("2000-01-02"), end=JulianDate("2000-01-02"))
    )
    torch.testing.assert_close(sliced.area_weights, source.area_weights)


def test_transposed_cell_areas_are_aligned_by_dimension_names():
    source = _source(
        xr.DataArray([[1.0, 4.0], [2.0, 10.0], [3.0, 0.0]], dims=("x", "y"))
    )
    torch.testing.assert_close(
        source.area_weights,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 10.0, 0.0]], dtype=torch.float64) / 20,
    )


def test_legacy_gaussian_weights_are_unchanged():
    source = _source(grid_type="gaussian")
    lat, lon = source.resolution
    expected = torch.cos(torch.deg2rad(lat)).repeat(lon.numel(), 1).T
    torch.testing.assert_close(
        source.area_weights, expected / expected.sum(), rtol=0, atol=0
    )


@pytest.mark.parametrize("grid_type", ["tripolar", "llc"])
def test_curvilinear_source_requires_cell_areas(grid_type):
    with pytest.raises(ValueError, match="requires 'areacello'"):
        _ = _source(grid_type=grid_type).area_weights


@pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0, np.inf])
def test_invalid_wet_areas_fail(bad):
    with pytest.raises(ValueError, match="areacello"):
        _ = _source([[bad, 2.0, 3.0], [4.0, 10.0, 0.0]]).area_weights


@pytest.mark.parametrize("bad", [-1.0, np.inf])
def test_invalid_land_areas_fail(bad):
    with pytest.raises(ValueError, match="areacello"):
        _ = _source([[1.0, 2.0, 3.0], [4.0, 10.0, bad]]).area_weights


def test_wrong_area_dimensions_fail():
    area = xr.DataArray(np.ones((2, 3)), dims=("y", "other_x"))
    with pytest.raises(ValueError, match="dimensions"):
        _ = _source(area).area_weights


def test_boundary_only_wet_cell_requires_valid_area():
    source = _source([[1.0, 2.0, 3.0], [4.0, 10.0, np.nan]])
    boundary = source.masks.boundary.clone()
    boundary[1, 2] = True
    source = replace(source, masks=replace(source.masks, boundary=boundary))
    with pytest.raises(ValueError, match="wet cells"):
        _ = source.area_weights


def test_native_reader_uses_the_same_static_areas():
    from tests.test_native_loader import ArrayBulkReader

    source = _source([[1.0, 2.0, 3.0], [4.0, 10.0, np.nan]])
    native = replace(
        source,
        _reader=ArrayBulkReader(source, source.reader, np.arange(source.time.size), []),
    )
    torch.testing.assert_close(native.area_weights, source.area_weights)


def test_empty_area_grid_fails_even_when_all_cells_are_dry():
    source = _source(np.zeros((2, 3)))
    masks = replace(
        source.masks,
        prognostic=torch.zeros_like(source.masks.prognostic),
        boundary=torch.zeros_like(source.masks.boundary),
    )
    with pytest.raises(ValueError, match="positive total"):
        _ = replace(source, masks=masks).area_weights
