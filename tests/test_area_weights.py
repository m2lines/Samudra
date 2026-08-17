import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.aggregator.metrics import area_weighted_mean, area_weighted_rmse
from ocean_emulators.utils.data import (
    DataSource,
    Masks,
    cell_area_weights,
    spherical_area_weights,
)


def _source(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    cell_area: torch.Tensor | None,
    name: str = "test",
) -> DataSource:
    """A DataSource carrying only what cell_area_weights reads."""
    data = xr.Dataset(coords={"lat": lat, "lon": lon})
    empty = xr.Dataset()
    masks = Masks(
        torch.ones((1, lat.size, lon.size), dtype=torch.bool),
        torch.ones((lat.size, lon.size), dtype=torch.bool),
    )
    return DataSource(
        name=name,
        data=data,
        means=empty,
        stds=empty,
        masks=masks,
        cell_area=cell_area,
    )


def test_cell_area_weights_uses_the_grids_own_areas():
    """rA is the only correct source on a curvilinear grid, and it must come
    through as a positive, normalized field preserving the relative areas."""
    area = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = cell_area_weights(
        _source(lat=np.arange(2.0), lon=np.arange(2.0), cell_area=area)
    )

    assert weights.sum() == pytest.approx(1.0)
    assert bool((weights > 0).all())
    torch.testing.assert_close(weights, area / area.sum())


def test_cell_area_weights_falls_back_to_uniform_on_index_latitudes(caplog):
    """Regression: an LLC packed cache has integer i/j in lat/lon, and
    cos(deg2rad(index)) went negative for half the domain -- which made a
    weighted mean of squared error negative and its square root NaN."""
    latitudes = np.arange(720.0, 1088.0)
    longitudes = np.arange(2880.0, 3248.0)

    with caplog.at_level("WARNING"):
        weights = cell_area_weights(
            _source(lat=latitudes, lon=longitudes, cell_area=None, name="no-rA")
        )

    assert "AREA WEIGHTS" in caplog.text
    assert "no-rA" in caplog.text
    assert weights.shape == (latitudes.size, longitudes.size)
    assert bool((weights > 0).all()), "uniform weights must be strictly positive"
    assert weights.sum() == pytest.approx(1.0)
    torch.testing.assert_close(weights, torch.full_like(weights, weights.flatten()[0]))


def test_cell_area_weights_keeps_cosine_weighting_for_real_latitudes():
    """A genuinely rectilinear grid in degrees still gets latitude weighting."""
    latitudes = np.linspace(-60.0, 60.0, 7)
    longitudes = np.linspace(0.0, 90.0, 4)
    source = _source(lat=latitudes, lon=longitudes, cell_area=None)

    weights = cell_area_weights(source)

    torch.testing.assert_close(weights, spherical_area_weights(source.data))
    assert bool((weights > 0).all())
    # The equator must outweigh 60 degrees, or it is not latitude weighting.
    assert weights[3, 0] > weights[0, 0]


def test_area_weighted_metrics_stay_finite_with_real_areas():
    """The consequence that mattered: with negative weights, a non-negative
    field could produce a negative weighted mean and a NaN RMSE."""
    torch.manual_seed(0)
    area = 1.0 + torch.rand(16, 16)
    weights = cell_area_weights(
        _source(lat=np.arange(16.0), lon=np.arange(16.0), cell_area=area)
    )

    target = torch.zeros(1, 1, 16, 16)
    # Error concentrated in half the domain, which is what broke the cosine path.
    gen = torch.zeros_like(target)
    gen[0, 0, :8] = 1.0

    assert float(area_weighted_mean((gen - target) ** 2, weights)) >= 0.0
    rmse = area_weighted_rmse(target, gen, weights)
    assert bool(torch.isfinite(rmse).all())
