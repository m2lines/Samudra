import pytest
import torch

from ocean_emulators.config import RegionBoundsConfig
from ocean_emulators.utils.regions import build_region_weights


def test_build_region_weights_basic_0_360():
    lat = torch.tensor([-10.0, 0.0, 10.0])
    lon = torch.tensor([0.0, 90.0, 180.0, 270.0])
    area_weights = torch.ones(len(lat), len(lon))
    regions = [
        RegionBoundsConfig(
            name="box", lat_min=-5.0, lat_max=5.0, lon_min=80.0, lon_max=200.0
        )
    ]

    weights = build_region_weights(
        regions, lat=lat, lon=lon, area_weights=area_weights
    )["box"]
    mask = weights > 0

    expected = torch.zeros_like(mask)
    expected[1, 1] = True
    expected[1, 2] = True
    assert torch.equal(mask, expected)


def test_build_region_weights_wrap_0_360():
    lat = torch.tensor([-10.0, 10.0])
    lon = torch.tensor([0.0, 60.0, 120.0, 300.0, 330.0])
    area_weights = torch.ones(len(lat), len(lon))
    regions = [
        RegionBoundsConfig(
            name="wrap", lat_min=-90.0, lat_max=90.0, lon_min=300.0, lon_max=60.0
        )
    ]

    weights = build_region_weights(
        regions, lat=lat, lon=lon, area_weights=area_weights
    )["wrap"]
    mask = weights > 0

    expected_cols = torch.tensor([True, True, False, True, True])
    assert torch.equal(mask[0], expected_cols)
    assert torch.equal(mask[1], expected_cols)


def test_build_region_weights_wrap_neg_180():
    lat = torch.tensor([-10.0, 10.0])
    lon = torch.tensor([-170.0, -90.0, 0.0, 90.0, 170.0])
    area_weights = torch.ones(len(lat), len(lon))
    regions = [
        RegionBoundsConfig(
            name="wrap", lat_min=-90.0, lat_max=90.0, lon_min=170.0, lon_max=-170.0
        )
    ]

    weights = build_region_weights(
        regions, lat=lat, lon=lon, area_weights=area_weights
    )["wrap"]
    mask = weights > 0

    expected_cols = torch.tensor([True, False, False, False, True])
    assert torch.equal(mask[0], expected_cols)
    assert torch.equal(mask[1], expected_cols)


def test_build_region_weights_empty_raises():
    lat = torch.tensor([-10.0, 10.0])
    lon = torch.tensor([0.0, 90.0])
    area_weights = torch.ones(len(lat), len(lon))
    regions = [
        RegionBoundsConfig(
            name="empty", lat_min=80.0, lat_max=90.0, lon_min=0.0, lon_max=10.0
        )
    ]

    with pytest.raises(ValueError, match="mask is empty"):
        build_region_weights(regions, lat=lat, lon=lon, area_weights=area_weights)


def test_build_region_weights_duplicate_name_raises():
    lat = torch.tensor([-10.0, 10.0])
    lon = torch.tensor([0.0, 90.0])
    area_weights = torch.ones(len(lat), len(lon))
    regions = [
        RegionBoundsConfig(
            name="dup", lat_min=-10.0, lat_max=10.0, lon_min=0.0, lon_max=90.0
        ),
        RegionBoundsConfig(
            name="dup", lat_min=-10.0, lat_max=10.0, lon_min=0.0, lon_max=90.0
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate region name"):
        build_region_weights(regions, lat=lat, lon=lon, area_weights=area_weights)
