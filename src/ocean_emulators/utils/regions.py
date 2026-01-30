"""Utilities for region masking based on lat/lon bounds."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def _infer_lon_domain(lons: torch.Tensor) -> str:
    lon_min = float(lons.min().item())
    lon_max = float(lons.max().item())
    if lon_min >= 0.0 and lon_max <= 360.0:
        return "0_360"
    return "neg_180"


def _to_0_360(lon: float) -> float:
    lon = lon % 360.0
    if lon < 0.0:
        lon += 360.0
    return lon


def _to_neg_180_180(lon: float) -> float:
    lon_norm = ((lon + 180.0) % 360.0) - 180.0
    if lon_norm == -180.0 and lon > 0.0:
        lon_norm = 180.0
    return lon_norm


def _normalize_lon(lon: float, domain: str) -> float:
    if domain == "0_360":
        return _to_0_360(lon)
    return _to_neg_180_180(lon)


def build_region_weights(
    regions: Iterable,
    *,
    lat: torch.Tensor | list[float] | tuple[float, ...],
    lon: torch.Tensor | list[float] | tuple[float, ...],
    area_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build per-region area weights from lat/lon bounds.

    Args:
        regions: Iterable of objects with name/lat_min/lat_max/lon_min/lon_max.
        lat: 1D latitude values.
        lon: 1D longitude values.
        area_weights: 2D area weights tensor matching (lat, lon).

    Returns:
        Mapping of region name to area weights masked to that region.
    """
    regions = list(regions)
    if len(regions) == 0:
        return {}

    device = area_weights.device
    lat_t = torch.as_tensor(lat, device=device)
    lon_t = torch.as_tensor(lon, device=device)
    if lat_t.ndim != 1 or lon_t.ndim != 1:
        raise ValueError("lat and lon must be 1D arrays")
    if area_weights.shape != (lat_t.numel(), lon_t.numel()):
        raise ValueError(
            "area_weights shape must match lat/lon lengths; "
            f"got {area_weights.shape} vs ({lat_t.numel()}, {lon_t.numel()})"
        )

    domain = _infer_lon_domain(lon_t)
    region_weights: dict[str, torch.Tensor] = {}

    for region in regions:
        name = region.name
        if name in region_weights:
            raise ValueError(f"Duplicate region name: {name}")

        lat_min = float(region.lat_min)
        lat_max = float(region.lat_max)
        if lat_min > lat_max:
            raise ValueError(
                f"Region '{name}' lat_min must be <= lat_max, got {lat_min} > {lat_max}"
            )

        lon_min = _normalize_lon(float(region.lon_min), domain)
        lon_max = _normalize_lon(float(region.lon_max), domain)

        lat_mask = (lat_t >= lat_min) & (lat_t <= lat_max)
        if lon_min <= lon_max:
            lon_mask = (lon_t >= lon_min) & (lon_t <= lon_max)
        else:
            lon_mask = (lon_t >= lon_min) | (lon_t <= lon_max)

        mask_2d = lat_mask[:, None] & lon_mask[None, :]
        weights = area_weights * mask_2d.to(area_weights.dtype)
        if not torch.any(weights > 0):
            raise ValueError(f"Region '{name}' mask is empty after applying bounds")
        region_weights[name] = weights

    return region_weights
