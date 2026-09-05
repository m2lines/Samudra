# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Native-grid basin masks built from OM4 region codes."""

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.basin_masks import (
    OM4_BASIN_CODES,
    basin_masks_from_static,
    validate_basin_masks,
)

NY, NX = 4, 6


def _static(codes: np.ndarray | None = None, wet: np.ndarray | None = None):
    """A miniature `ocean_static` on a nonseparable (curvilinear) grid."""
    y = np.linspace(-60.0, 60.0, NY)
    x = np.linspace(0.0, 270.0, NX)
    # Both coordinates vary along both dims, as they do near the tripolar fold.
    lat2d = y[:, None] + 0.1 * x[None, :]
    lon2d = x[None, :] + 0.1 * y[:, None]

    if codes is None:
        # One row per basin, plus a row of land and marginal seas.
        codes = np.array(
            [
                [1] * NX,
                [2] * NX,
                [3] * NX,
                [0, 4, 4, 5, 5, 6],
            ],
            dtype=np.int32,
        )
    if wet is None:
        wet = (codes != 0).astype(np.float64)

    return xr.Dataset(
        {
            "basin": (("yh", "xh"), codes),
            "wet": (("yh", "xh"), wet),
            "geolat": (("yh", "xh"), lat2d),
            "geolon": (("yh", "xh"), lon2d),
        },
        coords={"yh": y, "xh": x},
    )


def test_masks_follow_the_region_codes():
    masks = basin_masks_from_static(_static())

    assert set(masks.data_vars) == set(OM4_BASIN_CODES)
    np.testing.assert_array_equal(masks["basin_southern"].values[0], np.ones(NX))
    np.testing.assert_array_equal(masks["basin_atlantic"].values[1], np.ones(NX))
    np.testing.assert_array_equal(masks["basin_pacific"].values[2], np.ones(NX))
    np.testing.assert_array_equal(masks["basin_arctic"].values[3], [0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(masks["basin_indian"].values[3], [0, 0, 0, 1, 1, 0])


def test_marginal_seas_are_left_unassigned():
    """Matches the published Gaussian masks, where they are zero everywhere.

    The last cell carries code 6 (Mediterranean). No basin may claim it.
    """
    masks = basin_masks_from_static(_static())

    claimed = sum(masks[name].values[3, -1] for name in OM4_BASIN_CODES)
    assert claimed == 0


def test_land_is_never_assigned():
    masks = basin_masks_from_static(_static())

    claimed = sum(masks[name].values[3, 0] for name in OM4_BASIN_CODES)
    assert claimed == 0


def test_every_wet_cell_belongs_to_at_most_one_basin():
    masks = basin_masks_from_static(_static())

    stacked = np.stack([masks[name].values for name in OM4_BASIN_CODES])
    assert stacked.sum(axis=0).max() <= 1


def test_masks_carry_the_real_2d_cell_centers():
    """Without these, a mask can only be matched to data by position."""
    static = _static()
    masks = basin_masks_from_static(static)

    assert masks["lat_2d"].dims == ("lat", "lon")
    assert masks["lon_2d"].dims == ("lat", "lon")
    np.testing.assert_array_equal(masks["lat_2d"].values, static["geolat"].values)
    np.testing.assert_array_equal(masks["lon_2d"].values, static["geolon"].values)


def test_cell_centers_are_not_recoverable_by_broadcasting():
    """Guards the fixture: a rectilinear grid would make the test above vacuous."""
    static = _static()
    lat2d = static["geolat"].values
    assert not np.allclose(
        np.broadcast_to(static["yh"].values[:, None], lat2d.shape), lat2d
    )


def test_masks_are_on_lat_lon_index_axes():
    """`samudra viz` renames its own y/x to lat/lon; masks must match."""
    masks = basin_masks_from_static(_static())

    assert masks["basin_atlantic"].dims == ("lat", "lon")
    assert masks.sizes["lat"] == NY and masks.sizes["lon"] == NX


def test_masks_describe_themselves_like_the_published_stores():
    masks = basin_masks_from_static(_static())

    attrs = masks["basin_atlantic"].attrs
    assert attrs["basin_id"] == 2
    assert attrs["long_name"] == "Atlantic basin mask"
    assert attrs["om4_basin_code"] == OM4_BASIN_CODES["basin_atlantic"]


def test_missing_inputs_fail_loudly():
    static = _static().drop_vars("geolat")

    with pytest.raises(ValueError, match="geolat"):
        basin_masks_from_static(static)


def test_mismatched_grids_fail_loudly():
    """A wet mask on a different grid than the region codes is refused."""
    static = _static()
    # A distinct dim name, so xarray keeps the mismatch instead of aligning it.
    static["wet"] = (("yh", "xh_other"), np.ones((NY, NX - 1)))

    with pytest.raises(ValueError, match="same horizontal grid"):
        basin_masks_from_static(static)


def test_validation_rejects_overlapping_basins():
    masks = basin_masks_from_static(_static())
    masks["basin_pacific"] = masks["basin_atlantic"]

    with pytest.raises(ValueError, match="more than one basin"):
        validate_basin_masks(masks)
