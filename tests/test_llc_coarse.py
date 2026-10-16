from pathlib import Path

import numpy as np
import pytest

from ocean_emulators.llc_coarse import (
    ARCTIC_FACE,
    bottom_hfac,
    box_sum,
    center_inplace,
    faces_to_latlon,
    fill_sources,
    lonlat_from_vector,
    ratio,
    rotate_to_east_north,
    unit_vector,
    weighted_box_sum,
)

RAW_LLC = Path("/orcd/data/abodner/003/LLC4320/LLC4320")


def _sums(x, area, wet, b):
    return weighted_box_sum(x, area, wet, b), weighted_box_sum(np.float32(1), area, wet, b)


def _field(rng, shape=(2, 48, 48)):
    x = rng.normal(size=shape).astype(np.float32)
    area = rng.uniform(0.5, 1.5, size=shape[-2:]).astype(np.float32)
    wet = rng.uniform(size=shape) > 0.3
    return x, area, wet


def test_weighted_box_sums_conserve():
    x, area, wet = _field(np.random.default_rng(0))
    num, den = _sums(x, area, wet, 12)
    mean = ratio(num, den)
    fine = np.where(wet, x * area, 0).sum(axis=(-2, -1), dtype=np.float64)
    coarse = np.nansum(mean.astype(np.float64) * den, axis=(-2, -1))
    np.testing.assert_allclose(coarse, fine, rtol=1e-6)


def test_one_degree_from_quarter_sums_matches_direct():
    x, area, wet = _field(np.random.default_rng(1), shape=(2, 96, 96))
    num, den = _sums(x, area, wet, 12)
    via_quarter = ratio(box_sum(num, 4), box_sum(den, 4))
    direct = ratio(*_sums(x, area, wet, 48))
    np.testing.assert_allclose(via_quarter, direct, rtol=1e-6)


def test_all_land_box_is_nan_and_one_wet_pixel_is_its_value():
    x = np.arange(24 * 12, dtype=np.float32).reshape(1, 12, 24)
    area = np.ones((12, 24), dtype=np.float32)
    wet = np.zeros_like(x, dtype=bool)
    wet[0, 5, 17] = True
    hfac = wet.astype(np.float32)
    mean = ratio(*_sums(x, area, wet, 12))
    assert np.isnan(mean[0, 0, 0])
    assert mean[0, 0, 1] == x[0, 5, 17]
    wet_frac = ratio(box_sum(area * hfac, 12), box_sum(area, 12))
    np.testing.assert_allclose(wet_frac[0, 0], [0.0, 1 / 144])


def test_wet_frac_is_hfac_at_box_one_and_shrinks_down_a_shelf():
    rng = np.random.default_rng(2)
    area = rng.uniform(0.5, 1.5, size=(24, 24)).astype(np.float32)
    hfac = rng.uniform(size=(3, 24, 24)).astype(np.float32)
    np.testing.assert_allclose(ratio(box_sum(area * hfac, 1), box_sum(area, 1)), hfac, rtol=1e-6)
    # Seafloor deepens eastward: level k is wet east of column 8k.
    shelf = np.stack([(np.arange(24) >= 8 * k) * np.ones((24, 1)) for k in range(3)])
    frac = ratio(box_sum(area * shelf.astype(np.float32), 12), box_sum(area, 12))
    assert np.all(np.diff(frac[:, 0, 0]) < 0)


def test_center_inplace_averages_neighbours_and_copies_edge():
    u = np.array([[1.0, 3.0, np.nan, 7.0]], dtype=np.float32)
    center_inplace(u, axis=-1)
    np.testing.assert_array_equal(u, [[2.0, 1.5, 3.5, 7.0]])
    v = np.array([[1.0], [5.0], [9.0]], dtype=np.float32)
    center_inplace(v, axis=-2)
    np.testing.assert_array_equal(v[:, 0], [3.0, 7.0, 9.0])


def test_rotation_identity_and_rotated_faces():
    u, v = np.float32(2.0), np.float32(-3.0)
    assert rotate_to_east_north(u, v, 1.0, 0.0) == (u, v)
    # Faces 7-12: +i south, +j east, so east = v and north = -u.
    east, north = rotate_to_east_north(u, v, 0.0, -1.0)
    assert (east, north) == (v, -u)


def test_bottom_hfac():
    # Columns: deep, partial bottom at k=1, land, partial at the last level.
    hfac = np.array(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 0.4, 0.0, 1.0], [1.0, 0.0, 0.0, 0.7]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(bottom_hfac(hfac), [1.0, 0.4, 0.0, 0.7])


def test_lonlat_mean_across_the_dateline():
    xyz = unit_vector(np.array([179.0, -179.0]), np.array([10.0, 10.0])).sum(axis=-1)
    lon, lat = lonlat_from_vector(xyz)
    assert abs(abs(lon) - 180.0) < 1e-4
    assert abs(lat - 10.0) < 1e-2


def test_fill_sources():
    assert fill_sources(288, 10311, {0, 288}) == [287, 289]
    assert fill_sources(0, 10311, {0, 288}) == [1]
    with pytest.raises(ValueError):
        fill_sources(288, 10311, {288, 289})


def test_faces_to_latlon_layout_and_partner_swap():
    n = 2
    ids = np.broadcast_to(np.arange(13, dtype=np.float32)[:, None, None], (13, n, n)).copy()
    grid, cap = faces_to_latlon(ids)
    assert grid.shape == (3 * n, 4 * n) and np.all(cap == ARCTIC_FACE)
    blocks = grid.reshape(3, n, 4, n)[:, 0, :, 0]
    np.testing.assert_array_equal(blocks, [[0, 3, 9, 12], [1, 4, 8, 11], [2, 5, 7, 10]])
    # rot90(k=1): a rotated face's north edge (last j row) becomes its east column.
    face = np.zeros((13, n, n), dtype=np.float32)
    face[7, -1, :] = 1
    grid, _ = faces_to_latlon(face)
    np.testing.assert_array_equal(grid[2 * n:, 2 * n:3 * n][:, -1], [1, 1])
    dx, dy = np.full((13, n, n), 1.0), np.full((13, n, n), 2.0)
    grid, _ = faces_to_latlon(dx, partner=dy)
    assert np.all(grid[:, : 2 * n] == 1.0) and np.all(grid[:, 2 * n:] == 2.0)


@pytest.mark.skipif(not RAW_LLC.exists(), reason="raw LLC4320 store not mounted")
def test_faces_to_latlon_on_real_grid():
    """Stitched XC rises eastward and YC northward wherever YC > -70."""
    import zarr

    source = zarr.open_group(str(RAW_LLC), mode="r")
    xc = np.asarray(source["XC"][:, 24::48, 24::48])
    yc = np.asarray(source["YC"][:, 24::48, 24::48])
    lon, _ = faces_to_latlon(xc)
    lat, _ = faces_to_latlon(yc)
    ok = lat > -70
    dlon = np.diff(np.unwrap(np.deg2rad(lon), axis=-1), axis=-1)
    assert np.all(dlon[ok[:, 1:] & ok[:, :-1]] > 0)
    dlat = np.diff(lat, axis=0)
    assert np.all(dlat[ok[1:] & ok[:-1]] > 0)
