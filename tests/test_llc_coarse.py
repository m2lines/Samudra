from pathlib import Path

import numpy as np
import pytest

from ocean_emulators.llc_coarse import (
    aligned_weights,
    apply_weights,
    bottom_hfac,
    box_sum,
    cell_area,
    center_inplace,
    corner_grid,
    repair_candidates,
    lonlat_from_vector,
    nearest,
    ratio,
    rotate_to_east_north,
    split_edges,
    subcell_weights,
    tile_matrix,
    unit_vector,
)

RAW_LLC = Path("/orcd/data/abodner/003/LLC4320/LLC4320")
COARSE_LAT = np.array([-2.0, -1.003, 0.0, 1.2, 2.0])  # uneven, like Samudra's bands


def _llc_like_tile(n=48, m=48, lon0=10.0, uneven=True):
    """A lat/lon-rectangle tile: 1/48-degree columns, rows not on target edges."""
    lon_e = lon0 + np.arange(m + 1) / 48
    lat_e = np.cumsum(np.r_[-1.27, np.linspace(0.015, 0.03 if uneven else 0.015, n)])
    lon_c, lat_c = np.meshgrid(lon_e, lat_e)
    ra = cell_area(lat_e, lon_e, 1.0)
    xc = np.broadcast_to((lon_e[:-1] + lon_e[1:]) / 2, ra.shape)
    return lon_c, lat_c, xc, ra, lat_e, lon_e


def _exact_cell_areas(lat_e, lon_e, target_lat, n_lon):
    """Overlap area of the tile with each target cell, from the edges alone."""
    out = {}
    lon_t = np.arange(n_lon + 1) * 360 / n_lon
    for r in range(target_lat.size - 1):
        s, n = max(lat_e[0], target_lat[r]), min(lat_e[-1], target_lat[r + 1])
        for c in range(n_lon):
            w, e = max(lon_e[0], lon_t[c]), min(lon_e[-1], lon_t[c + 1])
            if s < n and w < e:
                out[r * n_lon + c] = (np.sin(np.deg2rad(n)) - np.sin(np.deg2rad(s))) * np.deg2rad(e - w)
    return out


def _cell_sums(cell, area):
    cells, inverse = np.unique(cell, return_inverse=True)
    return dict(zip(cells.tolist(), np.bincount(inverse, weights=area)))


def test_split_edges_and_cell_area():
    edges = split_edges(COARSE_LAT, 4)
    assert edges.size == 17 and np.allclose(edges[::4], COARSE_LAT)
    assert np.allclose(np.diff(edges)[:4], (COARSE_LAT[1] - COARSE_LAT[0]) / 4)
    area = cell_area(np.linspace(-90, 90, 181), np.linspace(0, 360, 361), 2.0)
    assert np.isclose(area.sum(), 4 * np.pi * 4.0)


def test_aligned_weights_are_exact():
    lon_c, lat_c, xc, ra, lat_e, lon_e = _llc_like_tile()
    target = split_edges(COARSE_LAT, 4)
    pixel, cell, area = aligned_weights(lon_c, lat_c, xc, ra, target, 1440)
    np.testing.assert_allclose(np.bincount(pixel, weights=area), ra.ravel(), rtol=1e-6)
    got, want = _cell_sums(cell, area), _exact_cell_areas(lat_e, lon_e, target, 1440)
    assert got.keys() == want.keys()
    np.testing.assert_allclose([got[k] for k in want], list(want.values()), rtol=1e-5)


def test_aligned_weights_ignore_float_noise_at_target_edges():
    lon_c, lat_c, xc, ra, lat_e, _ = _llc_like_tile()
    # Put a target edge on a fine row edge, off by float noise only.
    target = np.r_[-90.0, lat_e[10] - 1e-12, 90.0]
    pixel, cell, area = aligned_weights(lon_c, lat_c, xc, ra, target, 1440)
    assert pixel.size == ra.size  # no cell was split
    assert area.min() > 0.5 * ra.min()


def test_aligned_weights_on_a_rotated_face_and_rejecting_curved_cells():
    lon_c, lat_c, xc, ra, _, _ = _llc_like_tile()
    target = split_edges(COARSE_LAT, 4)
    _, cell, area = aligned_weights(lon_c, lat_c, xc, ra, target, 1440)
    # Faces 7-12 store the same cells transposed (+i south, +j east).
    _, cell_t, area_t = aligned_weights(lon_c.T[::-1], lat_c.T[::-1], xc.T[::-1], ra.T[::-1], target, 1440)
    assert _cell_sums(cell, area).keys() == _cell_sums(cell_t, area_t).keys()
    np.testing.assert_allclose(sorted(_cell_sums(cell, area).values()),
                               sorted(_cell_sums(cell_t, area_t).values()), rtol=1e-6)
    curved = lon_c + 0.01 * (lat_c - lat_c.mean())
    assert aligned_weights(curved, lat_c, xc, ra, target, 1440) is None


def test_subcell_weights_conserve_and_approximate():
    lon_c, lat_c, _, ra, lat_e, lon_e = _llc_like_tile()
    target = split_edges(COARSE_LAT, 4)
    pixel, cell, area = subcell_weights(lon_c, lat_c, ra, target, 1440)
    np.testing.assert_allclose(np.bincount(pixel, weights=area), ra.ravel(), rtol=1e-6)
    got, want = _cell_sums(cell, area), _exact_cell_areas(lat_e, lon_e, target, 1440)
    err = np.array([abs(got.get(k, 0) - v) for k, v in want.items()])
    # At worst a quarter of a fine row across one 0.25-degree column (12 cells).
    assert err.max() < 12 * ra.max() / 4
    assert err.sum() / ra.sum() < 0.01


def test_one_degree_from_quarter_sums_matches_direct_weights():
    lon_c, lat_c, xc, ra, _, _ = _llc_like_tile(n=96, m=96, lon0=9.0)
    quarter = aligned_weights(lon_c, lat_c, xc, ra, split_edges(COARSE_LAT, 4), 1440)
    direct = aligned_weights(lon_c, lat_c, xc, ra, COARSE_LAT, 360)
    row, col = np.divmod(quarter[1], 1440)
    up = _cell_sums((row // 4) * 360 + col // 4, quarter[2])
    want = _cell_sums(direct[1], direct[2])
    assert up.keys() == want.keys()
    np.testing.assert_allclose([up[k] for k in want], list(want.values()), rtol=1e-5)


def test_tile_matrix_applies_area_sums():
    rng = np.random.default_rng(3)
    pixel = rng.integers(0, 100, 300).astype(np.int32)
    cell = rng.integers(5000, 5040, 300).astype(np.int32)
    area = rng.uniform(0.1, 1, 300).astype(np.float32)
    cells, matrix = tile_matrix(pixel, cell, area, 100)
    x = rng.normal(size=(3, 100)).astype(np.float32)
    got = apply_weights(matrix, x)
    for k in range(3):
        want = np.bincount(np.searchsorted(cells, cell), weights=area * x[k, pixel], minlength=cells.size)
        np.testing.assert_allclose(got[k], want, rtol=1e-5)


def test_corner_grid_extrapolates_last_row_and_column():
    lon_c, lat_c, *_ = _llc_like_tile(uneven=False)
    lon, lat = corner_grid(lon_c[:-1, :-1], lat_c[:-1, :-1])
    np.testing.assert_allclose(lon, lon_c, atol=1e-9)
    np.testing.assert_allclose(lat, lat_c, atol=2e-6)


def test_nearest_finds_the_closest_candidate():
    candidates = unit_vector(np.array([0.0, 90.0, 180.0]), np.array([89.0, 89.5, 89.9]))
    index, _ = nearest(unit_vector(np.array([179.0]), np.array([89.95])), candidates)
    assert index[0] == 2


def test_ratio_and_box_sum():
    num, den = np.array([2.0, 1.0]), np.array([4.0, 0.0])
    out = ratio(num, den)
    assert out[0] == 0.5 and np.isnan(out[1])
    assert box_sum(np.ones((8, 8), np.float32), 4).tolist() == [[16, 16], [16, 16]]


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
    assert abs(abs(lon) - 180.0) < 1e-9
    assert abs(lat - 10.0) < 1e-2


def test_repair_candidates_look_back_first():
    assert repair_candidates(288, 10311, window=3) == [287, 286, 289, 290]
    assert repair_candidates(0, 10311, window=3) == [1, 2]
    assert repair_candidates(10310, 10311, window=3)[:2] == [10309, 10308]


@pytest.mark.skipif(not RAW_LLC.exists(), reason="raw LLC4320 store not mounted")
def test_real_tiles_align_where_expected():
    """Face 1 tiles are lon/lat rectangles; the Arctic cap's are not."""
    import zarr

    source = zarr.open_group(str(RAW_LLC), mode="r")
    target = split_edges(np.linspace(-90, 90, 181), 4)

    def tile(f, j0, i0):
        xg = source["XG"][f, j0:j0 + 721, i0:i0 + 721]
        yg = source["YG"][f, j0:j0 + 721, i0:i0 + 721]
        lon_c, lat_c = corner_grid(xg, yg)
        lon_c, lat_c = lon_c[:721, :721], lat_c[:721, :721]
        xc = source["XC"][f, j0:j0 + 720, i0:i0 + 720]
        ra = source["rA"][f, j0:j0 + 720, i0:i0 + 720]
        return aligned_weights(lon_c, lat_c, xc, ra, target, 1440), ra

    weights, ra = tile(1, 2160, 720)
    assert weights is not None
    np.testing.assert_allclose(np.bincount(weights[0], weights=weights[2]), ra.ravel(), rtol=1e-5)
    assert tile(6, 1440, 1440)[0] is None
