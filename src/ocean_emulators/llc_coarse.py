"""LLC4320 -> Samudra lat/lon grid helpers: target grid, conservative regrid
weights, C-grid velocity centring and rotation, bottom partial cells.

Faces 0-5 have +i east and +j north. Faces 7-12 are rotated: +i south, +j east.
Face 6 is the Arctic cap. See docs/10-15-26-coarsen_LLC_plan.md.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

N_FACES = 13
#: Sub-cells per side for tiles that are not lon/lat rectangles, and for those
#: reaching POLAR_LAT, where target cells narrow to tens of metres.
SUBCELLS = 4
POLAR_SUBCELLS = 16
POLAR_LAT = 85.0
#: Degrees. XG/YG are float32, so equal edges agree to ~1e-5.
ALIGN_TOL = 2e-4
#: Degrees (~0.1 m). A cell edge this close to a target edge counts as on it,
#: so float noise never splits off a sliver into the next row.
EDGE_TOL = 1e-6


def split_edges(edges: np.ndarray, n: int) -> np.ndarray:
    """Split every interval of `edges` into `n` equal parts."""
    edges = np.asarray(edges, dtype=np.float64)
    inner = edges[:-1, None] + np.diff(edges)[:, None] * (np.arange(n) / n)
    return np.append(inner.ravel(), edges[-1])


def cell_area(lat_b: np.ndarray, lon_b: np.ndarray, radius: float) -> np.ndarray:
    """`(ny, nx)` area of lat/lon cells on a sphere of `radius`, from 1D edges."""
    dsin = np.diff(np.sin(np.deg2rad(lat_b)))
    dlon = np.deg2rad(np.diff(lon_b))
    return radius**2 * dsin[:, None] * dlon[None, :]


def box_sum(x: np.ndarray, b: int) -> np.ndarray:
    """Sum non-overlapping b x b blocks over the last two axes, in float64."""
    *lead, ny, nx = x.shape
    if ny % b or nx % b:
        raise ValueError(f"shape {x.shape[-2:]} is not divisible by box {b}")
    # Rows first (contiguous), then columns: much faster than one 2-axis reduction.
    rows = x.reshape(*lead, ny, nx // b, b).sum(axis=-1, dtype=np.float64)
    return rows.reshape(*lead, ny // b, b, nx // b).sum(axis=-2)


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """`num / den` as float32, NaN where `den == 0` (all-land cells)."""
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 0, casting="unsafe")
    return out


def unit_vector(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """`[3, ...]` Cartesian position on the unit sphere."""
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def lonlat_from_vector(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `unit_vector` for any non-zero `[3, ...]` vector, in degrees."""
    x, y, z = xyz
    return np.rad2deg(np.arctan2(y, x)), np.rad2deg(np.arctan2(z, np.hypot(x, y)))


def wrap180(d: np.ndarray) -> np.ndarray:
    """Longitude difference folded into [-180, 180)."""
    return (d + 180.0) % 360.0 - 180.0


def corner_grid(xg: np.ndarray, yg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(n+1, m+1)` corner lon/lat from MITgcm's south-west corners `XG`/`YG`.

    The far row and column belong to the next face, so they are extrapolated
    along the sphere from the last two.
    """
    n, m = xg.shape
    v = np.empty((3, n + 1, m + 1))
    v[:, :n, :m] = unit_vector(np.asarray(xg, np.float64), np.asarray(yg, np.float64))
    v[:, n, :m] = 2 * v[:, n - 1, :m] - v[:, n - 2, :m]
    v[:, :, m] = 2 * v[:, :, m - 1] - v[:, :, m - 2]
    lon, lat = lonlat_from_vector(v)
    lon[:n, :m], lat[:n, :m] = xg, yg  # keep the stored corners exactly
    return lon, lat


def column(lon: np.ndarray, n_lon: int) -> np.ndarray:
    """Target column of each longitude, for `n_lon` equal columns starting at 0E."""
    return np.clip((np.mod(lon, 360.0) * (n_lon / 360.0)).astype(np.int64), 0, n_lon - 1)


def locate(lon: np.ndarray, lat: np.ndarray, lat_edges: np.ndarray, n_lon: int) -> np.ndarray:
    """Flat index `row * n_lon + col` of the target cell containing each point."""
    row = np.clip(np.searchsorted(lat_edges, lat, side="right") - 1, 0, lat_edges.size - 2)
    return row * n_lon + column(lon, n_lon)


def aligned_weights(lon_c, lat_c, xc, ra, lat_edges, n_lon):
    """Exact `(pixel, cell, area)` for a tile of lon/lat-rectangle cells, else None.

    Each cell lies in one target column (LLC longitude edges sit on 1/48 degree
    multiples) and in one or two target rows; a straddling cell is split by its
    spherical area, i.e. by sin(latitude).
    """
    sw, se = (lon_c[:-1, :-1], lat_c[:-1, :-1]), (lon_c[:-1, 1:], lat_c[:-1, 1:])
    nw, ne = (lon_c[1:, :-1], lat_c[1:, :-1]), (lon_c[1:, 1:], lat_c[1:, 1:])

    def same_lon(a, b):
        return np.abs(wrap180(a[0] - b[0])) < ALIGN_TOL

    def same_lat(a, b):
        return np.abs(a[1] - b[1]) < ALIGN_TOL

    j_meridian = same_lon(sw, nw) & same_lon(se, ne) & same_lat(sw, se) & same_lat(nw, ne)
    i_meridian = same_lon(sw, se) & same_lon(nw, ne) & same_lat(sw, nw) & same_lat(se, ne)
    if not (j_meridian.all() or i_meridian.all()):
        return None

    xc = np.asarray(xc, np.float64)
    half = 0.5 * np.abs(wrap180(ne[0] - sw[0]))
    col = column(xc, n_lon)
    if not (np.array_equal(col, column(xc - half + ALIGN_TOL, n_lon))
            and np.array_equal(col, column(xc + half - ALIGN_TOL, n_lon))):
        return None

    south, north = np.minimum(sw[1], ne[1]), np.maximum(sw[1], ne[1])
    ny = lat_edges.size - 1
    row_s = np.clip(np.searchsorted(lat_edges, south + EDGE_TOL, side="right") - 1, 0, ny - 1)
    row_n = np.clip(np.searchsorted(lat_edges, north - EDGE_TOL, side="left") - 1, 0, ny - 1)
    if np.any(row_n - row_s > 1):
        return None
    split = (row_n > row_s).ravel()
    sin = np.sin(np.deg2rad(np.stack([south, north, lat_edges[row_n]])))
    frac = np.where(row_n > row_s, (sin[2] - sin[0]) / (sin[1] - sin[0]), 1.0).ravel()

    ra, col = np.asarray(ra, np.float64).ravel(), col.ravel()
    pixel = np.arange(ra.size)
    cell_s = row_s.ravel() * n_lon + col
    cell_n = row_n.ravel() * n_lon + col
    return (
        np.concatenate([pixel, pixel[split]]).astype(np.int32),
        np.concatenate([cell_s, cell_n[split]]).astype(np.int32),
        np.concatenate([ra * frac, (ra * (1 - frac))[split]]).astype(np.float32),
    )


def subcell_weights(lon_c, lat_c, ra, lat_edges, n_lon, n_sub: int = SUBCELLS):
    """`(pixel, cell, area)` sharing each cell's area among its n_sub x n_sub sub-cells.

    Sub-cell centres are bilinear between the four corner unit vectors, and
    each carries `rA / n_sub**2` to the target cell it falls in.
    """
    v = unit_vector(np.asarray(lon_c, np.float64), np.asarray(lat_c, np.float64))
    c00, c01, c10, c11 = v[:, :-1, :-1], v[:, :-1, 1:], v[:, 1:, :-1], v[:, 1:, 1:]
    offsets = (np.arange(n_sub) + 0.5) / n_sub
    cells = []
    for t in offsets:
        for s in offsets:
            p = (1 - s) * (1 - t) * c00 + s * (1 - t) * c01 + (1 - s) * t * c10 + s * t * c11
            cells.append(locate(*lonlat_from_vector(p), lat_edges, n_lon).ravel().astype(np.int32))
    n_cells = (lat_edges.size - 1) * n_lon
    n_pix = cells[0].size
    keys = (np.arange(n_pix, dtype=np.int64)[None] * n_cells + np.stack(cells)).ravel()
    keys, counts = np.unique(keys, return_counts=True)
    pixel = keys // n_cells
    area = np.asarray(ra, np.float64).ravel()[pixel] * counts / n_sub**2
    return pixel.astype(np.int32), (keys % n_cells).astype(np.int32), area.astype(np.float32)


def tile_matrix(pixel, cell, area, n_pixels: int) -> tuple[np.ndarray, sp.csr_matrix]:
    """Target cells touched by one tile, and the `(cells, pixels)` area matrix."""
    cells, local = np.unique(cell, return_inverse=True)
    matrix = sp.csr_matrix((area, (local, pixel)), shape=(cells.size, n_pixels))
    return cells, matrix


def apply_weights(matrix: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """`[levels, cells]` area sums of `x` (`[levels, pixels]` or `[pixels]`)."""
    if x.ndim == 1:
        return matrix @ x
    return (matrix @ np.ascontiguousarray(x.reshape(x.shape[0], -1).T)).T


def nearest(points: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of, and chord distance to, the nearest candidate for each `[3, n]` point."""
    distance, index = cKDTree(candidates.T).query(points.T)
    return index, distance


def center_inplace(a: np.ndarray, axis: int) -> np.ndarray:
    """Average each value with the next one along `axis`; the last one is kept.

    U on `i_g` (west faces) centres along i, V on `j_g` (south faces) along j.
    The last column/row needs the next face's first one, so it copies its own
    value instead, like xgcm's `boundary="extend"` in llc_cross_face.ipynb.
    NaN (land) becomes 0 first: no flow through a closed face.
    """
    np.nan_to_num(a, copy=False)
    head = [slice(None)] * a.ndim
    tail = [slice(None)] * a.ndim
    head[axis], tail[axis] = slice(None, -1), slice(1, None)
    a[tuple(head)] += a[tuple(tail)]
    a[tuple(head)] *= np.float32(0.5)
    return a


def rotate_to_east_north(
    uc: np.ndarray, vc: np.ndarray, cs: np.ndarray, sn: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centred model-grid (u, v) -> (east, north), as in `rotate_to_geo`."""
    return uc * cs - vc * sn, uc * sn + vc * cs


def bottom_hfac(hfac: np.ndarray) -> np.ndarray:
    """Open-water fraction of each column's deepest wet cell; `hfac` is `[k, ...]`.

    1 where the column is full depth to the last level, the partial value where
    the bottom cell is partial, 0 on land.
    """
    hfac = np.nan_to_num(hfac)
    wet = hfac > 0
    last = hfac.shape[0] - 1 - np.argmax(wet[::-1], axis=0)
    value = np.take_along_axis(hfac, last[None], axis=0)[0]
    return np.where(wet.any(axis=0), value, np.float32(0)).astype(np.float32)


def repair_candidates(t: int, n_time: int, window: int = 48) -> list[int]:
    """Hours to copy a bad hour `t` from, in order: earlier first, then later (for hour 0)."""
    return [*range(t - 1, max(t - window, -1), -1), *range(t + 1, min(t + window, n_time))]
