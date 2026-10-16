"""LLC4320 coarsening helpers: box sums, C-grid velocity centring and rotation,
bottom partial cells, and stitching native faces into a lat/lon layout.

Faces 0-5 have +i east and +j north. Faces 7-12 are rotated: +i south, +j east.
Face 6 is the Arctic cap. See docs/10-15-26-coarsen_LLC_plan.md.
"""

from __future__ import annotations

import numpy as np

N_FACES = 13
ARCTIC_FACE = 6
ROTATED_FACES = (7, 8, 9, 10, 11, 12)
#: West -> east facets, each listed south -> north. Checked against raw XC/YC.
FACETS = ((0, 1, 2), (3, 4, 5), (9, 8, 7), (12, 11, 10))


def box_sum(x: np.ndarray, b: int) -> np.ndarray:
    """Sum non-overlapping b x b blocks over the last two axes, in float64."""
    *lead, ny, nx = x.shape
    if ny % b or nx % b:
        raise ValueError(f"shape {x.shape[-2:]} is not divisible by box {b}")
    # Rows first (contiguous), then columns: much faster than one 2-axis reduction.
    rows = x.reshape(*lead, ny, nx // b, b).sum(axis=-1, dtype=np.float64)
    return rows.reshape(*lead, ny // b, b, nx // b).sum(axis=-2)


def weighted_box_sum(x, area: np.ndarray, wet: np.ndarray, b: int) -> np.ndarray:
    """`sum(x * area)` over the wet pixels of each b x b box; `x=1` gives the wet area."""
    xw = np.zeros(np.broadcast_shapes(np.shape(x), wet.shape), dtype=np.float32)
    np.multiply(x, area, out=xw, where=wet)
    return box_sum(xw, b)


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """`num / den` as float32, NaN where `den == 0` (all-land boxes)."""
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 0, casting="unsafe")
    return out


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


def unit_vector(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """`[3, ...]` Cartesian position on the unit sphere."""
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def lonlat_from_vector(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `unit_vector` for any non-zero `[3, ...]` vector, in degrees."""
    x, y, z = xyz
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    return lon.astype(np.float32), lat.astype(np.float32)


def fill_sources(t: int, n_time: int, missing: set[int]) -> list[int]:
    """Steps to average in place of a missing step `t`: t-1 and t+1, or one at an end."""
    sources = [s for s in (t - 1, t + 1) if 0 <= s < n_time]
    if any(s in missing for s in sources):
        raise ValueError(f"step {t} has a missing neighbour among {sources}")
    return sources


def faces_to_latlon(
    a: np.ndarray, partner: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch `[..., 13, n, n]` native faces into `([..., 3n, 4n], [..., n, n])`.

    Row 0 is the south edge and columns run west -> east. Faces 7-12 are turned
    with `rot90(k=1)`. For a direction-dependent pair like dxF/dyF, pass the
    other member as `partner` and its values are used on faces 7-12, where the
    model's x axis runs north-south. Vectors already in east/north need no
    partner. The Arctic cap (face 6) is returned unchanged as the second value.
    """
    def face(f: int) -> np.ndarray:
        src = partner if partner is not None and f in ROTATED_FACES else a
        out = src[..., f, :, :]
        return np.rot90(out, k=1, axes=(-2, -1)) if f in ROTATED_FACES else out

    columns = [np.concatenate([face(f) for f in facet], axis=-2) for facet in FACETS]
    return np.concatenate(columns, axis=-1), a[..., ARCTIC_FACE, :, :]
