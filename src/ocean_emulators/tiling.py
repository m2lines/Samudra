"""Tile geometry, catalogs, and the overlap blending operator.

This module is deliberately pure: it imports neither the trainer nor the
evaluator, so replay training, validation, and inference can all share one
blending implementation rather than growing three that drift apart.

Two ideas are kept strictly separate because they are orthogonal:

* **Overlap** -- pad-internal cells that more than one tile predicts. They are
  reconciled here, by the blender.
* **Halo** -- a padding type; a ring of cells a tile reads as context but does
  not own or predict. ``TileSpec`` carries the fields for it so that adding
  halos later is a data change, but nothing in this module supplies one yet.
"""

import dataclasses
import logging
import re
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)

# Sides are named for the axis and which end of it they sit at. ``j`` is the
# row / y axis and ``i`` is the column / x axis, matching the LLC index names
# and the ``(y, x)`` dimension order of a packed cache.
Side = Literal["jlo", "jhi", "ilo", "ihi"]
SIDES: tuple[Side, ...] = ("jlo", "jhi", "ilo", "ihi")

WindowKind = Literal["quintic", "kbd"]

#: Default Kaiser shape parameter for the ``kbd`` window. 6.0 is the value used
#: for the long MDCT window in AAC and is a reasonable general-purpose taper.
DEFAULT_KBD_BETA = 6.0

_FACE_IN_NAME = re.compile(r"face(\d+)")


# --------------------------------------------------------------------------
# 1D ramp profiles
# --------------------------------------------------------------------------


def ramp_profile(
    kind: WindowKind,
    width: int,
    *,
    beta: float = DEFAULT_KBD_BETA,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """A monotone 0 -> 1 taper of length ``width``, sampled at cell centres.

    Sampling at centres (``s = (k + 0.5) / width``) rather than endpoints is
    what makes the two sides of an overlap exactly complementary: the sample
    points of one tile's ascending ramp mirror the other's descending ramp.

    ``quintic``
        Smootherstep ``f(s) = 6s^5 - 15s^4 + 10s^3``. Satisfies
        ``f(1-s) = 1-f(s)``, so opposing weights sum to one and the operator is
        a true partition of unity; its first and second derivatives vanish at
        both ends, which is what removes the seam kink.
    ``kbd``
        Kaiser-Bessel-derived, the window STRATA stitches with. Opposing
        weights are *power* complementary (``w^2 + w'^2 = 1``) rather than
        sum complementary, so the normalization in :class:`TileBlender` is
        load-bearing here in a way it is not for ``quintic``.
    """
    if width < 1:
        raise ValueError(f"ramp width must be >= 1, got {width}")

    if kind == "quintic":
        s = (torch.arange(width, dtype=dtype) + 0.5) / width
        return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))

    if kind == "kbd":
        # A Kaiser window of length width+1 is symmetric, so its cumulative sum
        # satisfies cum[k] + cum[width-1-k] == cum[width]; taking the square
        # root therefore gives the Princen-Bradley property exactly.
        kaiser = torch.kaiser_window(
            width + 1, periodic=False, beta=beta, dtype=dtype
        )
        cumulative = torch.cumsum(kaiser, dim=0)
        return torch.sqrt(cumulative[:width] / cumulative[width])

    raise ValueError(f"Unknown window kind {kind!r}; expected 'quintic' or 'kbd'")


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TileSpec:
    """One tile's place in the world, in absolute per-face LLC indices.

    ``owned`` and ``halo`` are stored separately from ``shape`` so that a tile
    whose stored array is wider than the region it predicts -- the halo regime
    -- needs no new structure here.
    """

    tile_id: int
    dataset_index: int
    face: int
    i_start: int
    i_end: int
    j_start: int
    j_end: int
    #: Local ``(j0, j1, i0, i1)`` bounds of the predicted-and-owned region.
    owned: tuple[int, int, int, int]
    #: Per-side halo width in ``SIDES`` order; all zeros until halos land.
    halo: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.j_end - self.j_start, self.i_end - self.i_start)

    def __post_init__(self) -> None:
        if self.i_end <= self.i_start or self.j_end <= self.j_start:
            raise ValueError(f"Tile {self.tile_id} has an empty extent")
        j0, j1, i0, i1 = self.owned
        height, width = self.shape
        if not (0 <= j0 < j1 <= height and 0 <= i0 < i1 <= width):
            raise ValueError(
                f"Tile {self.tile_id} owned region {self.owned} is outside its "
                f"{height}x{width} extent"
            )


def _overlap_width(tile: TileSpec, other: TileSpec, side: Side) -> int:
    """Cells of ``tile``'s ``side`` that ``other`` also covers, else 0.

    A neighbour on an ``i`` side must also share ``j`` cells (and vice versa),
    otherwise the diagonal tile of a 2x2 group would be mistaken for an edge
    neighbour and taper the wrong strip.
    """
    if other.face != tile.face:
        return 0

    if side in ("ilo", "ihi"):
        along = (tile.j_start, tile.j_end, other.j_start, other.j_end)
        across = (tile.i_start, tile.i_end, other.i_start, other.i_end)
    else:
        along = (tile.i_start, tile.i_end, other.i_start, other.i_end)
        across = (tile.j_start, tile.j_end, other.j_start, other.j_end)

    tile_lo, tile_hi, other_lo, other_hi = along
    if other_lo >= tile_hi or other_hi <= tile_lo:
        return 0

    tile_lo, tile_hi, other_lo, other_hi = across
    if side in ("ilo", "jlo"):
        if other_lo >= tile_lo:
            return 0
        return max(0, min(other_hi, tile_hi) - tile_lo)
    if other_hi <= tile_hi:
        return 0
    return max(0, tile_hi - max(other_lo, tile_lo))


@dataclasses.dataclass(frozen=True)
class TileGroupLayout:
    """A set of tiles trained, blended, and advanced as one unit."""

    group_id: int
    tiles: tuple[TileSpec, ...]
    canonical_origin: tuple[int, int]
    canonical_shape: tuple[int, int]
    #: ``(tile_id, side)`` pairs with no neighbour. These keep weight 1: there
    #: is nothing to blend against, so tapering there would only throw the
    #: tile's own edge away and drive the denominator toward zero.
    exterior_sides: frozenset[tuple[int, Side]]
    #: ``(tile_id, side) -> overlap width in cells`` for sides with a neighbour.
    overlaps: dict[tuple[int, Side], int]

    @property
    def num_tiles(self) -> int:
        return len(self.tiles)

    def canonical_bounds(self, tile: TileSpec) -> tuple[int, int, int, int]:
        """``(j0, j1, i0, i1)`` of ``tile`` within the canonical grid."""
        origin_j, origin_i = self.canonical_origin
        return (
            tile.j_start - origin_j,
            tile.j_end - origin_j,
            tile.i_start - origin_i,
            tile.i_end - origin_i,
        )


def build_group_layout(tiles: Sequence[TileSpec], *, group_id: int = 0) -> TileGroupLayout:
    """Derive neighbours, overlap widths, and exterior sides from extents."""
    if not tiles:
        raise ValueError("A tile group needs at least one tile")
    faces = {tile.face for tile in tiles}
    if len(faces) != 1:
        raise ValueError(
            f"Group {group_id} spans faces {sorted(faces)}; cross-face groups "
            "need an explicit halo/rotation operator that does not exist yet."
        )
    if len({tile.tile_id for tile in tiles}) != len(tiles):
        raise ValueError(f"Group {group_id} has duplicate tile_ids")

    overlaps: dict[tuple[int, Side], int] = {}
    exterior: set[tuple[int, Side]] = set()
    for tile in tiles:
        height, width = tile.shape
        for side in SIDES:
            limit = width if side in ("ilo", "ihi") else height
            found = max(
                (_overlap_width(tile, other, side) for other in tiles if other is not tile),
                default=0,
            )
            if found == 0:
                exterior.add((tile.tile_id, side))
                continue
            if found > limit:
                raise ValueError(
                    f"Tile {tile.tile_id} side {side} overlaps {found} cells, "
                    f"more than its own extent of {limit}."
                )
            overlaps[(tile.tile_id, side)] = found

    origin = (min(t.j_start for t in tiles), min(t.i_start for t in tiles))
    shape = (
        max(t.j_end for t in tiles) - origin[0],
        max(t.i_end for t in tiles) - origin[1],
    )
    return TileGroupLayout(
        group_id=group_id,
        tiles=tuple(tiles),
        canonical_origin=origin,
        canonical_shape=shape,
        exterior_sides=frozenset(exterior),
        overlaps=overlaps,
    )


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


def resolve_face(data: xr.Dataset, *, name_hint: str | None = None) -> int:
    """Recover the LLC face, which older caches do not record.

    A global LLC coordinate is ``(face, i, j)`` but only ``i`` and ``j`` are
    stored, so this falls back through: an explicit ``llc_face`` attr written
    by the current cache builder, then the filename, then failure. Adjacency is
    never trusted to this -- see :func:`validate_tile_group`, which checks
    ``XC``/``YC`` directly and does not care about face at all.
    """
    face = data.attrs.get("llc_face")
    if face is not None:
        return int(face)
    if name_hint is not None:
        match = _FACE_IN_NAME.search(name_hint)
        if match is not None:
            logger.warning(
                "Cache %s has no llc_face attr; inferring face=%s from its name. "
                "Rebuild with the current builder to record it explicitly.",
                name_hint,
                match.group(1),
            )
            return int(match.group(1))
    raise ValueError(
        "Cannot determine the LLC face: the cache has no llc_face attr and no "
        "face could be parsed from its name. Pass `faces=` explicitly."
    )


def tile_spec_from_coords(
    *,
    tile_id: int,
    dataset_index: int,
    face: int,
    x: np.ndarray,
    y: np.ndarray,
) -> TileSpec:
    """Build a spec from a cache's absolute ``x``/``y`` index coordinates."""
    x = np.asarray(x)
    y = np.asarray(y)
    for name, values in (("x", x), ("y", y)):
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"Tile {tile_id} coordinate {name} must be 1-D and non-empty")
        steps = np.diff(values.astype(np.int64))
        if steps.size and not np.all(steps == 1):
            raise ValueError(
                f"Tile {tile_id} coordinate {name} is not contiguous and "
                "ascending; the catalog assumes unit-stride absolute indices."
            )
    return TileSpec(
        tile_id=tile_id,
        dataset_index=dataset_index,
        face=face,
        i_start=int(x[0]),
        i_end=int(x[-1]) + 1,
        j_start=int(y[0]),
        j_end=int(y[-1]) + 1,
        owned=(0, len(y), 0, len(x)),
    )


def tile_origins(extent: int, size: int, stride: int, *, lead: int) -> list[int]:
    """Start indices of ``extent // stride`` windows of ``size``, clamped inside.

    The interior origin is ``stride * c - lead``: the window reaches ``lead``
    cells back into the previous tile and ``lead`` cells forward into the next,
    so neighbouring windows share ``2 * lead`` cells. At the two ends that would
    run off the face, so the origin is clamped into ``[0, extent - size]``
    rather than the window being shrunk.

    Clamping rather than shrinking is the whole point: every window keeps the
    same shape, so one batched forward covers them all and
    :class:`TileBlender` -- which requires uniform tiles -- needs no change. The
    price is that the two end seams share ``3 * lead`` cells instead of
    ``2 * lead``, which the blender already handles because it stores an overlap
    width per ``(tile, side)`` rather than one width for the group.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if size < stride:
        raise ValueError(
            f"window size {size} is smaller than the stride {stride}; the "
            "windows would not cover the face"
        )
    if size > extent:
        raise ValueError(
            f"window size {size} does not fit in an extent of {extent}"
        )
    if extent % stride:
        raise ValueError(
            f"extent {extent} is not a whole number of {stride}-cell tiles"
        )
    return [
        min(max(0, stride * index - lead), extent - size)
        for index in range(extent // stride)
    ]


def face_tile_windows(
    face: int,
    *,
    extent: int = 4320,
    tile: int = 720,
    overlap: int = 16,
) -> list[tuple[int, int, int, int, int]]:
    """Cover one LLC face with uniformly shaped, overlapping tile windows.

    Returns ``(face, i_start, i_end, j_start, j_end)`` in row-major order, which
    is the shape ``data.llc_tiles`` takes, so a whole face becomes a replay
    group without a directory of prebuilt caches.

    ``tile`` is the stride -- the region a tile is the sole owner of -- and
    ``overlap`` is how far it reads into each neighbour, so the stored window is
    ``tile + 2 * overlap`` and adjacent tiles share ``2 * overlap`` cells. For
    the default LLC face that is 36 windows of 752x752 sharing 32-cell seams,
    which is exactly the geometry the `752-4-tile` caches already use.
    """
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    size = tile + 2 * overlap
    if 2 * overlap >= tile:
        raise ValueError(
            f"overlap {overlap} is at least half the {tile}-cell tile; "
            "neighbouring windows would overlap by more than a whole tile"
        )
    origins = tile_origins(extent, size, tile, lead=overlap)
    return [
        (face, i_start, i_start + size, j_start, j_start + size)
        for j_start in origins
        for i_start in origins
    ]


def tile_catalog_from_windows(
    windows: Sequence[tuple[int, int, int, int, int]],
) -> list[TileSpec]:
    """Build a catalog from configured `(face, i_start, i_end, j_start, j_end)`.

    This is the `data.llc_tiles` path: every tile is a window into one raw LLC
    store, so unlike :func:`build_tile_catalog` there is nothing to recover from
    the stores themselves -- they all share a location and their coordinates
    span the whole globe. Each tile owns its whole extent; overlaps between
    tiles are resolved by :func:`build_group_layout` exactly as for caches.
    """
    if not windows:
        raise ValueError("tile_catalog_from_windows needs at least one window")
    specs: list[TileSpec] = []
    for tile_id, window in enumerate(windows):
        if len(window) != 5:
            raise ValueError(
                f"Tile {tile_id} window must be "
                f"(face, i_start, i_end, j_start, j_end); got {window!r}"
            )
        face, i_start, i_end, j_start, j_end = (int(value) for value in window)
        specs.append(
            TileSpec(
                tile_id=tile_id,
                dataset_index=tile_id,
                face=face,
                i_start=i_start,
                i_end=i_end,
                j_start=j_start,
                j_end=j_end,
                owned=(0, j_end - j_start, 0, i_end - i_start),
            )
        )
    return specs


def build_tile_catalog(
    datasets: Sequence[xr.Dataset],
    *,
    faces: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    dataset_indices: Sequence[int] | None = None,
) -> list[TileSpec]:
    """Build a catalog from packed caches, keyed on their ``x``/``y`` coords."""
    if faces is not None and len(faces) != len(datasets):
        raise ValueError("faces must have one entry per dataset")
    if names is not None and len(names) != len(datasets):
        raise ValueError("names must have one entry per dataset")
    if dataset_indices is not None and len(dataset_indices) != len(datasets):
        raise ValueError("dataset_indices must have one entry per dataset")

    specs: list[TileSpec] = []
    for position, data in enumerate(datasets):
        # A raw packed cache stores absolute LLC indices as x/y, but
        # `with_lat_lon_coords` renames those dims to lon/lat when a DataSource
        # loads one. The values are the same integers either way.
        index_coords = _absolute_index_coords(data)
        if index_coords is None:
            raise ValueError(
                f"Cache at position {position} is missing coordinate(s) x/y "
                "(or lon/lat); the catalog is built from absolute LLC indices."
            )
        x_values, y_values = index_coords
        face = (
            faces[position]
            if faces is not None
            else resolve_face(
                data, name_hint=names[position] if names is not None else None
            )
        )
        specs.append(
            tile_spec_from_coords(
                tile_id=position,
                dataset_index=(
                    dataset_indices[position] if dataset_indices is not None else position
                ),
                face=face,
                x=x_values,
                y=y_values,
            )
        )
    return specs


def _spatial_dim_names(data: xr.Dataset) -> tuple[str, str]:
    """Return ``(j_dim, i_dim)``, which a loaded DataSource renames to lat/lon."""
    for j_name, i_name in (("y", "x"), ("lat", "lon")):
        if j_name in data.sizes and i_name in data.sizes:
            return j_name, i_name
    raise ValueError(
        f"Dataset has no y/x or lat/lon dimensions; found {sorted(data.sizes)}"
    )


def _absolute_index_coords(data: xr.Dataset) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(i, j)`` index arrays under either the x/y or lon/lat names."""
    for i_name, j_name in (("x", "y"), ("lon", "lat")):
        if i_name in data.coords and j_name in data.coords:
            return (
                data.coords[i_name].to_numpy(),
                data.coords[j_name].to_numpy(),
            )
    return None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TileGroupReport:
    """Data-quality findings that the caller decides how to act on.

    Structural problems raise instead of landing here: a mismatched channel
    list is a bug, whereas an unusable timestamp is a fact about the data that
    the seed sampler simply needs to skip.
    """

    #: Time indices where a probed variable was non-finite in a covered cell.
    nonfinite_times: tuple[int, ...] = ()
    #: ``(time index, variable) -> count of non-finite cells``.
    nonfinite_detail: dict[tuple[int, str], int] = dataclasses.field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return not self.nonfinite_times


def validate_tile_group(
    layout: TileGroupLayout,
    datasets: Sequence[xr.Dataset],
    *,
    probe_times: Sequence[int] = (0, -1),
    probe_vars: Sequence[str] = ("prognostic", "boundary"),
) -> TileGroupReport:
    """Check a group is coherent, and report unusable timestamps.

    The load-bearing check is that ``XC``/``YC`` agree *exactly* in every
    claimed overlap. That validates adjacency without trusting ``(face, i, j)``
    at all, which is why it will still work once tiles meet across LLC faces.
    """
    if len(datasets) != layout.num_tiles:
        raise ValueError(
            f"Group {layout.group_id} has {layout.num_tiles} tiles but "
            f"{len(datasets)} datasets were supplied"
        )
    by_tile = dict(zip(layout.tiles, datasets, strict=True))

    reference_tile, reference = next(iter(by_tile.items()))
    reference_times = reference["time"].to_numpy()
    for tile, data in by_tile.items():
        height, width = tile.shape
        j_dim, i_dim = _spatial_dim_names(data)
        if (data.sizes[j_dim], data.sizes[i_dim]) != (height, width):
            raise ValueError(
                f"Tile {tile.tile_id} declares shape {(height, width)} but its "
                f"cache is {(data.sizes[j_dim], data.sizes[i_dim])}"
            )
        times = data["time"].to_numpy()
        if times.shape != reference_times.shape or not np.array_equal(
            times, reference_times
        ):
            raise ValueError(
                f"Tile {tile.tile_id} time coordinate does not match tile "
                f"{reference_tile.tile_id}; a group advances on one cursor."
            )
        for attr in ("prognostic_channel_names_json", "boundary_channel_names_json"):
            if data.attrs.get(attr) != reference.attrs.get(attr):
                raise ValueError(
                    f"Tile {tile.tile_id} {attr} differs from tile "
                    f"{reference_tile.tile_id}; tiles must share a channel layout."
                )
        for stat in ("prognostic_mean", "prognostic_std", "boundary_mean", "boundary_std"):
            if stat not in data or stat not in reference:
                continue
            if not np.allclose(
                np.asarray(data[stat].to_numpy(), dtype=np.float64),
                np.asarray(reference[stat].to_numpy(), dtype=np.float64),
                equal_nan=True,
            ):
                raise ValueError(
                    f"Tile {tile.tile_id} {stat} differs from tile "
                    f"{reference_tile.tile_id}; tiles must share normalization."
                )

    _validate_overlap_agreement(layout, by_tile)
    return _probe_nonfinite(by_tile, probe_times=probe_times, probe_vars=probe_vars)


def _validate_overlap_agreement(
    layout: TileGroupLayout,
    by_tile: dict[TileSpec, xr.Dataset],
    *,
    grid_vars: Sequence[str] = ("XC", "YC", "rA"),
) -> None:
    tiles = list(by_tile)
    compared = 0
    for position, tile in enumerate(tiles):
        for other in tiles[position + 1 :]:
            if other.face != tile.face:
                continue
            j0 = max(tile.j_start, other.j_start)
            j1 = min(tile.j_end, other.j_end)
            i0 = max(tile.i_start, other.i_start)
            i1 = min(tile.i_end, other.i_end)
            if j1 <= j0 or i1 <= i0:
                continue
            shared = [
                name
                for name in grid_vars
                if name in by_tile[tile] and name in by_tile[other]
            ]
            if not shared:
                raise ValueError(
                    f"Tiles {tile.tile_id} and {other.tile_id} share no grid "
                    f"variable from {list(grid_vars)}, so their claimed overlap "
                    "cannot be verified. Validate against the raw packed caches, "
                    "which carry XC/YC/rA; a loaded DataSource does not."
                )
            for name in shared:
                left = _crop_absolute(by_tile[tile][name], tile, j0, j1, i0, i1)
                right = _crop_absolute(by_tile[other][name], other, j0, j1, i0, i1)
                if not np.array_equal(left, right, equal_nan=True):
                    worst = np.nanmax(np.abs(left - right))
                    raise ValueError(
                        f"Tiles {tile.tile_id} and {other.tile_id} disagree on "
                        f"{name} in their claimed overlap "
                        f"j[{j0}:{j1}) i[{i0}:{i1}) (max |diff| {worst}). "
                        "They are not the neighbours the catalog thinks they are."
                    )
                compared += 1

    if len(tiles) > 1 and compared == 0:
        raise ValueError(
            "No overlap was verified for a multi-tile group. Either the catalog "
            "found no adjacency, or no comparable grid variable was present."
        )


def _crop_absolute(
    array: xr.DataArray, tile: TileSpec, j0: int, j1: int, i0: int, i1: int
) -> np.ndarray:
    j_dim, i_dim = _spatial_dim_names(array.to_dataset(name="_"))
    return np.asarray(
        array.isel(
            {
                j_dim: slice(j0 - tile.j_start, j1 - tile.j_start),
                i_dim: slice(i0 - tile.i_start, i1 - tile.i_start),
            }
        ).to_numpy(),
        dtype=np.float64,
    )


def _probe_nonfinite(
    by_tile: dict[TileSpec, xr.Dataset],
    *,
    probe_times: Sequence[int],
    probe_vars: Sequence[str],
) -> TileGroupReport:
    bad_times: set[int] = set()
    detail: dict[tuple[int, str], int] = {}
    for tile, data in by_tile.items():
        length = int(data.sizes.get("time", 0))
        if length == 0:
            continue
        for raw_index in probe_times:
            index = raw_index if raw_index >= 0 else length + raw_index
            if not 0 <= index < length:
                continue
            for name in probe_vars:
                if name not in data:
                    continue
                values = np.asarray(
                    data[name].isel(time=index).to_numpy(), dtype=np.float64
                )
                count = int(np.count_nonzero(~np.isfinite(values)))
                if count:
                    bad_times.add(index)
                    key = (index, name)
                    detail[key] = detail.get(key, 0) + count
    if bad_times:
        logger.warning(
            "Tile group has non-finite values at time index/indices %s: %s. "
            "These must be excluded as seed times.",
            sorted(bad_times),
            detail,
        )
    return TileGroupReport(nonfinite_times=tuple(sorted(bad_times)), nonfinite_detail=detail)


# --------------------------------------------------------------------------
# Blending
# --------------------------------------------------------------------------


def ownership_boxes(layout: TileGroupLayout) -> tuple[tuple[int, int, int, int], ...]:
    """Local ``(j0, j1, i0, i1)`` of the cells each tile alone is scored on.

    Overlapping tiles cover some cells twice, so scoring every tile in full would
    weight those cells double and quietly bias any domain-wide metric toward the
    seams. Splitting each shared band down the middle gives every cell exactly one
    owner: the boxes tile the canonical grid with no gap and no overlap.

    Only sides that actually have a neighbour are trimmed -- an exterior side has
    nobody to hand its cells to, and trimming it would drop real domain.
    """
    boxes = []
    for tile in layout.tiles:
        height, width = tile.shape
        # A shared band of width W splits at its midpoint: the low tile keeps
        # W - W//2 of it and the high tile keeps W//2, which sums to W.
        j0 = layout.overlaps.get((tile.tile_id, "jlo"), 0)
        j0 -= j0 // 2
        j1 = height - layout.overlaps.get((tile.tile_id, "jhi"), 0) // 2
        i0 = layout.overlaps.get((tile.tile_id, "ilo"), 0)
        i0 -= i0 // 2
        i1 = width - layout.overlaps.get((tile.tile_id, "ihi"), 0) // 2
        if j1 <= j0 or i1 <= i0:
            raise ValueError(
                f"Tile {tile.tile_id} owns nothing after trimming its overlaps; "
                "the tiles overlap by more than their own extent."
            )
        boxes.append((j0, j1, i0, i1))
    return tuple(boxes)


def ownership_masks(
    layout: TileGroupLayout, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``[T, 1, H, W]`` indicator of each tile's owned cells.

    Zeroing the loss weight outside the owned box counts every physical cell once
    while leaving the tensors tile-shaped, so masks, area weights and metadata all
    stay exactly as they are.
    """
    height, width = layout.tiles[0].shape
    masks = torch.zeros(len(layout.tiles), 1, height, width, dtype=dtype)
    for index, (j0, j1, i0, i1) in enumerate(ownership_boxes(layout)):
        masks[index, :, j0:j1, i0:i1] = 1.0
    return masks


@dataclasses.dataclass(frozen=True)
class ReplayGroup:
    """The set of training datasets that one replay row advances together.

    ``layout``/``blender`` are ``None`` for an ungrouped row. That is not a
    degenerate case to be tidied away later -- it is what keeps single-cache
    replay bit-identical, since a one-tile group has no overlap to reconcile and
    every code path collapses to exactly what it did before.
    """

    group_id: int
    dataset_indices: tuple[int, ...]
    layout: TileGroupLayout | None = None
    blender: "TileBlender | None" = None

    @property
    def num_tiles(self) -> int:
        return len(self.dataset_indices)

    @property
    def is_grouped(self) -> bool:
        return self.layout is not None and self.num_tiles > 1


def build_replay_groups(
    *,
    num_sources: int,
    num_strides: int,
    grouped: bool,
    tiles: Sequence[TileSpec] | None = None,
    window: WindowKind = "quintic",
    ramp_width: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[ReplayGroup]:
    """Map replay rows onto training datasets.

    ``train_datasets`` is the flattened cross product ``[source x stride]``, so
    dataset index is ``source_index * num_strides + stride_index``. A group is
    therefore taken across sources *at a fixed stride*: tiles of one cluster must
    advance on one cursor, and two strides are two different clocks.

    When ``grouped`` is false the result is one single-tile group per dataset, in
    dataset order, so a group index and a dataset index are the same number and
    nothing downstream can tell the difference.
    """
    if num_sources < 1 or num_strides < 1:
        raise ValueError("num_sources and num_strides must both be >= 1")

    if not grouped or num_sources == 1:
        # Identity mapping: group i is dataset i.
        return [
            ReplayGroup(group_id=index, dataset_indices=(index,))
            for index in range(num_sources * num_strides)
        ]

    if tiles is None or len(tiles) != num_sources:
        raise ValueError(
            f"Grouped replay needs one TileSpec per source; got "
            f"{0 if tiles is None else len(tiles)} for {num_sources} sources."
        )

    groups: list[ReplayGroup] = []
    for stride_index in range(num_strides):
        members = tuple(
            dataclasses.replace(
                tile, dataset_index=position * num_strides + stride_index
            )
            for position, tile in enumerate(tiles)
        )
        layout = build_group_layout(members, group_id=stride_index)
        groups.append(
            ReplayGroup(
                group_id=stride_index,
                dataset_indices=tuple(tile.dataset_index for tile in members),
                layout=layout,
                blender=TileBlender(
                    layout, window=window, ramp_width=ramp_width, dtype=dtype
                ),
            )
        )
    return groups


def tile_window(
    layout: TileGroupLayout,
    tile: TileSpec,
    *,
    window: WindowKind = "quintic",
    kbd_beta: float = DEFAULT_KBD_BETA,
    ramp_width: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """One tile's ``[H, W]`` blending weight, tapered on every shared side.

    Module level rather than a `TileBlender` method because the distributed
    blender needs the same weights without owning a canonical grid: a rank
    holding nine of thirty-six tiles must still build the exact weights the
    single-process blender would have used, or the two disagree in the seams.
    """
    height, width = tile.shape
    profile_j = torch.ones(height, dtype=dtype)
    profile_i = torch.ones(width, dtype=dtype)
    spans: dict[Side, int] = {}
    for side in SIDES:
        overlap = layout.overlaps.get((tile.tile_id, side))
        if not overlap:
            # Exterior: no neighbour to hand off to, so hold weight at 1.
            continue
        # A ramp wider than the overlap is legitimate -- it is how STRATA
        # windows a whole tile rather than just its seam. Outside the
        # overlap the tile is the sole contributor, so normalization
        # returns its weight to one; only the shared cells see the change.
        spans[side] = overlap if ramp_width is None else ramp_width

    for axis, (lo, hi, extent) in (
        ("j", ("jlo", "jhi", height)),
        ("i", ("ilo", "ihi", width)),
    ):
        if spans.get(lo, 0) + spans.get(hi, 0) > extent:
            raise ValueError(
                f"Tile {tile.tile_id} ramps of {spans.get(lo, 0)} and "
                f"{spans.get(hi, 0)} cells collide on the {axis} axis "
                f"(extent {extent}); reduce ramp_width."
            )

    for side, span in spans.items():
        ramp = ramp_profile(window, span, beta=kbd_beta, dtype=torch.float64).to(
            dtype=dtype
        )
        if side == "jlo":
            profile_j[:span] = ramp
        elif side == "jhi":
            profile_j[height - span :] = ramp.flip(0)
        elif side == "ilo":
            profile_i[:span] = ramp
        else:
            profile_i[width - span :] = ramp.flip(0)
    return profile_j.unsqueeze(1) * profile_i.unsqueeze(0)


class TileBlender(torch.nn.Module):
    r"""Reconcile per-tile predictions in their overlaps.

    Implements the normalized overlap-add

    .. math::
        \bar{\delta}(g) = \frac{\sum_i c_i(g)\, w_i(g)\, \delta_i(g)}
                               {\sum_i c_i(g)\, w_i(g)}

    which is STRATA's :math:`S = D^{-1} P^{T} W` written for axis-aligned
    tiles: :math:`W` is the window, :math:`P^{T}` the scatter back to the
    canonical grid, and :math:`D` the incoming weight sum.

    The normalized form is evaluated even for ``quintic``, whose separable
    weights provably sum to one -- ``(1-f_x + f_x)(1-f_y + f_y) = 1``, so the
    four weights meeting at a corner sum to one. It costs nothing and is what
    lets exterior sides, missing contributors, coverage masks, and eventually
    cross-face topology work without special cases.

    ``blend`` returns tiles rather than a canonical field because both replay
    writeback and autoregressive inference need tiles back; ``to_canonical``
    exists for diagnostics and for writing a stitched product.
    """

    def __init__(
        self,
        layout: TileGroupLayout,
        *,
        window: WindowKind = "quintic",
        kbd_beta: float = DEFAULT_KBD_BETA,
        ramp_width: int | None = None,
        coverage: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        shapes = {tile.shape for tile in layout.tiles}
        if len(shapes) != 1:
            raise ValueError(
                f"TileBlender needs uniformly shaped tiles, got {sorted(shapes)}"
            )
        self.layout = layout
        self.window = window
        self.kbd_beta = kbd_beta
        self.ramp_width = ramp_width
        self.eps = eps
        self.tile_shape = layout.tiles[0].shape
        self.canonical_shape = layout.canonical_shape
        self._bounds = tuple(layout.canonical_bounds(tile) for tile in layout.tiles)

        weights = torch.stack(
            [self._tile_window(tile, dtype=dtype) for tile in layout.tiles]
        ).unsqueeze(1)  # [T, 1, H, W]

        if coverage is not None:
            coverage = coverage.to(dtype=dtype)
            if coverage.shape != weights.shape:
                raise ValueError(
                    f"coverage must be shaped {tuple(weights.shape)}, got "
                    f"{tuple(coverage.shape)}"
                )
            if torch.any(coverage < 0):
                raise ValueError("coverage must be non-negative")
            weights = weights * coverage

        # The denominator depends only on geometry and coverage, so it is
        # computed once here rather than on every blend.
        ones = torch.ones(
            1, layout.num_tiles, 1, *self.tile_shape, dtype=weights.dtype
        )
        denominator = self._scatter_with(ones, weights)
        uncovered = denominator <= 0
        if bool(uncovered.any()):
            raise ValueError(
                f"Group {layout.group_id} leaves {int(uncovered.sum())} canonical "
                "cells with zero total weight; the tiles do not cover their "
                "canonical grid, or a window tapers to zero with no neighbour."
            )

        self.register_buffer("weights", weights)
        self.register_buffer("denominator", denominator)

    def _tile_window(self, tile: TileSpec, *, dtype: torch.dtype) -> torch.Tensor:
        return tile_window(
            self.layout,
            tile,
            window=self.window,
            kbd_beta=self.kbd_beta,
            ramp_width=self.ramp_width,
            dtype=dtype,
        )

    def _scatter_with(
        self, tiles: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate weighted ``[B, T, C, H, W]`` onto ``[B, C, Hg, Wg]``."""
        batch, _, channels = tiles.shape[:3]
        height, width = self.canonical_shape
        total = torch.zeros(
            batch, channels, height, width, dtype=tiles.dtype, device=tiles.device
        )
        for index, (j0, j1, i0, i1) in enumerate(self._bounds):
            total[:, :, j0:j1, i0:i1] = (
                total[:, :, j0:j1, i0:i1] + tiles[:, index] * weights[index]
            )
        return total

    def _gather(self, canonical: torch.Tensor) -> torch.Tensor:
        """Read ``[B, C, Hg, Wg]`` back out as ``[B, T, C, H, W]``."""
        return torch.stack(
            [canonical[:, :, j0:j1, i0:i1] for (j0, j1, i0, i1) in self._bounds],
            dim=1,
        )

    @staticmethod
    def _as_batched(tiles: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tiles.ndim == 4:
            return tiles.unsqueeze(0), True
        if tiles.ndim == 5:
            return tiles, False
        raise ValueError(
            f"Expected [T, C, H, W] or [B, T, C, H, W], got {tuple(tiles.shape)}"
        )

    def _check_shape(self, tiles: torch.Tensor) -> None:
        expected_tiles = self.layout.num_tiles
        if tiles.shape[1] != expected_tiles:
            raise ValueError(
                f"Expected {expected_tiles} tiles, got {tiles.shape[1]}"
            )
        if tuple(tiles.shape[-2:]) != self.tile_shape:
            raise ValueError(
                f"Expected tiles shaped {self.tile_shape}, got "
                f"{tuple(tiles.shape[-2:])}"
            )

    def _weighted_mean(self, tiles: torch.Tensor) -> torch.Tensor:
        """The shared core: ``D^-1 P^T W`` applied to ``[B, T, C, H, W]``."""
        weights = self.weights.to(dtype=tiles.dtype, device=tiles.device)
        denominator = self.denominator.to(dtype=tiles.dtype, device=tiles.device)
        return self._scatter_with(tiles, weights) / denominator.clamp_min(self.eps)

    def to_canonical(self, tiles: torch.Tensor) -> torch.Tensor:
        """Assemble per-tile fields into one canonical grid (diagnostics/output)."""
        batched, squeeze = self._as_batched(tiles)
        self._check_shape(batched)
        canonical = self._weighted_mean(batched)
        return canonical.squeeze(0) if squeeze else canonical

    def blend(self, tiles: torch.Tensor) -> torch.Tensor:
        """Reconcile overlaps and scatter the consensus back to every tile.

        Applied to *residuals*: with the current state identical in every
        overlap -- which replay seeding and this scatter-back both preserve --
        blending residuals and blending full fields are equivalent, and the
        residual form avoids averaging a large shared background.
        """
        batched, squeeze = self._as_batched(tiles)
        self._check_shape(batched)
        blended = self._gather(self._weighted_mean(batched))
        return blended.squeeze(0) if squeeze else blended

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.blend(tiles)

    def extra_repr(self) -> str:
        return (
            f"tiles={self.layout.num_tiles}, window={self.window!r}, "
            f"tile_shape={self.tile_shape}, canonical={self.canonical_shape}"
        )


# --------------------------------------------------------------------------
# Distributed blending
# --------------------------------------------------------------------------


def contiguous_tile_blocks(num_tiles: int, world_size: int) -> list[list[int]]:
    """Split a square tile grid into ``world_size`` contiguous, equal blocks.

    Contiguity is an I/O decision, not a load-balancing one. Tiles overlap
    their neighbours, so a rank holding a compact block of them touches far
    fewer store chunks than one holding the same number scattered over the
    face -- and a compact block also has the shortest perimeter, which is
    exactly the surface the halo exchange has to move.

    Land is deliberately *not* balanced here. Per-rank loss normalization is
    what land would otherwise distort, and that is fixed exactly by scoring
    against a face-wide denominator instead, which no assignment can affect.
    """
    side = int(round(num_tiles**0.5))
    if side * side != num_tiles:
        raise ValueError(f"{num_tiles} tiles is not a square grid")
    if world_size < 1 or num_tiles % world_size:
        raise ValueError(
            f"{num_tiles} tiles do not divide evenly over {world_size} rank(s)"
        )
    per_rank = num_tiles // world_size

    # The squarest block whose area is `per_rank` and which tiles the grid.
    shape = None
    for rows in range(1, side + 1):
        if per_rank % rows or side % rows:
            continue
        cols = per_rank // rows
        if cols > side or side % cols:
            continue
        if shape is None or abs(rows - cols) < abs(shape[0] - shape[1]):
            shape = (rows, cols)
    if shape is None:
        raise ValueError(
            f"a {side}x{side} tile grid cannot be split into {world_size} equal "
            "rectangular blocks; choose a world size that divides the grid"
        )
    rows, cols = shape

    return [
        [
            (block_j * rows + dj) * side + (block_i * cols + di)
            for dj in range(rows)
            for di in range(cols)
        ]
        for block_j in range(side // rows)
        for block_i in range(side // cols)
    ]


@dataclasses.dataclass(frozen=True)
class HaloOp:
    """One tile's contribution to another tile, over the cells they share.

    ``src_box`` and ``dst_box`` are ``(j0, j1, i0, i1)`` in the source tile's
    and the destination tile's own local coordinates; they address the same
    physical cells.
    """

    src_tile: int
    dst_tile: int
    src_box: tuple[int, int, int, int]
    dst_box: tuple[int, int, int, int]

    @property
    def shape(self) -> tuple[int, int]:
        j0, j1, i0, i1 = self.dst_box
        return (j1 - j0, i1 - i0)


def halo_ops(layout: TileGroupLayout) -> list[HaloOp]:
    """Every ordered pair of overlapping tiles, in a deterministic order.

    Ordered pairs, not unordered: tile A contributes to tile B's cells *and*
    B contributes to A's, and once the two live on different ranks those are
    two separate messages.

    The order is ``(src_tile, dst_tile)`` ascending and is load-bearing twice.
    NCCL matches point-to-point operations by the order they are posted rather
    than by tag, so both ends must walk the same list; and accumulating in
    ascending source order is what makes the distributed result bit-identical
    to :meth:`TileBlender.blend`, which scatters tiles in catalog order.
    """
    ops: list[HaloOp] = []
    for src, source in enumerate(layout.tiles):
        for dst, destination in enumerate(layout.tiles):
            if src == dst or source.face != destination.face:
                continue
            j0 = max(source.j_start, destination.j_start)
            j1 = min(source.j_end, destination.j_end)
            i0 = max(source.i_start, destination.i_start)
            i1 = min(source.i_end, destination.i_end)
            if j1 <= j0 or i1 <= i0:
                continue
            ops.append(
                HaloOp(
                    src_tile=src,
                    dst_tile=dst,
                    src_box=(
                        j0 - source.j_start,
                        j1 - source.j_start,
                        i0 - source.i_start,
                        i1 - source.i_start,
                    ),
                    dst_box=(
                        j0 - destination.j_start,
                        j1 - destination.j_start,
                        i0 - destination.i_start,
                        i1 - destination.i_start,
                    ),
                )
            )
    return ops


class DistributedTileBlender(torch.nn.Module):
    r"""The same normalized overlap-add as :class:`TileBlender`, tile-sharded.

    A face is 36 tiles and one GPU holds at most a handful, so the canonical
    grid that :class:`TileBlender` scatters onto cannot be built on any single
    rank. It does not need to be. For a cell ``g`` of tile ``i`` the consensus
    is

    .. math:: \bar\delta(g) = \frac{\sum_k w_k(g)\,\delta_k(g)}{\sum_k w_k(g)}

    over the tiles ``k`` that cover ``g`` -- and every such ``k`` overlaps tile
    ``i``. So the sum can be accumulated directly on tile ``i``'s own extent,
    and the only thing that has to cross a rank boundary is the shared band
    itself. The denominator is pure geometry and is precomputed per local tile.

    The weights are 1 outside a tile's overlaps and form a partition of unity,
    so this is exact, not an approximation: the interior of every tile is
    untouched and only the seams move.

    ``tile_ranks[t]`` is the rank that owns tile ``t``. With one rank, or with
    every tile local, no collective is issued at all -- which is what keeps
    single-process replay and the tests working without a process group.
    """

    def __init__(
        self,
        layout: TileGroupLayout,
        *,
        tile_ranks: Sequence[int],
        rank: int,
        window: WindowKind = "quintic",
        kbd_beta: float = DEFAULT_KBD_BETA,
        ramp_width: int | None = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
        process_group: object | None = None,
    ) -> None:
        super().__init__()
        if len(tile_ranks) != layout.num_tiles:
            raise ValueError(
                f"tile_ranks has {len(tile_ranks)} entries for "
                f"{layout.num_tiles} tiles"
            )
        shapes = {tile.shape for tile in layout.tiles}
        if len(shapes) != 1:
            raise ValueError(
                f"DistributedTileBlender needs uniformly shaped tiles, got "
                f"{sorted(shapes)}"
            )

        self.layout = layout
        self.window = window
        self.kbd_beta = kbd_beta
        self.ramp_width = ramp_width
        self.eps = eps
        self.rank = rank
        self.tile_ranks = tuple(int(owner) for owner in tile_ranks)
        self.process_group = process_group
        self.tile_shape = layout.tiles[0].shape

        self.local_tiles: tuple[int, ...] = tuple(
            index for index, owner in enumerate(self.tile_ranks) if owner == rank
        )
        if not self.local_tiles:
            raise ValueError(f"Rank {rank} owns no tile of group {layout.group_id}")
        self._position = {tile: slot for slot, tile in enumerate(self.local_tiles)}

        all_weights = [
            tile_window(
                layout,
                tile,
                window=window,
                kbd_beta=kbd_beta,
                ramp_width=ramp_width,
                dtype=dtype,
            )
            for tile in layout.tiles
        ]
        weights = torch.stack(
            [all_weights[index] for index in self.local_tiles]
        ).unsqueeze(1)  # [Tlocal, 1, H, W]

        ops = halo_ops(layout)
        full_box = (0, self.tile_shape[0], 0, self.tile_shape[1])
        # Contributors to each local tile, in ascending source order with the
        # tile's own contribution at its own index. That order is what makes
        # the floating-point sum here identical to the catalog-order scatter
        # `TileBlender` performs on the canonical grid -- the numerator below
        # and this denominator both walk it.
        plan: dict[int, list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]]] = {
            tile: [(tile, full_box, full_box)] for tile in self.local_tiles
        }
        for op in ops:
            if self.tile_ranks[op.dst_tile] == rank:
                plan[op.dst_tile].append((op.src_tile, op.dst_box, op.src_box))
        self._plan = {
            tile: tuple(sorted(items)) for tile, items in plan.items()
        }

        # Sum of every contributor's weight, evaluated on each local tile's own
        # extent. Same number `TileBlender` divides by, read off the canonical
        # grid at this tile's box -- just never materialized globally.
        denominator = torch.zeros_like(weights)
        for tile, items in self._plan.items():
            slot = self._position[tile]
            for src, (dj0, dj1, di0, di1), (sj0, sj1, si0, si1) in items:
                denominator[slot, :, dj0:dj1, di0:di1] = (
                    denominator[slot, :, dj0:dj1, di0:di1]
                    + all_weights[src][sj0:sj1, si0:si1]
                )
        uncovered = denominator <= 0
        if bool(uncovered.any()):
            raise ValueError(
                f"Group {layout.group_id} leaves {int(uncovered.sum())} cells of "
                f"rank {rank}'s tiles with zero total weight; the tiles do not "
                "cover their canonical grid, or a window tapers to zero with no "
                "neighbour."
            )

        self.register_buffer("weights", weights)
        self.register_buffer("denominator", denominator)

        self.local_ops: tuple[HaloOp, ...] = tuple(
            op
            for op in ops
            if self.tile_ranks[op.dst_tile] == rank
            and self.tile_ranks[op.src_tile] == rank
        )
        self.recv_ops: tuple[HaloOp, ...] = tuple(
            op
            for op in ops
            if self.tile_ranks[op.dst_tile] == rank
            and self.tile_ranks[op.src_tile] != rank
        )
        self.send_ops: tuple[HaloOp, ...] = tuple(
            op
            for op in ops
            if self.tile_ranks[op.src_tile] == rank
            and self.tile_ranks[op.dst_tile] != rank
        )
        # The global cross-rank list, so every rank posts in the same order.
        self.exchange_ops: tuple[HaloOp, ...] = tuple(
            op for op in ops if self.tile_ranks[op.src_tile] != self.tile_ranks[op.dst_tile]
        )

    @property
    def exchanges(self) -> int:
        return len(self.send_ops) + len(self.recv_ops)

    def exchanged_elements(self) -> int:
        """Cells crossing a rank boundary per blend, per channel."""
        return sum(op.shape[0] * op.shape[1] for op in self.send_ops)

    def _as_batched(self, tiles: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tiles.ndim == 4:
            return tiles.unsqueeze(0), True
        if tiles.ndim == 5:
            return tiles, False
        raise ValueError(
            f"Expected [T, C, H, W] or [B, T, C, H, W], got {tuple(tiles.shape)}"
        )

    def _exchange(self, weighted: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
        """Trade shared bands with the other ranks; returns them keyed by op."""
        if not self.exchange_ops:
            return {}

        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "DistributedTileBlender has cross-rank overlaps but "
                "torch.distributed is not initialized."
            )

        batch, _, channels = weighted.shape[:3]
        received: dict[tuple[int, int], torch.Tensor] = {}
        outgoing: list[torch.Tensor] = []
        requests = []
        for tag, op in enumerate(self.exchange_ops):
            height, width = op.shape
            if self.tile_ranks[op.src_tile] == self.rank:
                sj0, sj1, si0, si1 = op.src_box
                payload = weighted[
                    :, self._position[op.src_tile], :, sj0:sj1, si0:si1
                ].contiguous()
                outgoing.append(payload)
                requests.append(
                    dist.P2POp(
                        dist.isend,
                        payload,
                        self.tile_ranks[op.dst_tile],
                        group=self.process_group,
                        tag=tag,
                    )
                )
            elif self.tile_ranks[op.dst_tile] == self.rank:
                buffer = torch.empty(
                    (batch, channels, height, width),
                    dtype=weighted.dtype,
                    device=weighted.device,
                )
                received[(op.src_tile, op.dst_tile)] = buffer
                requests.append(
                    dist.P2POp(
                        dist.irecv,
                        buffer,
                        self.tile_ranks[op.src_tile],
                        group=self.process_group,
                        tag=tag,
                    )
                )
        if requests:
            for work in dist.batch_isend_irecv(requests):
                work.wait()
        return received

    def blend(self, tiles: torch.Tensor) -> torch.Tensor:
        """Reconcile this rank's tiles against the whole group's overlaps.

        Takes and returns ``[Tlocal, C, H, W]`` (or a batched ``[B, Tlocal, ...]``)
        holding only the tiles this rank owns, in ``local_tiles`` order.

        Applied to *residuals*, exactly as :meth:`TileBlender.blend` is: the
        current state already agrees in every overlap, so blending residuals
        and blending full fields are the same thing, and the residual form
        avoids averaging a large shared background.
        """
        batched, squeeze = self._as_batched(tiles)
        if batched.shape[1] != len(self.local_tiles):
            raise ValueError(
                f"Rank {self.rank} owns {len(self.local_tiles)} tile(s) but got "
                f"{batched.shape[1]}"
            )
        if tuple(batched.shape[-2:]) != self.tile_shape:
            raise ValueError(
                f"Expected tiles shaped {self.tile_shape}, got "
                f"{tuple(batched.shape[-2:])}"
            )

        weights = self.weights.to(dtype=batched.dtype, device=batched.device)
        weighted = batched * weights
        received = self._exchange(weighted)

        # Walk the same per-destination, ascending-source plan the denominator
        # was built from, so the sum lands in the same order as TileBlender's
        # canonical scatter and the two agree bit for bit.
        total = torch.zeros_like(weighted)
        for tile, items in self._plan.items():
            slot = self._position[tile]
            for src, (dj0, dj1, di0, di1), (sj0, sj1, si0, si1) in items:
                if self.tile_ranks[src] == self.rank:
                    payload = weighted[
                        :, self._position[src], :, sj0:sj1, si0:si1
                    ]
                else:
                    payload = received[(src, tile)]
                total[:, slot, :, dj0:dj1, di0:di1] = (
                    total[:, slot, :, dj0:dj1, di0:di1] + payload
                )

        denominator = self.denominator.to(dtype=batched.dtype, device=batched.device)
        blended = total / denominator.clamp_min(self.eps)
        return blended.squeeze(0) if squeeze else blended

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.blend(tiles)

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, local_tiles={len(self.local_tiles)}/"
            f"{self.layout.num_tiles}, window={self.window!r}, "
            f"tile_shape={self.tile_shape}, exchanges={self.exchanges}"
        )
