#!/usr/bin/env python
"""Build one train-ready cache covering a whole LLC4320 face, as a single zarr.

This is the flat-face counterpart to the per-tile builders in this directory.
Those exist to cut many small, overlapping caches out of the LLC store, and
carry a good deal of machinery for it. This one has a single job: take one face
(or a 720-aligned window of one), and write it out once, whole, with no
overlaps. Training recovers whatever tiling and halo it wants by slicing this
store on the CPU.

Two things make it more than a reshape:

Normalization happens BEFORE the float16 reduction.
    The per-tile builders store raw physical values as float16 and let the
    trainer z-score them at load time. float16 carries ~3 significant digits of
    RELATIVE precision, so a channel whose mean is far from zero spends its
    mantissa on the constant part. Salt is the casualty: it sits at ~34.6 psu,
    where the float16 step is 0.031 psu, against a standard deviation of
    0.41 psu at level 50 -- the quantization is 7.7% of a standard deviation,
    and deep salinity is stored as a staircase. Normalizing first puts every
    channel at O(1), where the float16 step is a uniform 0.098% of a standard
    deviation: a 79x precision gain on deep Salt, ~1x on Theta, U, V and Eta
    (already centred near zero), and it costs ~20% more bytes because the
    normalized field genuinely carries more information than the rounded one.

    The store therefore holds normalized values, and records the true means and
    stds it used (in float32, not float16 -- rounding the statistics would undo
    part of the precision this exists to buy). `DataSource.from_packed_dataset`
    reads the `pre_normalized` attr and skips the load-time normalization, so
    eval and `Normalize.unnormalize_*` still map back to physical units.

Boundary fields are read once per timestep, not once per tile.
    LLC's 2D arrays put the whole globe -- all 13 faces -- in a single chunk, so
    reading one face inflates 970 MB to hand back 75 MB. Reading that per tile
    would inflate it 36 times over. Each surface field is read once, normalized
    once across the face, and then sliced into all 36 tiles.

Land cells stay NaN, exactly as the source has them, so that missing data never
becomes an ordinary-looking 0 -- and 0 IS ordinary-looking here, being the mean
of a normalized channel. The loader collapses NaN to the masked fill value at
read time (`_normalize_and_mask_steps`), which is also the safety net for a NaN
on a WET cell: a variable's NaN set does not always agree with `mask_c`, since U
and V live on staggered faces (`i_g`, `j_g`) and carry their own land edges, so
the mask alone would not catch every one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cftime
import numcodecs
import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

CACHE_FORMAT = "llc-train-ready-v1"
#: LLC's horizontal chunk edge, and therefore this cache's. A 3D source array is
#: chunked (1, 51, 1, 720, 720), so one tile of one timestep is exactly one
#: source chunk per variable -- never a partial read, never a shared one.
TILE = 720
FACE_SIZE = 4320
#: Axis-name vocabulary, matching the source store's `_ARRAY_DIMENSIONS`.
LEVEL_DIMS = ("k", "k_p1", "k_l", "k_u", "lev")
ROW_DIMS = ("j", "j_g", "y", "lat")
COL_DIMS = ("i", "i_g", "x", "lon")

FLOAT_TYPES = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


# ---------------------------------------------------------------- domain


@dataclass(frozen=True)
class Domain:
    """The face, and the 720-aligned window of it, this cache covers."""

    face: int
    i_start: int
    i_end: int
    j_start: int
    j_end: int

    @property
    def height(self) -> int:
        return self.j_end - self.j_start

    @property
    def width(self) -> int:
        return self.i_end - self.i_start

    def tiles(self) -> list[tuple[int, int]]:
        """Absolute (j, i) origins of every 720x720 tile in the window."""
        return [
            (j, i)
            for j in range(self.j_start, self.j_end, TILE)
            for i in range(self.i_start, self.i_end, TILE)
        ]


def parse_domain(raw: str) -> Domain:
    """`[face, i_start, i_end, j_start, j_end]`, as the batch script writes it."""
    entry = json.loads(raw)
    if isinstance(entry, dict):
        values = [entry[k] for k in ("face", "i_start", "i_end", "j_start", "j_end")]
    elif len(entry) == 5:
        values = entry
    else:
        raise ValueError(
            f"--domain must be [face, i_start, i_end, j_start, j_end]; got {entry}"
        )
    domain = Domain(*(int(v) for v in values))
    for name, value in (
        ("i_start", domain.i_start), ("i_end", domain.i_end),
        ("j_start", domain.j_start), ("j_end", domain.j_end),
    ):
        if value % TILE:
            raise ValueError(
                f"--domain {name}={value} is not a multiple of {TILE}. The cache "
                "is chunked on the source's own 720 grid, so a misaligned window "
                "would make every read straddle two source chunks."
            )
    if not 0 <= domain.i_start < domain.i_end <= FACE_SIZE:
        raise ValueError(f"--domain i range {domain.i_start}:{domain.i_end} is invalid")
    if not 0 <= domain.j_start < domain.j_end <= FACE_SIZE:
        raise ValueError(f"--domain j range {domain.j_start}:{domain.j_end} is invalid")
    if not 0 <= domain.face < 13:
        raise ValueError(f"--domain face {domain.face} is not an LLC4320 face")
    return domain


# ------------------------------------------------------- source access


@dataclass(frozen=True)
class SourceArray:
    """A source array plus where each LLC axis sits in its dimension order."""

    array: zarr.Array
    time: int | None
    level: int | None
    face: int | None
    row: int
    col: int
    ndim: int

    @property
    def levels(self) -> int:
        return 1 if self.level is None else self.array.shape[self.level]

    def read(
        self, t: int | None, face: int, j0: int, j1: int, i0: int, i1: int
    ) -> np.ndarray:
        """`[level, j, i]`, or `[j, i]` when the array has no vertical axis."""
        index: list[object] = [slice(None)] * self.ndim
        if self.time is not None:
            index[self.time] = t
        if self.face is not None:
            index[self.face] = face
        index[self.row] = slice(j0, j1)
        index[self.col] = slice(i0, i1)
        return np.asarray(self.array[tuple(index)], dtype=np.float32)


def open_source_array(group: zarr.Group, name: str) -> SourceArray:
    if name not in group:
        raise KeyError(f"Source store has no array named {name!r}")
    array = group[name]
    dims = array.attrs.get("_ARRAY_DIMENSIONS")
    if dims is None:
        raise KeyError(f"Source array {name} has no _ARRAY_DIMENSIONS")

    def find(candidates) -> int | None:
        for position, dim in enumerate(dims):
            if dim in candidates:
                return position
        return None

    row, col = find(ROW_DIMS), find(COL_DIMS)
    if row is None or col is None:
        raise KeyError(f"Source array {name} has dims {dims}; expected j/i axes")
    return SourceArray(
        array=array,
        time=find(("time",)),
        level=find(LEVEL_DIMS),
        face=find(("face",)),
        row=row,
        col=col,
        ndim=len(dims),
    )


def split_channel(channel: str) -> tuple[str, int | None]:
    """`Theta_7` -> `("Theta", 7)`; `Eta` -> `("Eta", None)`."""
    base, _, level = channel.rpartition("_")
    if base and level.isdigit():
        return base, int(level)
    return channel, None


def stats_var_name(channel: str, stats: xr.Dataset) -> str:
    """`Theta_7` -> `Theta_lev_7`, `Eta` -> `Eta`, as the stats stores name them."""
    if channel in stats.data_vars:
        return channel
    base, level = split_channel(channel)
    candidate = f"{base}_lev_{level}"
    if level is None or candidate not in stats.data_vars:
        raise KeyError(f"No statistics for channel {channel!r} (tried {candidate!r})")
    return candidate


def expand_channels(
    requested: list[str], source: zarr.Group, levels: int
) -> list[str]:
    """Expand a bare 3D variable name into one channel per depth level.

    The batch script names variables, not channels -- `"Theta"` rather than 51
    entries -- because that is the readable thing to put in a job file. A name
    that already carries a level (`Theta_7`) is taken as-is, and a variable the
    source stores without a vertical axis (`Eta`) stays a single channel.
    """
    out: list[str] = []
    for name in requested:
        base, level = split_channel(name)
        if level is not None:
            out.append(name)
            continue
        array = open_source_array(source, name)
        if array.level is None:
            out.append(name)
            continue
        available = array.levels
        if levels > available:
            raise ValueError(
                f"{name} has {available} level(s) in the source, but {levels} "
                "were requested"
            )
        out.extend(f"{name}_{k}" for k in range(levels))
    duplicates = {c for c in out if out.count(c) > 1}
    if duplicates:
        raise ValueError(f"Duplicate channels after expansion: {sorted(duplicates)}")
    return out


# ----------------------------------------------------------- the store


def build_compressor(args: argparse.Namespace) -> numcodecs.Blosc:
    shuffle = {
        "shuffle": numcodecs.Blosc.SHUFFLE,
        "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
        "noshuffle": numcodecs.Blosc.NOSHUFFLE,
    }[args.shuffle]
    return numcodecs.Blosc(
        cname=args.compressor, clevel=args.compression_level, shuffle=shuffle
    )


def create_array(group, name, shape, chunks, dtype, dims, compressor=None,
                 fill_value="auto"):
    """Create one array with xarray-compatible fill-value semantics.

    `fill_value` is not cosmetic. xarray treats a zarr `fill_value` as
    `_FillValue` and masks any stored value equal to it, so zarr's default of 0
    is actively wrong here:

    * `time` would decode index 0 to NaT, which then overflows cftime in
      `_with_julian_time_coord` -- the trainer cannot open the store at all;
    * a wet mask would have every land cell (0) masked to NaN, and NaN casts to
      True, so land would silently become ocean.

    So: NaN for float arrays, and no fill value at all for integer coordinates
    and masks, where 0 is a legitimate value. The data arrays store 0 over land
    rather than NaN, so nothing is masked out of them by this.
    """
    if fill_value == "auto":
        fill_value = float("nan") if np.issubdtype(np.dtype(dtype), np.floating) else None
    array = group.create_dataset(
        name, shape=shape, chunks=chunks, dtype=dtype,
        compressor=compressor, overwrite=True, fill_value=fill_value,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    return array


def channel_stats(
    channels: list[str], means: xr.Dataset, stds: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std, in float32, in channel order.

    float32 deliberately, not the store's float16. These are the numbers the
    build divides by and the numbers eval multiplies back by; rounding them to
    float16 would put a 0.05% error on every unnormalized value and give back a
    slice of the precision this cache exists to gain.
    """
    mu = np.array(
        [float(means[stats_var_name(c, means)].values) for c in channels],
        dtype=np.float32,
    )
    sd = np.array(
        [float(stds[stats_var_name(c, stds)].values) for c in channels],
        dtype=np.float32,
    )
    bad = [c for c, s in zip(channels, sd) if not np.isfinite(s) or s == 0]
    if bad:
        raise ValueError(f"Channels with zero or non-finite std: {bad}")
    if not np.isfinite(mu).all():
        raise ValueError("Non-finite channel mean in the statistics store")
    return mu, sd


def init_store(
    path: Path,
    domain: Domain,
    *,
    args: argparse.Namespace,
    prognostic_channels: list[str],
    boundary_channels: list[str],
    prognostic_stats: tuple[np.ndarray, np.ndarray],
    boundary_stats: tuple[np.ndarray, np.ndarray],
    times: np.ndarray,
    time_attrs: dict,
    train_count: int,
    val_count: int,
    source: zarr.Group,
    pool: ThreadPoolExecutor | None,
) -> None:
    """Create the store with its full time extent and every static array."""
    float_dtype = FLOAT_TYPES[args.float_type]
    compressor = build_compressor(args)
    n_prog, n_bound = len(prognostic_channels), len(boundary_channels)
    n_time, height, width = len(times), domain.height, domain.width

    group = zarr.open_group(str(path), mode="w")

    create_array(group, "prognostic", (n_time, n_prog, height, width),
                 (args.time_chunk, n_prog, TILE, TILE), float_dtype,
                 ("time", "prognostic_channel", "y", "x"), compressor)
    create_array(group, "boundary", (n_time, n_bound, height, width),
                 (args.time_chunk, n_bound, TILE, TILE), float_dtype,
                 ("time", "boundary_channel", "y", "x"), compressor)

    # The statistics actually applied, so eval can invert them exactly.
    for prefix, (mu, sd) in (("prognostic", prognostic_stats),
                             ("boundary", boundary_stats)):
        create_array(group, f"{prefix}_mean", mu.shape, mu.shape, "f4",
                     (f"{prefix}_channel",))[:] = mu
        create_array(group, f"{prefix}_std", sd.shape, sd.shape, "f4",
                     (f"{prefix}_channel",))[:] = sd

    # Statics come tile by tile, on the same 720 grid as everything else: mask_c
    # is chunked (51, 1, 720, 720) and the grid fields (1, 720, 720), so a whole
    # face is 36 chunk-aligned reads per array rather than one 4320x4320 read
    # that straddles all of them.
    mask_source = open_source_array(source, "mask_c")
    grid_sources = {name: open_source_array(source, name) for name in ("XC", "YC", "rA")}

    prognostic_levels = [
        0 if level is None else level
        for _, level in (split_channel(c) for c in prognostic_channels)
    ]
    prognostic_mask = create_array(
        group, "prognostic_mask", (n_prog, height, width), (n_prog, TILE, TILE),
        "i1", ("prognostic_channel", "y", "x"))
    boundary_mask = create_array(
        group, "boundary_mask", (n_bound, height, width), (n_bound, TILE, TILE),
        "i1", ("boundary_channel", "y", "x"))
    grids = {
        name: create_array(group, name, (height, width), (TILE, TILE), "f4",
                           ("y", "x"))
        for name in grid_sources
    }

    def statics_for(origin: tuple[int, int]) -> None:
        j0, i0 = origin
        y = slice(j0 - domain.j_start, j0 - domain.j_start + TILE)
        x = slice(i0 - domain.i_start, i0 - domain.i_start + TILE)
        cube = mask_source.read(None, domain.face, j0, j0 + TILE, i0, i0 + TILE)
        # W sits on cell interfaces and borrows the mask of the cell below its
        # face, which is the convention the per-tile builders use.
        prognostic_mask[:, y, x] = cube[prognostic_levels].astype("i1")
        boundary_mask[:, y, x] = np.repeat(
            cube[0][None].astype("i1"), n_bound, axis=0
        )
        for name, array in grid_sources.items():
            grids[name][y, x] = array.read(None, domain.face, j0, j0 + TILE, i0, i0 + TILE)

    origins = domain.tiles()
    if pool is not None:
        list(pool.map(statics_for, origins))
    else:
        for origin in origins:
            statics_for(origin)

    create_array(group, "time", (n_time,), (min(n_time, 1024),), times.dtype,
                 ("time",))[:] = times
    group["time"].attrs.update(time_attrs)
    create_array(group, "y", (height,), (height,), "i4", ("y",))[:] = np.arange(
        domain.j_start, domain.j_end, dtype=np.int32)
    create_array(group, "x", (width,), (width,), "i4", ("x",))[:] = np.arange(
        domain.i_start, domain.i_end, dtype=np.int32)
    create_array(group, "prognostic_channel", (n_prog,), (n_prog,), "i4",
                 ("prognostic_channel",))[:] = np.arange(n_prog, dtype=np.int32)
    create_array(group, "boundary_channel", (n_bound,), (n_bound,), "i4",
                 ("boundary_channel",))[:] = np.arange(n_bound, dtype=np.int32)

    group.attrs.update({
        "cache_format": CACHE_FORMAT,
        "built_by": "build_llc_face_cache.py",
        "source_path": str(args.source),
        "means_path": str(args.means),
        "stds_path": str(args.stds),
        # The one attr that changes how this store must be read. See the module
        # docstring; `DataSource.from_packed_dataset` picks it up.
        "pre_normalized": True,
        "normalization": "z-score applied in float32 before the float16 cast",
        "land_fill": "NaN",
        "train_start": args.train_start, "train_end": args.train_end,
        "val_start": args.val_start, "val_end": args.val_end,
        "train_time_count": int(train_count), "val_time_count": int(val_count),
        "time_chunk": args.time_chunk,
        "float_type": args.float_type,
        "compression_codec": args.compressor,
        "compression_level": args.compression_level,
        "compression_shuffle": args.shuffle,
        "compression_target_vars": "prognostic,boundary",
        "prognostic_channel_count": n_prog,
        "boundary_channel_count": n_bound,
        "prognostic_channel_names_json": json.dumps(prognostic_channels),
        "boundary_channel_names_json": json.dumps(boundary_channels),
        "llc_face": domain.face,
        "llc_i_start": domain.i_start, "llc_i_end": domain.i_end,
        "llc_j_start": domain.j_start, "llc_j_end": domain.j_end,
        "spatial_chunk": TILE,
    })
    zarr.consolidate_metadata(str(path))


# ------------------------------------------------------------- filling


@dataclass(frozen=True)
class Plan:
    """Which source arrays to read, and which output channels each one feeds."""

    arrays: dict[str, SourceArray]
    #: base variable -> [(output channel index, source level), ...]
    volume: dict[str, list[tuple[int, int]]]
    surface: dict[str, list[tuple[int, int]]]

    @staticmethod
    def build(
        source: zarr.Group,
        groups: dict[int, list[str]],
    ) -> "Plan":
        arrays: dict[str, SourceArray] = {}
        volume: dict[str, list[tuple[int, int]]] = {}
        surface: dict[str, list[tuple[int, int]]] = {}
        for which, channels in groups.items():
            for position, channel in enumerate(channels):
                base, level = split_channel(channel)
                if base not in arrays:
                    arrays[base] = open_source_array(source, base)
                array = arrays[base]
                table = volume if array.level is not None else surface
                if array.level is not None and level is None:
                    raise ValueError(
                        f"{channel!r} has no level but {base} is depth-resolved"
                    )
                table.setdefault(base, []).append(
                    ((which, position), 0 if level is None else level)
                )
        return Plan(arrays=arrays, volume=volume, surface=surface)


def fill_time_range(
    store: zarr.Group,
    store_path: Path,
    source: zarr.Group,
    domain: Domain,
    *,
    args: argparse.Namespace,
    prognostic_channels: list[str],
    boundary_channels: list[str],
    prognostic_stats: tuple[np.ndarray, np.ndarray],
    boundary_stats: tuple[np.ndarray, np.ndarray],
    time_indices: np.ndarray,
    start: int,
    stop: int,
    pool: ThreadPoolExecutor | None,
) -> None:
    """Write `[start, stop)` of the output time axis."""
    float_dtype = FLOAT_TYPES[args.float_type]
    plan = Plan.build(
        source, {0: prognostic_channels, 1: boundary_channels}
    )
    stats = (prognostic_stats, boundary_stats)
    shapes = (len(prognostic_channels), len(boundary_channels))
    out = (store["prognostic"], store["boundary"])
    origins = domain.tiles()
    chunk_probe = store_path / "prognostic"

    # A surface field is normalized ONCE for the whole face and then sliced into
    # every tile, which is the read-once optimization this builder exists for.
    # That is only sound while every output channel fed by one source array
    # shares its statistics -- true whenever a variable appears in a single
    # channel group, and true in practice even across groups since both stats
    # come from the same store under the same name. Checked here rather than
    # assumed, because getting it wrong would mis-scale one of the two copies
    # and nothing downstream would flag it.
    for base, slots in plan.surface.items():
        stat_values = {
            (float(stats[which][0][position]), float(stats[which][1][position]))
            for (which, position), _ in slots
        }
        if len(stat_values) > 1:
            raise ValueError(
                f"{base} feeds channels with differing statistics {stat_values}; "
                "it cannot share one normalized plane. Normalize it per channel "
                "group instead."
            )

    def normalize(plane: np.ndarray, slot: tuple[int, int]) -> np.ndarray:
        """Physical float32 -> normalized float16, land left as NaN.

        NaN is kept rather than collapsed to 0 so that "no data here" stays
        distinguishable from "this cell is 0 standard deviations from the mean",
        which is a perfectly ordinary value for a normalized field. The loader
        resolves it: see `_normalize_and_mask_steps`.
        """
        which, position = slot
        mu, sd = stats[which]
        result = (plane - mu[position]) / sd[position]
        return result.astype(float_dtype, copy=False)

    def tile_job(
        origin: tuple[int, int],
        out_index: int,
        source_index: int,
        surface: dict[str, np.ndarray],
    ) -> None:
        j0, i0 = origin
        y = slice(j0 - domain.j_start, j0 - domain.j_start + TILE)
        x = slice(i0 - domain.i_start, i0 - domain.i_start + TILE)
        buffers = [
            np.full((n, TILE, TILE), np.nan, dtype=float_dtype) for n in shapes
        ]
        for base, slots in plan.volume.items():
            # One source chunk: (1, 51, 1, 720, 720) is exactly this read.
            cube = plan.arrays[base].read(
                source_index, domain.face, j0, j0 + TILE, i0, i0 + TILE
            )
            for slot, level in slots:
                which, position = slot
                buffers[which][position] = normalize(cube[level], slot)
            del cube
        for base, slots in plan.surface.items():
            plane = surface[base]
            for slot, _ in slots:
                which, position = slot
                buffers[which][position] = plane[y, x]
        # Boundary first: a fill is resumed by looking for the prognostic chunk,
        # so it must be the last thing written for a timestep-tile to count.
        out[1][out_index, :, y, x] = buffers[1]
        out[0][out_index, :, y, x] = buffers[0]

    def chunks_present(out_index: int) -> bool:
        return all(
            (chunk_probe / f"{out_index}.0.{(j - domain.j_start) // TILE}"
             f".{(i - domain.i_start) // TILE}").exists()
            for j, i in origins
        )

    started = _time.perf_counter()
    written = 0
    for offset, out_index in enumerate(range(start, stop)):
        source_index = int(time_indices[out_index])
        if args.skip_existing and chunks_present(out_index):
            continue

        # Every 2D source array holds all 13 faces in one chunk, so this read
        # inflates ~970 MB to hand back 75 MB. Once per timestep, then sliced
        # into all 36 tiles -- doing it per tile would pay that 36 times over.
        def read_surface(item):
            base, slots = item
            plane = plan.arrays[base].read(
                source_index, domain.face,
                domain.j_start, domain.j_end, domain.i_start, domain.i_end,
            )
            return base, normalize(plane, slots[0][0])

        items = list(plan.surface.items())
        surface = dict(
            pool.map(read_surface, items) if pool else map(read_surface, items)
        )

        def run(origin, out_index=out_index, source_index=source_index,
                surface=surface):
            tile_job(origin, out_index, source_index, surface)

        if pool is not None:
            list(pool.map(run, origins))
        else:
            for origin in origins:
                run(origin)
        del surface
        written += 1

        if written and (written % args.log_every == 0 or out_index == stop - 1):
            rate = (_time.perf_counter() - started) / written
            remaining = stop - start - offset - 1
            logger.info(
                "time %d/%d (store index %d, source index %d) | %.1f s/step | "
                "eta %.1f h", offset + 1, stop - start, out_index, source_index,
                rate, rate * remaining / 3600,
            )
    if args.skip_existing:
        logger.info("Wrote %d timestep(s); %d already present",
                    written, stop - start - written)


# ---------------------------------------------------------------- main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--means", type=Path, required=True)
    p.add_argument("--stds", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--domain", required=True,
                   help="[face, i_start, i_end, j_start, j_end]; 720-aligned")
    p.add_argument("--name-suffix", default="",
                   help="Appended to the cache directory name, before .zarr")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--prognostic-channels", default=None,
                   help="JSON list of variables or channels; defaults to "
                        "PROGNOSTIC_VARS['all']. A bare 3D name expands over "
                        "--levels depth levels.")
    p.add_argument("--boundary-channels", default=None,
                   help="JSON list; defaults to BOUNDARY_VARS['all_fw_noeta']")
    p.add_argument("--levels", type=int, default=51,
                   help="Depth levels a bare 3D variable name expands to")
    p.add_argument("--float-type", default="float16", choices=sorted(FLOAT_TYPES))
    p.add_argument("--time-chunk", type=int, default=1)
    p.add_argument("--compressor", default="zstd")
    p.add_argument("--compression-level", type=int, default=3)
    p.add_argument("--shuffle", default="shuffle",
                   choices=("shuffle", "bitshuffle", "noshuffle"))
    p.add_argument("--workers", type=int, default=36,
                   help="Threads for the per-tile read/compress/write jobs. One "
                        "tile is one job, so there is nothing for a thread past "
                        "the tile count to do.")
    p.add_argument("--init", action="store_true", help="Create the store and its statics")
    p.add_argument("--fill", action="store_true", help="Write a time range into an existing store")
    p.add_argument("--time-index-start", type=int, default=0)
    p.add_argument("--time-index-stop", type=int, default=None,
                   help="Exclusive; defaults to the end of the selected time axis")
    p.add_argument("--time-splits", type=int, default=1,
                   help="Split the time axis into this many equal ranges (for job arrays)")
    p.add_argument("--time-split-index", type=int, default=None,
                   help="Which split this job fills, 0-based. Overrides --time-index-*")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip timesteps whose chunks are all on disk already, so "
                        "a job killed at its wall clock can be resubmitted as-is. "
                        "Checks for presence, not integrity: a task killed mid-write "
                        "can leave one truncated chunk, so rerun that range without "
                        "this flag if a job died uncleanly.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.init and not args.fill:
        args.init = args.fill = True
    if args.time_chunk != 1:
        # A chunk spanning several timesteps stops being the unit a single task
        # owns: two array tasks filling adjacent ranges would read-modify-write
        # the same chunk concurrently and lose each other's timesteps. The
        # resume probe below also assumes one timestep per chunk.
        if args.time_splits > 1 or args.time_split_index is not None:
            raise ValueError(
                f"--time-chunk {args.time_chunk} cannot be combined with a "
                "split fill: adjacent tasks would write the same chunk "
                "concurrently. Use --time-chunk 1 for array jobs."
            )
        if args.skip_existing:
            raise ValueError(
                "--skip-existing assumes one timestep per chunk; it cannot be "
                f"used with --time-chunk {args.time_chunk}."
            )
    return args


def resolve_time_axis(args: argparse.Namespace, source: zarr.Group):
    """The selected timesteps, and the encoded time coordinate for the store."""
    data = xr.open_zarr(args.source, chunks={}, consolidated=False)
    train = data.sel(time=slice(args.train_start, args.train_end)).time.values
    val = data.sel(time=slice(args.val_start, args.val_end)).time.values
    if train.size == 0 or val.size == 0:
        raise ValueError("Train or val time selection is empty")
    combined = np.unique(np.concatenate([train, val]))
    lookup = {value: index for index, value in enumerate(data.time.values)}
    time_indices = np.array([lookup[value] for value in combined], dtype=np.int64)

    raw_time = source["time"]
    # Encode as integer offsets from the first sample, matching what the
    # per-tile builders write, so a cache from either tool decodes identically.
    decoded = cftime.num2date(
        raw_time[:][time_indices],
        raw_time.attrs["units"],
        calendar=raw_time.attrs.get("calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    origin = decoded[0]
    offsets = np.array([(d - origin).total_seconds() for d in decoded], dtype=np.int64)
    if np.all(offsets % 3600 == 0):
        times, unit = offsets // 3600, "hours"
    else:
        times, unit = offsets, "seconds"
    time_attrs = {
        "units": f"{unit} since {origin.strftime('%Y-%m-%d %H:%M:%S')}",
        "calendar": "proleptic_gregorian",
    }
    return time_indices, times, time_attrs, int(train.size), int(val.size)


def main(argv=None) -> None:
    args = parse_args(argv)
    domain = parse_domain(args.domain)
    source = zarr.open_group(str(args.source), mode="r")

    # We do our own threading, one tile per thread, so blosc must not also try
    # to thread inside a single chunk -- 36 compressions each spawning a pool
    # would oversubscribe the node several times over.
    numcodecs.blosc.use_threads = False

    prognostic_channels = expand_channels(
        json.loads(args.prognostic_channels) if args.prognostic_channels
        else list(PROGNOSTIC_VARS["all"]),
        source, args.levels,
    )
    boundary_channels = expand_channels(
        json.loads(args.boundary_channels) if args.boundary_channels
        else list(BOUNDARY_VARS["all_fw_noeta"]),
        source, args.levels,
    )

    means = xr.open_zarr(args.means)
    stds = xr.open_zarr(args.stds)
    prognostic_stats = channel_stats(prognostic_channels, means, stds)
    boundary_stats = channel_stats(boundary_channels, means, stds)

    time_indices, times, time_attrs, train_count, val_count = resolve_time_axis(
        args, source
    )

    origins = domain.tiles()
    tile_bytes = len(prognostic_channels) * TILE * TILE * np.dtype(
        FLOAT_TYPES[args.float_type]).itemsize
    logger.info(
        "face %d, window i[%d:%d] j[%d:%d] -> %d tile(s) of %dx%d",
        domain.face, domain.i_start, domain.i_end, domain.j_start, domain.j_end,
        len(origins), TILE, TILE,
    )
    logger.info("channels: %d prognostic, %d boundary",
                len(prognostic_channels), len(boundary_channels))
    logger.info("%d train + %d val = %d unique timesteps",
                train_count, val_count, time_indices.size)
    logger.info(
        "uncompressed %.2f GB per timestep (%.1f MB per prognostic chunk)",
        len(origins) * tile_bytes / 1e9, tile_bytes / 1e6,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    name = (f"LLC4320_face{domain.face}"
            f"_i{domain.i_start}-{domain.i_end}"
            f"_j{domain.j_start}-{domain.j_end}{args.name_suffix}.zarr")
    path = args.output_root / name

    if args.time_split_index is not None:
        # Contiguous, equal-as-possible ranges. Disjoint by construction, and a
        # zarr chunk is keyed by time index, so parallel fills never collide.
        if not 0 <= args.time_split_index < args.time_splits:
            raise ValueError(f"--time-split-index must be in [0, {args.time_splits})")
        edges = np.linspace(0, time_indices.size, args.time_splits + 1).astype(int)
        args.time_index_start = int(edges[args.time_split_index])
        stop = int(edges[args.time_split_index + 1])
        logger.info("Split %d/%d -> time [%d:%d)", args.time_split_index + 1,
                    args.time_splits, args.time_index_start, stop)
    else:
        stop = (args.time_index_stop if args.time_index_stop is not None
                else time_indices.size)
    stop = min(stop, time_indices.size)
    if args.time_index_start >= stop:
        raise ValueError(f"Empty time range [{args.time_index_start}:{stop})")

    if args.dry_run:
        logger.info("[dry-run] would write %s", path)
        logger.info("[dry-run] time [%d:%d) of %d", args.time_index_start, stop,
                    time_indices.size)
        return

    pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        if args.init:
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite")
            logger.info("Init %s", path)
            init_store(
                path, domain, args=args,
                prognostic_channels=prognostic_channels,
                boundary_channels=boundary_channels,
                prognostic_stats=prognostic_stats,
                boundary_stats=boundary_stats,
                times=times, time_attrs=time_attrs,
                train_count=train_count, val_count=val_count,
                source=source, pool=pool,
            )

        if args.fill:
            store = zarr.open_group(str(path), mode="r+")
            logger.info("Filling time [%d:%d) of %d", args.time_index_start, stop,
                        time_indices.size)
            fill_time_range(
                store, path, source, domain, args=args,
                prognostic_channels=prognostic_channels,
                boundary_channels=boundary_channels,
                prognostic_stats=prognostic_stats,
                boundary_stats=boundary_stats,
                time_indices=time_indices,
                start=args.time_index_start, stop=stop,
                pool=pool,
            )
            # Deliberately NOT consolidating here. Init already wrote .zmetadata
            # and a fill changes no metadata, only chunk contents -- so with
            # several array tasks writing one store, re-consolidating would have
            # them all rewriting the same file concurrently.
    finally:
        if pool is not None:
            pool.shutdown()
    logger.info("Done.")


if __name__ == "__main__":
    main()
