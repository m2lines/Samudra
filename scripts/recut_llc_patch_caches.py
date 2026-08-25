#!/usr/bin/env python
"""Re-cut existing packed LLC caches into a different tiling, in place of a rebuild.

Going from the 1104x1104 tiles to 752x752 ones does not need the raw LLC store.
Every cell of the new tiling already sits in the old caches, and reading it back
out of them is far cheaper than reading the source again:

    from the caches   read  13 TB   (each cache chunk exactly once)
    from the raw store read  66 TB  (4 vars x 16 chunks + globe chunks per step)

The catch is that a new tile is not a sub-rectangle of one old tile. A 752 tile
with a 16-cell halo straddles the seams of the 1104 tiling, so each one is a
~748x748 bulk plus 28-wide slivers from up to three neighbours. This gathers
those pieces by absolute LLC index, the same way the chunk-first builder
assembles a tile out of source chunks.

Reads are chunk-first: each source cache is decompressed once per timestep and
scattered into every target tile that overlaps it, so adding target tiles costs
writes but not reads.

Channels are selected BY NAME, so dropping W is `--drop-channels W` and the
output keeps the surviving channels in their original order.

Boundary channels absent from the sources -- `oceFWflx`, which the 1104 caches
were built before -- are read from the raw LLC store in the same pass via
`--extra-boundary`. Doing it here rather than as an append costs one globe-chunk
read per timestep (~390 MiB) instead of a second rewrite of the whole boundary
array, and means the output is complete the first time.

Parallelism is by time. `--init` creates the stores once, then any number of
`--fill` jobs write disjoint time ranges; a zarr chunk is keyed by its time
index, so they never collide.
"""

from __future__ import annotations

import argparse
import json
import logging
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numcodecs
import numpy as np
import zarr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger(__name__)

CACHE_FORMAT = "llc-train-ready-v1"
#: Arrays whose 0 is a real value, so they must carry no fill value at all.
#: A wet mask with fill_value=0 has every land cell masked to NaN by xarray, and
#: NaN casts to True -- land silently becomes ocean.
NO_FILL = {"time", "x", "y", "prognostic_channel", "boundary_channel",
           "prognostic_mask", "boundary_mask"}


@dataclass(frozen=True)
class Window:
    """A tile's extent in absolute LLC index space."""

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

    def overlap(self, other: "Window") -> tuple[int, int, int, int] | None:
        """Shared region with `other`, in absolute indices, or None."""
        if self.face != other.face:
            return None
        i0, i1 = max(self.i_start, other.i_start), min(self.i_end, other.i_end)
        j0, j1 = max(self.j_start, other.j_start), min(self.j_end, other.j_end)
        return (i0, i1, j0, j1) if i1 > i0 and j1 > j0 else None


@dataclass
class SourceCache:
    """One existing packed cache, plus where it sits on the globe."""

    path: Path
    group: zarr.Group
    window: Window
    prognostic_names: list[str]
    boundary_names: list[str]

    @classmethod
    def open(cls, path: Path) -> "SourceCache":
        group = zarr.open_group(str(path), mode="r")
        attrs = group.attrs.asdict()
        missing = [
            key for key in ("llc_face", "llc_i_start", "llc_i_end",
                            "llc_j_start", "llc_j_end")
            if key not in attrs
        ]
        if missing:
            raise ValueError(f"{path.name} is missing {missing}; it cannot be placed")
        return cls(
            path=path,
            group=group,
            window=Window(
                int(attrs["llc_face"]), int(attrs["llc_i_start"]),
                int(attrs["llc_i_end"]), int(attrs["llc_j_start"]),
                int(attrs["llc_j_end"]),
            ),
            prognostic_names=json.loads(attrs["prognostic_channel_names_json"]),
            boundary_names=json.loads(attrs["boundary_channel_names_json"]),
        )


def parse_tiles(raw: str) -> list[Window]:
    text = Path(raw).read_text() if Path(raw).is_file() else raw
    tiles = []
    for entry in json.loads(text):
        if len(entry) != 5:
            raise ValueError(
                f"Each tile must be [face, i_start, i_end, j_start, j_end]; got {entry}"
            )
        tiles.append(Window(*(int(v) for v in entry)))
    if not tiles:
        raise ValueError("No tiles given")
    shapes = {(t.height, t.width) for t in tiles}
    if len(shapes) != 1:
        raise ValueError(f"All tiles must share one shape; got {sorted(shapes)}")
    return tiles


def scatter(target: np.ndarray, tile: Window, block: np.ndarray, source: Window) -> None:
    """Copy the region `source` and `tile` share into `target`.

    `target` is `[channel, j, i]` over the tile; `block` is `[channel, j, i]`
    over the source cache.
    """
    shared = tile.overlap(source)
    if shared is None:
        return
    i0, i1, j0, j1 = shared
    target[
        :, j0 - tile.j_start:j1 - tile.j_start, i0 - tile.i_start:i1 - tile.i_start
    ] = block[
        :, j0 - source.j_start:j1 - source.j_start,
        i0 - source.i_start:i1 - source.i_start,
    ]


def coverage_report(tiles: list[Window], sources: list[SourceCache]) -> None:
    """Refuse to build a tile the sources cannot completely fill."""
    for tile in tiles:
        covered = np.zeros((tile.height, tile.width), dtype=bool)
        contributors = 0
        for source in sources:
            shared = tile.overlap(source.window)
            if shared is None:
                continue
            contributors += 1
            i0, i1, j0, j1 = shared
            covered[
                j0 - tile.j_start:j1 - tile.j_start,
                i0 - tile.i_start:i1 - tile.i_start,
            ] = True
        if not covered.all():
            raise SystemExit(
                f"Tile i[{tile.i_start}:{tile.i_end}) j[{tile.j_start}:{tile.j_end}) "
                f"is only {100 * covered.mean():.1f}% covered by the source caches. "
                "The missing cells are not in any of them, so this tiling cannot be "
                "cut from these sources -- it would need the raw store."
            )
        logger.info(
            "  tile i[%d:%d) j[%d:%d): fully covered by %d source cache(s)",
            tile.i_start, tile.i_end, tile.j_start, tile.j_end, contributors,
        )


def create(group, name, shape, chunks, dtype, dims, compressor=None):
    """Create one array with xarray-compatible fill-value semantics."""
    fill = None if name in NO_FILL or not np.issubdtype(np.dtype(dtype), np.floating) \
        else float("nan")
    array = group.create_dataset(
        name, shape=shape, chunks=chunks, dtype=dtype,
        compressor=compressor, overwrite=True, fill_value=fill,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    return array


def init_store(
    path: Path, tile: Window, *, args, sources: list[SourceCache],
    prognostic_keep: list[int], boundary_keep: list[int],
) -> None:
    """Create one output cache and fill in everything that does not vary in time."""
    reference = sources[0]
    n_time = reference.group["prognostic"].shape[0]
    n_prog = len(prognostic_keep)
    n_bound = len(boundary_keep) + len(args.extra_boundary)
    dtype = reference.group["prognostic"].dtype
    compressor = numcodecs.Blosc(
        cname=args.compressor, clevel=args.compression_level,
        shuffle=getattr(numcodecs.Blosc, args.shuffle.upper()),
    )
    group = zarr.open_group(str(path), mode="w")

    create(group, "prognostic", (n_time, n_prog, tile.height, tile.width),
           (args.time_chunk, n_prog, tile.height, tile.width), dtype,
           ("time", "prognostic_channel", "y", "x"), compressor)
    create(group, "boundary", (n_time, n_bound, tile.height, tile.width),
           (args.time_chunk, n_bound, tile.height, tile.width), dtype,
           ("time", "boundary_channel", "y", "x"), compressor)

    # Stats are per channel and identical across tiles; carry the kept ones over.
    # An extra boundary variable has no entry in the source cache, so its mean
    # and std come from the LLC statistics store -- which must already contain
    # it (`notebooks/LLC_add_mean_std.py --vars <name>` writes it in place).
    extra_stats = {}
    if args.extra_boundary:
        for stat, store in (("mean", args.means), ("std", args.stds)):
            stats_group = zarr.open_group(str(store), mode="r")
            for name in args.extra_boundary:
                if name not in stats_group:
                    raise SystemExit(
                        f"{store} has no `{name}`. Add it first with\n"
                        f"  uv run notebooks/LLC_add_mean_std.py --vars {name}\n"
                        "otherwise the cache would carry a channel the trainer "
                        "cannot normalize."
                    )
                extra_stats[(stat, name)] = float(np.asarray(stats_group[name][...]).ravel()[0])
    for prefix, keep in (("prognostic", prognostic_keep), ("boundary", boundary_keep)):
        for stat in ("mean", "std"):
            values = reference.group[f"{prefix}_{stat}"][:][keep]
            if prefix == "boundary" and args.extra_boundary:
                values = np.concatenate([values, np.asarray(
                    [extra_stats[(stat, n)] for n in args.extra_boundary],
                    dtype=values.dtype)])
            create(group, f"{prefix}_{stat}", values.shape, values.shape,
                   values.dtype, (f"{prefix}_channel",))[:] = values

    # Masks and grid fields are spatial, so they get gathered like the data.
    for name, keep, dims in (
        ("prognostic_mask", prognostic_keep, ("prognostic_channel", "y", "x")),
        ("boundary_mask", boundary_keep, ("boundary_channel", "y", "x")),
    ):
        out = np.zeros((len(keep), tile.height, tile.width),
                       dtype=reference.group[name].dtype)
        for source in sources:
            if tile.overlap(source.window) is not None:
                scatter(out, tile, source.group[name][:][keep], source.window)
        if name == "boundary_mask" and args.extra_boundary:
            # Every boundary field shares the surface wet mask.
            out = np.concatenate(
                [out] + [out[:1]] * len(args.extra_boundary), axis=0)
        create(group, name, out.shape, out.shape, out.dtype, dims)[:] = out

    for name in ("XC", "YC", "rA"):
        if name not in reference.group:
            continue
        out = np.zeros((1, tile.height, tile.width),
                       dtype=reference.group[name].dtype)
        for source in sources:
            if tile.overlap(source.window) is not None:
                scatter(out, tile, source.group[name][:][None], source.window)
        create(group, name, out.shape[1:], out.shape[1:], out.dtype,
               ("y", "x"))[:] = out[0]

    times = reference.group["time"][:]
    create(group, "time", times.shape, (min(len(times), 1024),), times.dtype,
           ("time",))[:] = times
    group["time"].attrs.update(
        {k: v for k, v in reference.group["time"].attrs.asdict().items()
         if k != "_ARRAY_DIMENSIONS"}
    )
    create(group, "y", (tile.height,), (tile.height,), "i2", ("y",))[:] = np.arange(
        tile.j_start, tile.j_end, dtype=np.int16)
    create(group, "x", (tile.width,), (tile.width,), "i2", ("x",))[:] = np.arange(
        tile.i_start, tile.i_end, dtype=np.int16)
    create(group, "prognostic_channel", (n_prog,), (n_prog,), "i4",
           ("prognostic_channel",))[:] = np.arange(n_prog, dtype=np.int32)
    create(group, "boundary_channel", (n_bound,), (n_bound,), "i4",
           ("boundary_channel",))[:] = np.arange(n_bound, dtype=np.int32)

    attrs = dict(reference.group.attrs.asdict())
    attrs.update({
        "cache_format": CACHE_FORMAT,
        "built_by": "recut_llc_patch_caches.py",
        "recut_from": str(reference.path.parent),
        "prognostic_channel_count": n_prog,
        "boundary_channel_count": n_bound,
        "prognostic_channel_names_json": json.dumps(
            [reference.prognostic_names[i] for i in prognostic_keep]),
        "boundary_channel_names_json": json.dumps(
            [reference.boundary_names[i] for i in boundary_keep]
            + list(args.extra_boundary)),
        "llc_face": tile.face,
        "llc_i_start": tile.i_start, "llc_i_end": tile.i_end,
        "llc_j_start": tile.j_start, "llc_j_end": tile.j_end,
        "halo": args.halo,
    })
    group.attrs.update(attrs)
    zarr.consolidate_metadata(str(path))


def fill(
    tiles: list[Window], stores, sources: list[SourceCache], *, args,
    prognostic_keep: list[int], boundary_keep: list[int], start: int, stop: int,
) -> None:
    dtype = sources[0].group["prognostic"].dtype
    n_bound = len(boundary_keep) + len(args.extra_boundary)
    buffers = {
        index: (
            np.zeros((len(prognostic_keep), tile.height, tile.width), dtype=dtype),
            np.zeros((n_bound, tile.height, tile.width), dtype=dtype),
        )
        for index, tile in enumerate(tiles)
    }
    extra_arrays = {}
    if args.extra_boundary:
        raw = zarr.open_group(str(args.raw_store), mode="r")
        extra_arrays = {name: raw[name] for name in args.extra_boundary}
    started = _time.perf_counter()
    for done, t in enumerate(range(start, stop), start=1):
        for source in sources:
            # Each source cache decompressed once, then scattered into every
            # target that overlaps it. This is the whole saving.
            prognostic = source.group["prognostic"][t][prognostic_keep]
            boundary = source.group["boundary"][t][boundary_keep]
            for index, tile in enumerate(tiles):
                if tile.overlap(source.window) is None:
                    continue
                scatter(buffers[index][0], tile, prognostic, source.window)
                # The buffer also holds the --extra-boundary channels appended
                # after the source ones; those come from the raw store below, so
                # scatter into a view of just the source-derived leading ones.
                scatter(buffers[index][1][:len(boundary_keep)], tile, boundary,
                        source.window)
            del prognostic, boundary
        for offset, (name, array) in enumerate(extra_arrays.items()):
            # One globe chunk holds every face, so this is read once per
            # timestep however many tiles want it.
            channel = len(boundary_keep) + offset
            for index, tile in enumerate(tiles):
                plane = array[t, tile.face,
                              tile.j_start:tile.j_end, tile.i_start:tile.i_end]
                buffers[index][1][channel] = plane.astype(dtype)
        for index, _ in enumerate(tiles):
            stores[index]["prognostic"][t] = buffers[index][0]
            stores[index]["boundary"][t] = buffers[index][1]
        if done % args.log_every == 0 or t == stop - 1:
            rate = (_time.perf_counter() - started) / done
            logger.info("time %d/%d | %.1f s/step | eta %.1f h",
                        done, stop - start, rate, rate * (stop - start - done) / 3600)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", type=Path, required=True,
                   help="Directory of existing packed .zarr caches to cut from")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--tiles", required=True,
                   help="JSON [[face, i_start, i_end, j_start, j_end], ...] -- the "
                        "FULL extent including the halo")
    p.add_argument("--halo", type=int, default=16,
                   help="Recorded in the output attrs; does not change the extent")
    p.add_argument("--name-suffix", default="_trainval_ready")
    p.add_argument("--drop-channels", nargs="*", default=(),
                   help="Prognostic variables to leave out, by base name (e.g. W)")
    p.add_argument("--extra-boundary", nargs="*", default=(),
                   help="Boundary variables to append from the raw store, e.g. oceFWflx")
    p.add_argument("--raw-store", type=Path,
                   default=Path("/orcd/data/abodner/003/LLC4320/LLC4320"),
                   help="Source for --extra-boundary variables")
    p.add_argument("--means", type=Path,
                   default=Path("/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr"))
    p.add_argument("--stds", type=Path,
                   default=Path("/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr"))
    p.add_argument("--time-chunk", type=int, default=1)
    p.add_argument("--compressor", default="lz4")
    p.add_argument("--compression-level", type=int, default=5)
    p.add_argument("--shuffle", default="shuffle",
                   choices=("shuffle", "bitshuffle", "noshuffle"))
    p.add_argument("--init", action="store_true")
    p.add_argument("--fill", action="store_true")
    p.add_argument("--time-index-start", type=int, default=0)
    p.add_argument("--time-index-stop", type=int, default=None)
    p.add_argument("--time-splits", type=int, default=1)
    p.add_argument("--time-split-index", type=int, default=None)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.init and not args.fill:
        args.init = args.fill = True

    sources = [SourceCache.open(path)
               for path in sorted(args.source_root.glob("*.zarr"))]
    if not sources:
        raise SystemExit(f"No .zarr caches under {args.source_root}")
    tiles = parse_tiles(args.tiles)
    logger.info("%d source cache(s), %d target tile(s) of %dx%d",
                len(sources), len(tiles), tiles[0].height, tiles[0].width)
    coverage_report(tiles, sources)

    names = sources[0].prognostic_names
    dropped = set(args.drop_channels)
    prognostic_keep = [
        i for i, n in enumerate(names)
        if (n.rsplit("_", 1)[0] if n.rsplit("_", 1)[-1].isdigit() else n) not in dropped
    ]
    boundary_keep = list(range(len(sources[0].boundary_names)))
    logger.info(
        "channels: %d -> %d prognostic (dropped %s), %d -> %d boundary (added %s)",
        len(names), len(prognostic_keep), sorted(dropped) or "nothing",
        len(boundary_keep), len(boundary_keep) + len(args.extra_boundary),
        list(args.extra_boundary) or "nothing",
    )

    n_time = sources[0].group["prognostic"].shape[0]
    if args.time_split_index is not None:
        edges = np.linspace(0, n_time, args.time_splits + 1).astype(int)
        args.time_index_start = int(edges[args.time_split_index])
        stop = int(edges[args.time_split_index + 1])
        logger.info("Split %d/%d -> time [%d:%d)", args.time_split_index + 1,
                    args.time_splits, args.time_index_start, stop)
    else:
        stop = args.time_index_stop if args.time_index_stop is not None else n_time
    stop = min(stop, n_time)

    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = [
        args.output_root / (
            f"LLC4320_face{t.face}_i{t.i_start}-{t.i_end}_j{t.j_start}-{t.j_end}"
            f"{args.name_suffix}.zarr")
        for t in tiles
    ]
    if args.dry_run:
        for tile, path in zip(tiles, paths):
            logger.info("[dry-run] %s", path)
        logger.info("[dry-run] time [%d:%d) of %d", args.time_index_start, stop, n_time)
        return

    if args.init:
        for tile, path in zip(tiles, paths):
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite")
            logger.info("Init %s", path)
            init_store(path, tile, args=args, sources=sources,
                       prognostic_keep=prognostic_keep, boundary_keep=boundary_keep)
    if args.fill:
        stores = [zarr.open_group(str(path), mode="r+") for path in paths]
        logger.info("Filling time [%d:%d) into %d cache(s)",
                    args.time_index_start, stop, len(tiles))
        fill(tiles, stores, sources, args=args, prognostic_keep=prognostic_keep,
             boundary_keep=boundary_keep, start=args.time_index_start, stop=stop)
    logger.info("Done.")


if __name__ == "__main__":
    main()
