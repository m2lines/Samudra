#!/usr/bin/env python
"""Add boundary forcing channels to an existing packed LLC cache, in place.

One store then serves runs that use the new channel and runs that do not: the
loader resolves packed channels BY NAME, so a 5-channel cache read with
`boundary_vars_key: all` yields the original four and its original num_in, while
`all_fw` yields five. No second store, no timestamp join, and two experiments
that start at different times cannot interfere.

The `boundary` array cannot be extended in place -- its chunk spans the whole
channel axis, so 4 -> 5 channels rewrites every chunk, and zarr cannot change a
chunk shape after creation. So this builds the wider array alongside the old one
and swaps at the end:

    --init      create `boundary__v2` and friends. The live arrays are untouched,
                and the temporaries carry a DIFFERENT dimension name so anything
                opening the store with xarray meanwhile sees no size clash.
    --fill      write time ranges into `boundary__v2`. Parallel-safe: a zarr
                chunk is keyed by its time index.
    --finalize  move the old arrays out of the group, rename the new ones into
                place, update the attrs, re-consolidate. Seconds, not hours.

Only --finalize changes what a reader sees, so the long pass can run while a
training job reads the store; schedule the swap for when nothing is mid-epoch.

Existing channels are copied from the cache (a few MiB per timestep); new ones
come from the raw store, where a 2D field is chunked one-globe-per-timestamp and
so costs ~390 MiB per timestep however small the tile. That read dominates.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numcodecs
import numpy as np
import zarr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger(__name__)

CACHE_FORMAT = "llc-train-ready-v1-boundaryonly"
NO_FILL = {"time", "x", "y", "boundary_channel", "boundary_mask"}


def create(group, name, shape, chunks, dtype, dims, compressor=None):
    """Create one array with xarray-compatible fill-value semantics.

    A wet mask with fill_value=0 has every land cell masked to NaN by xarray,
    and NaN casts to True -- land silently becomes ocean.
    """
    fill = (None if name in NO_FILL or not np.issubdtype(np.dtype(dtype), np.floating)
            else float("nan"))
    array = group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype,
                                 compressor=compressor, overwrite=True, fill_value=fill)
    array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    return array


def decode(array) -> np.ndarray:
    """Decode a CF time array to datetimes for matching between stores."""
    attrs = array.attrs.asdict()
    return cftime.num2date(
        array[:], attrs["units"], calendar=attrs.get("calendar", "standard"),
        only_use_cftime_datetimes=False,
    )


def align(source_times, raw_times, *, label: str) -> np.ndarray:
    """Index into `raw_times` for every entry of `source_times`.

    The two stores start on different days and hold different spans, so a shared
    position index would silently read the wrong hour.
    """
    lookup = {value: index for index, value in enumerate(raw_times)}
    try:
        return np.array([lookup[value] for value in source_times], dtype=np.int64)
    except KeyError as error:
        raise SystemExit(
            f"{label} does not cover timestamp {error.args[0]}, which the source "
            "cache holds. The extra channel cannot be built for that window."
        ) from None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-cache", type=Path, required=True,
                   help="Packed cache supplying the existing boundary channels")
    p.add_argument("--output", type=Path, default=None,
                   help="Write a separate boundary-only cache instead of extending "
                        "the source in place")
    p.add_argument("--in-place", action="store_true",
                   help="Extend --source-cache itself (recommended: one store "
                        "serves runs with and without the new channel)")
    p.add_argument("--finalize", action="store_true",
                   help="Swap the staged arrays into place; run once, after --fill")
    p.add_argument("--keep-backup", action="store_true", default=True,
                   help="Move the replaced arrays to a sibling directory")
    p.add_argument("--extra", nargs="+", required=True,
                   help="Variables to add from the raw store, e.g. oceFWflx")
    p.add_argument("--raw-store", type=Path,
                   default=Path("/orcd/data/abodner/003/LLC4320/LLC4320"))
    p.add_argument("--means", type=Path,
                   default=Path("/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr"))
    p.add_argument("--stds", type=Path,
                   default=Path("/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr"))
    p.add_argument("--face", type=int, default=None,
                   help="LLC face; inferred from the source cache name if omitted")
    p.add_argument("--compressor", default="lz4")
    p.add_argument("--compression-level", type=int, default=5)
    p.add_argument("--shuffle", default="shuffle",
                   choices=("shuffle", "bitshuffle", "noshuffle"))
    p.add_argument("--workers", type=int, default=8)
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
    if not (args.init or args.fill or args.finalize):
        args.init = args.fill = args.finalize = True
    if not args.in_place and args.output is None:
        raise SystemExit("Pass --in-place, or --output for a separate store")
    target = args.source_cache if args.in_place else args.output
    #: Staged names. The dimension name differs too, so a reader opening the
    #: store mid-build sees no `boundary_channel` size clash.
    STAGE = {n: f"{n}__v2" for n in
             ("boundary", "boundary_mean", "boundary_std", "boundary_mask",
              "boundary_channel")}
    STAGE_DIM = "boundary_channel__v2"

    source = zarr.open_group(str(args.source_cache), mode="r")
    attrs = source.attrs.asdict()
    names = json.loads(attrs["boundary_channel_names_json"])
    out_names = names + list(args.extra)
    clash = [n for n in args.extra if n in names]
    if clash and not args.finalize:
        raise SystemExit(f"{clash} already in the boundary channels of {target}")

    x, y = np.asarray(source["x"][:]), np.asarray(source["y"][:])
    i0, i1, j0, j1 = int(x[0]), int(x[-1]) + 1, int(y[0]), int(y[-1]) + 1
    face = args.face
    if face is None:
        match = re.search(r"face(\d+)", args.source_cache.name)
        if match is None:
            raise SystemExit("Could not infer --face from the cache name; pass it")
        face = int(match.group(1))
    n_time = source["boundary"].shape[0]
    logger.info("target %s (%s)", target.name, "in place" if args.in_place else "new store")
    logger.info("  tile face=%d i[%d:%d) j[%d:%d), %d timesteps", face, i0, i1, j0, j1, n_time)
    logger.info("  boundary %s -> %s", names, out_names)

    raw = zarr.open_group(str(args.raw_store), mode="r")
    for name in args.extra:
        if name not in raw:
            raise SystemExit(f"{args.raw_store} has no `{name}`")

    stop = args.time_index_stop if args.time_index_stop is not None else n_time
    if args.time_split_index is not None:
        edges = np.linspace(0, n_time, args.time_splits + 1).astype(int)
        args.time_index_start, stop = int(edges[args.time_split_index]), int(
            edges[args.time_split_index + 1])
        logger.info("Split %d/%d -> time [%d:%d)", args.time_split_index + 1,
                    args.time_splits, args.time_index_start, stop)
    stop = min(stop, n_time)

    if args.dry_run:
        logger.info("[dry-run] %s, time [%d:%d)", target, args.time_index_start, stop)
        return

    group = zarr.open_group(str(target), mode="r+" if args.in_place else "a")
    dtype = source["boundary"].dtype
    n_out = len(out_names)

    if args.init:
        if STAGE["boundary"] in group and not args.overwrite:
            raise SystemExit(
                f"{target} already has staged arrays; pass --overwrite to restage")
        stats = {}
        for stat, store in (("mean", args.means), ("std", args.stds)):
            stats_group = zarr.open_group(str(store), mode="r")
            for name in args.extra:
                if name not in stats_group:
                    raise SystemExit(
                        f"{store} has no `{name}`. Add it first with\n"
                        f"  sbatch JOBS/other/job_LLC_mean_std.sh add {name}")
                stats[(stat, name)] = float(np.asarray(stats_group[name][...]).ravel()[0])
        logger.info("Staging %s (%d channels)", STAGE["boundary"], n_out)
        create(group, STAGE["boundary"], (n_time, n_out, j1 - j0, i1 - i0),
               (1, n_out, j1 - j0, i1 - i0), dtype,
               ("time", STAGE_DIM, "y", "x"),
               numcodecs.Blosc(cname=args.compressor, clevel=args.compression_level,
                               shuffle=getattr(numcodecs.Blosc, args.shuffle.upper())))
        for stat in ("mean", "std"):
            values = np.concatenate([
                np.asarray(source[f"boundary_{stat}"][:]),
                np.asarray([stats[(stat, n)] for n in args.extra], dtype=dtype),
            ]).astype(dtype)
            create(group, STAGE[f"boundary_{stat}"], values.shape, values.shape,
                   dtype, (STAGE_DIM,))[:] = values
        mask = np.asarray(source["boundary_mask"][:])
        # Every boundary field shares the surface wet mask.
        mask = np.concatenate([mask] + [mask[:1]] * len(args.extra), axis=0)
        create(group, STAGE["boundary_mask"], mask.shape, mask.shape, mask.dtype,
               (STAGE_DIM, "y", "x"))[:] = mask
        create(group, STAGE["boundary_channel"], (n_out,), (n_out,), "i4",
               (STAGE_DIM,))[:] = np.arange(n_out, dtype=np.int32)

    if args.fill:
        staged = group[STAGE["boundary"]]
        raw_index = align(decode(source["time"]), decode(raw["time"]),
                          label=str(args.raw_store))
        extras = [raw[name] for name in args.extra]
        base = len(names)
        pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        started = _time.perf_counter()
        total = stop - args.time_index_start
        try:
            def one(t):
                """Existing channels from the cache, new ones from the raw store."""
                buffer = np.empty((n_out, j1 - j0, i1 - i0), dtype=dtype)
                buffer[:base] = source["boundary"][t]
                src_t = int(raw_index[t])
                for offset, array in enumerate(extras):
                    buffer[base + offset] = array[
                        src_t, face, j0:j1, i0:i1].astype(dtype)
                return t, buffer

            times = range(args.time_index_start, stop)
            # Parallelise ACROSS timesteps: with one extra variable there is only
            # one globe chunk per step, so there is nothing to overlap within a
            # step. Peak scratch is workers x one inflated globe chunk (~1 GiB).
            results = pool.map(one, times) if pool else map(one, times)
            for done, (t, buffer) in enumerate(results, start=1):
                staged[t] = buffer
                if done % args.log_every == 0 or done == total:
                    rate = (_time.perf_counter() - started) / done
                    logger.info("time %d/%d | %.2f s/step | eta %.1f h", done, total,
                                rate, rate * (total - done) / 3600)
        finally:
            if pool is not None:
                pool.shutdown()

    if args.finalize:
        if STAGE["boundary"] not in group:
            raise SystemExit(f"{target} has no staged arrays to finalize")
        backup = target.with_name(target.name + ".boundary_v1_backup")
        backup.mkdir(parents=True, exist_ok=True)
        logger.info("Swapping staged arrays in; old ones -> %s", backup.name)
        for live, stage in STAGE.items():
            live_path, stage_path = target / live, target / stage
            if live_path.exists():
                # Out of the group entirely: left inside, the 4-wide array would
                # clash with the now 5-wide `boundary_channel` dimension.
                (backup / live).exists() and __import__("shutil").rmtree(backup / live)
                live_path.rename(backup / live)
            stage_path.rename(live_path)
            spec = json.loads((live_path / ".zattrs").read_text())
            spec["_ARRAY_DIMENSIONS"] = [
                "boundary_channel" if d == STAGE_DIM else d
                for d in spec["_ARRAY_DIMENSIONS"]
            ]
            (live_path / ".zattrs").write_text(json.dumps(spec, indent=4))
        group = zarr.open_group(str(target), mode="r+")
        group.attrs.update({
            "boundary_channel_count": n_out,
            "boundary_channel_names_json": json.dumps(out_names),
            "boundary_channels_added": json.dumps(list(args.extra)),
        })
        zarr.consolidate_metadata(str(target))
        logger.info("boundary is now %s", out_names)
    logger.info("Done.")


if __name__ == "__main__":
    main()
