#!/usr/bin/env python
"""Coarsen LLC4320 to 1/4 and 1 degree, and rechunk its 2D fields into 720 tiles.

Plan and decisions: docs/10-15-26-coarsen_LLC_plan.md. Modes:

  init     create the two coarse stores and the rechunk store, write statics
  fill     --part boundary:   2D fields -> rechunk store + coarse vars_2d
           --part prognostic: U, V, Theta, Salt -> coarse vars_3d
  stats    per-channel area-weighted mean/std over 96 finished train steps
  verify   markers, chunk files, conservation spot checks, consolidate
           (run it only when no fill is running: it deletes *.partial files)

A step is done once `<coarse-root>/_progress/<part>/<t>` exists. The marker is
written after every chunk of that step, and zarr writes chunks by rename, so a
killed fill just redoes its unfinished steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import resource
import shutil
import time as _time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numcodecs
import numpy as np
import zarr

from build_llc_face_cache import FACE_SIZE, TILE, create_array, open_source_array

from ocean_emulators.llc_coarse import (  # noqa: E402  (src is put on sys.path above)
    N_FACES,
    bottom_hfac,
    box_sum,
    center_inplace,
    fill_sources,
    lonlat_from_vector,
    ratio,
    rotate_to_east_north,
    unit_vector,
    weighted_box_sum,
)

logger = logging.getLogger(__name__)

SOURCE = Path("/orcd/data/abodner/003/LLC4320/LLC4320")
COARSE_ROOT = Path("/orcd/scratch/seedfund/001/llc_emulator/coarse")
RECHUNK_ROOT = Path("/orcd/data/abodner/002/cody/LLC_boundary_rechunk")
QUARTER, ONE = 12, 48
BOXES = (QUARTER, ONE)
COARSE_NAMES = {QUARTER: "LLC4320_coarse_0p25deg.zarr", ONE: "LLC4320_coarse_1deg.zarr"}
RECHUNK_NAME = "LLC4320_2d_720.zarr"
LEVELS = 51
VARS_3D = ("U", "V", "Theta", "Salt")
VARS_2D = ("Eta", "oceTAUX", "oceTAUY", "oceQnet", "oceFWflx")
CHANNELS_3D = [f"{v}_{k}" for v in VARS_3D for k in range(LEVELS)]
TRAIN_STEPS = 8808  # 2011-09-13 .. 2012-09-13, as in the face-1 cache
TILES = [(j, i) for j in range(0, FACE_SIZE, TILE) for i in range(0, FACE_SIZE, TILE)]
F32 = np.float32


def coarse_path(args, b: int) -> Path:
    return args.coarse_root / COARSE_NAMES[b]


def rechunk_path(args) -> Path:
    return args.rechunk_root / RECHUNK_NAME


def marker_dir(args, part: str) -> Path:
    return args.coarse_root / "_progress" / part


def done_steps(args, part: str) -> set[int]:
    path = marker_dir(args, part)
    return {int(n) for n in os.listdir(path)} if path.exists() else set()


def compressor(args) -> numcodecs.Blosc:
    return numcodecs.Blosc(cname="zstd", clevel=args.clevel, shuffle=numcodecs.Blosc.BITSHUFFLE)


def tile_slices(j0: int, i0: int, b: int) -> tuple[slice, slice]:
    """Coarse rows/cols covered by the 720 tile at (j0, i0)."""
    return slice(j0 // b, (j0 + TILE) // b), slice(i0 // b, (i0 + TILE) // b)


@contextmanager
def timed(timing: dict, name: str):
    """Add the block's wall time to `timing[name]`."""
    tic = _time.perf_counter()
    yield
    timing[name] = timing.get(name, 0.0) + _time.perf_counter() - tic


def chunk_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------- init


def find_missing(source_path: Path, n_time: int) -> dict[str, list[int]]:
    """2D steps whose whole-globe chunk is absent; fill_sources rejects back-to-back gaps."""
    missing = {}
    for var in VARS_2D:
        present = {int(n) for n in os.listdir(source_path / var) if n.isdigit()}
        steps = sorted(set(range(n_time)) - present)
        for t in steps:
            fill_sources(t, n_time, set(steps))
        missing[var] = steps
    return missing


GRID_VARS = ("XC", "YC", "rA", "dxF", "dyF", "Depth", "CS", "SN", "hFacC")


def static_sums(grids: dict, f: int, j0: int, i0: int) -> tuple[dict, np.ndarray]:
    """1/4-degree sums of every static field for one tile, plus its fine hFacC_2d."""
    def read(name):
        return grids[name].read(None, f, j0, j0 + TILE, i0, i0 + TILE)

    xc, yc, ra, dx, dy, depth, cs, sn = (
        read(n) for n in ("XC", "YC", "rA", "dxF", "dyF", "Depth", "CS", "SN"))
    hfac = np.nan_to_num(read("hFacC"))
    wet0 = hfac[0] > 0
    if np.any(np.abs(cs * cs + sn * sn - 1)[wet0] > 1e-3):
        raise ValueError(f"CS^2 + SN^2 != 1 on wet cells of face {f} tile ({j0}, {i0})")
    hf2 = bottom_hfac(hfac)
    one = F32(1)
    return {
        "xyz": box_sum(unit_vector(xc, yc) * ra, QUARTER),
        "rA": box_sum(ra, QUARTER),
        "dxF": box_sum(dx, QUARTER),
        "dyF": box_sum(dy, QUARTER),
        "Depth": weighted_box_sum(np.nan_to_num(depth), ra, wet0, QUARTER),
        "hFacC_2d": weighted_box_sum(hf2, ra, wet0, QUARTER),
        "wet_area": weighted_box_sum(one, ra, wet0, QUARTER),
        "hfac_area": box_sum(hfac * ra, QUARTER),
    }, hf2


def statics_from_sums(sums: dict, b: int) -> dict[str, np.ndarray]:
    """Static coarse fields at box size `b` from the 1/4-degree sums."""
    def up(s):
        return box_sum(s, b // QUARTER) if b > QUARTER else s

    xc, yc = lonlat_from_vector(up(sums["xyz"]))
    ra, wet_area = up(sums["rA"]), up(sums["wet_area"])
    return {
        "XC": xc, "YC": yc, "rA": ra.astype(F32),
        "dxF": (up(sums["dxF"]) / b).astype(F32),
        "dyF": (up(sums["dyF"]) / b).astype(F32),
        "Depth": ratio(up(sums["Depth"]), wet_area),
        "hFacC_2d": ratio(up(sums["hFacC_2d"]), wet_area),
        "wet_frac": ratio(up(sums["hfac_area"]), ra),
    }


STATIC_DIMS = {"wet_frac": ("k", "face", "y", "x")}
STATIC_ATTRS = {
    "XC": "box centre longitude (area-weighted mean of the unit vector)",
    "YC": "box centre latitude (area-weighted mean of the unit vector)",
    "rA": "box area: sum of 1/48-degree rA",
    "dxF": "box width along the face's own i axis (mean row sum)",
    "dyF": "box height along the face's own j axis (mean column sum)",
    "Depth": "rA-weighted mean over surface-wet pixels",
    "hFacC_2d": "rA-weighted mean of the bottom cell's open-water fraction, surface-wet pixels",
    "wet_frac": "coarse hFacC: rA-weighted box mean of 1/48-degree hFacC; mask = wet_frac > 0",
}


def init(args) -> None:
    paths = [coarse_path(args, b) for b in BOXES] + [rechunk_path(args)]
    existing = [p for p in paths if p.exists()]
    if existing and not args.force:
        raise FileExistsError(f"{existing} exist; init would erase them. Pass --force.")
    for p in existing + [args.coarse_root / "_progress"]:
        shutil.rmtree(p, ignore_errors=True)

    source = zarr.open_group(str(args.source), mode="r")
    n_time = source["Theta"].shape[0]
    logger.info("Scanning %s for missing 2D steps", args.source)
    missing = find_missing(args.source, n_time)
    logger.info("Missing steps: %s", {v: len(s) for v, s in missing.items()})
    time_values = source["time"][:]
    time_attrs = {k: v for k, v in source["time"].attrs.items() if k != "_ARRAY_DIMENSIONS"}

    common = {
        "built_by": "coarsen_llc.py",
        "plan": "docs/10-15-26-coarsen_LLC_plan.md",
        "source_path": str(args.source),
        "filled_time_steps_json": json.dumps(missing),
        "fill_rule": "mean of steps t-1 and t+1; t=0 copies t=1",
        "train_steps": [0, TRAIN_STEPS],
        "val_steps": [TRAIN_STEPS, n_time],
    }

    # Rechunk store: raw values, raw compressor, one array per variable.
    rechunk = zarr.open_group(str(rechunk_path(args)), mode="w")
    for var in VARS_2D:
        raw = source[var]
        create_array(rechunk, var, raw.shape, (1, 1, TILE, TILE), "f4",
                     raw.attrs["_ARRAY_DIMENSIONS"], raw.compressor, dimension_separator="/")
    hf2_out = create_array(rechunk, "hFacC_2d", (N_FACES, FACE_SIZE, FACE_SIZE), (1, TILE, TILE),
                           "f4", ("face", "j", "i"), source["rA"].compressor,
                           dimension_separator="/")
    hf2_out.attrs["description"] = "open-water fraction of each column's bottom cell (deep=1, land=0)"
    create_array(rechunk, "time", (n_time,), (1024,), time_values.dtype, ("time",))[:] = time_values
    rechunk["time"].attrs.update(time_attrs)
    rechunk.attrs.update(common | {"description": "LLC4320 2D fields rechunked to 720x720 tiles; raw values"})

    # Statics, tile by tile.
    n_q = FACE_SIZE // QUARTER
    sums: dict[str, np.ndarray] = {}
    grids = {name: open_source_array(source, name) for name in GRID_VARS}

    def tile_job(job):
        f, (j0, i0) = job
        tile_sums, hf2 = static_sums(grids, f, j0, i0)
        hf2_out[f, j0:j0 + TILE, i0:i0 + TILE] = hf2
        return f, j0, i0, tile_sums

    jobs = [(f, origin) for f in range(N_FACES) for origin in TILES]
    started = _time.perf_counter()
    with ThreadPoolExecutor(args.read_threads) as pool:
        for count, (f, j0, i0, tile_sums) in enumerate(pool.map(tile_job, jobs), 1):
            rows, cols = tile_slices(j0, i0, QUARTER)
            for name, value in tile_sums.items():
                if name not in sums:
                    sums[name] = np.zeros(value.shape[:-2] + (N_FACES, n_q, n_q))
                sums[name][..., f, rows, cols] = value
            if count % 52 == 0:
                logger.info("statics %d/%d tiles (%.0f s)", count, len(jobs),
                            _time.perf_counter() - started)

    for b in BOXES:
        n = FACE_SIZE // b
        group = zarr.open_group(str(coarse_path(args, b)), mode="w")
        comp = compressor(args)
        create_array(group, "vars_3d", (n_time, len(CHANNELS_3D), N_FACES, n, n),
                     (1, len(CHANNELS_3D), N_FACES, n, n), "f4",
                     ("time", "channel_3d", "face", "y", "x"), comp)
        create_array(group, "vars_2d", (n_time, len(VARS_2D), N_FACES, n, n),
                     (1, len(VARS_2D), N_FACES, n, n), "f4",
                     ("time", "channel_2d", "face", "y", "x"), comp)
        for name, value in statics_from_sums(sums, b).items():
            dims = STATIC_DIMS.get(name, ("face", "y", "x"))
            chunks = (LEVELS, 1, n, n) if name == "wet_frac" else value.shape
            out = create_array(group, name, value.shape, chunks, "f4", dims, comp)
            out[:] = value
            out.attrs["description"] = STATIC_ATTRS[name]
        create_array(group, "time", (n_time,), (1024,), time_values.dtype, ("time",))[:] = time_values
        group["time"].attrs.update(time_attrs)
        group.attrs.update(common | {
            "box_size": b,
            "nominal_resolution_deg": b / 48,
            "channel_names_3d_json": json.dumps(CHANNELS_3D),
            "channel_names_2d_json": json.dumps(list(VARS_2D)),
            "velocity_frame": "east/north: U, V, oceTAUX, oceTAUY centred and rotated with CS/SN",
            "face_edge_velocity": "edge value copied (xgcm boundary='extend')",
            "weights": "rA over wet pixels; partial bottom cells count as full cells",
            "land_fill": "NaN",
            "layout": "native faces; stitch with ocean_emulators.llc_coarse.faces_to_latlon",
        })
        zarr.consolidate_metadata(str(coarse_path(args, b)))
    zarr.consolidate_metadata(str(rechunk_path(args)))
    for part in ("boundary", "prognostic"):
        marker_dir(args, part).mkdir(parents=True, exist_ok=True)
    logger.info("init done in %.0f s", _time.perf_counter() - started)


# ---------------------------------------------------------------- fill

#: Per-process state, set once by `init_worker`.
_W: dict = {}


def init_worker(cfg: dict) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    numcodecs.blosc.use_threads = False
    part = cfg["part"]
    source = zarr.open_group(cfg["source"], mode="r")
    pool = ThreadPoolExecutor(cfg["read_threads"])
    _W.update(cfg=cfg, source=source, pool=pool)
    _W["grid"] = {
        name: np.stack(list(pool.map(lambda f, a=source[name]: a[f], range(N_FACES))))
        for name in ("rA", "CS", "SN")
    }
    name = "vars_3d" if part == "prognostic" else "vars_2d"
    _W["out"] = {b: zarr.open_array(str(Path(cfg["coarse"][b]) / name), mode="r+") for b in BOXES}
    if part == "prognostic":
        _W["src"] = {v: open_source_array(source, v) for v in VARS_3D}
        wet_frac = zarr.open_array(str(Path(cfg["coarse"][QUARTER]) / "wet_frac"), mode="r")
        _W["wet_q"] = wet_frac[:] > 0
    else:
        rechunk = Path(cfg["rechunk"])
        _W["wet0"] = zarr.open_array(str(rechunk / "hFacC_2d"), mode="r")[:] > 0
        _W["src"] = {v: source[v] for v in VARS_2D}
        _W["rechunk"] = {
            v: zarr.open_array(str(rechunk / v), mode="r+", write_empty_chunks=False)
            for v in VARS_2D
        }
        _W["missing"] = {v: set(s) for v, s in json.loads(cfg["missing"]).items()}


def write_step(t: int, out: dict[int, np.ndarray], timing: dict) -> None:
    """1 degree first, then 1/4 degree, then the marker."""
    with timed(timing, "write"):
        for b in (ONE, QUARTER):
            _W["out"][b][t] = out[b]
        (Path(_W["cfg"]["markers"]) / f"{t:05d}").touch()


def run_step(t: int) -> tuple[int, float, int, float, dict]:
    started = _time.perf_counter()
    step = prognostic_step if _W["cfg"]["part"] == "prognostic" else boundary_step
    timing: dict[str, float] = {}
    nbytes = step(t, timing)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return t, _time.perf_counter() - started, nbytes, rss_gb, timing


def prognostic_step(t: int, timing: dict) -> int:
    pool, src, wet_q = _W["pool"], _W["src"], _W["wet_q"]
    ra, cs, sn = (_W["grid"][k] for k in ("rA", "CS", "SN"))
    source_path = Path(_W["cfg"]["source"])
    out = {b: np.full((len(CHANNELS_3D), N_FACES, FACE_SIZE // b, FACE_SIZE // b), np.nan, F32)
           for b in BOXES}
    nbytes = 0

    def key(var, f, j0, i0):
        return source_path / var / f"{t}/0/{f}/{j0 // TILE}/{i0 // TILE}"

    for f in range(N_FACES):
        def read_face(var):
            face = np.empty((LEVELS, FACE_SIZE, FACE_SIZE), F32)

            def one(origin):
                j0, i0 = origin
                # A missing chunk reads as NaN, which centring would turn into 0.
                # (An all-NaN U/V tile alone is legal: isolated lakes have no open faces.)
                nbytes = chunk_bytes(key(var, f, j0, i0))
                rows, cols = tile_slices(j0, i0, QUARTER)
                if nbytes == 0 and wet_q[:, f, rows, cols].any():
                    raise RuntimeError(f"{var} t={t} face={f} tile=({j0},{i0}): source chunk "
                                       "missing on a wet tile")
                face[:, j0:j0 + TILE, i0:i0 + TILE] = src[var].read(t, f, j0, j0 + TILE, i0, i0 + TILE)
                return nbytes

            return face, sum(pool.map(one, TILES))

        with timed(timing, "read_uv"):
            (u, bu), (v, bv) = read_face("U"), read_face("V")
        nbytes += bu + bv
        # One level per thread: a whole-face in-place centring allocates a hidden copy.
        with timed(timing, "centre"):
            list(pool.map(lambda k: center_inplace(u[k], axis=-1), range(LEVELS)))
            list(pool.map(lambda k: center_inplace(v[k], axis=-2), range(LEVELS)))

        def tile(origin):
            j0, i0 = origin
            tj, ti = slice(j0, j0 + TILE), slice(i0, i0 + TILE)
            theta = src["Theta"].read(t, f, j0, j0 + TILE, i0, i0 + TILE)
            salt = src["Salt"].read(t, f, j0, j0 + TILE, i0, i0 + TILE)
            wet = np.isfinite(theta)
            area = ra[f, tj, ti]
            den = weighted_box_sum(F32(1), area, wet, QUARTER)
            rows, cols = tile_slices(j0, i0, QUARTER)
            if not np.array_equal(den > 0, wet_q[:, f, rows, cols]):
                raise RuntimeError(f"Theta wet pattern differs from hFacC: t={t} face={f} "
                                   f"tile=({j0},{i0})")
            if not np.array_equal(np.isfinite(salt), wet):
                raise RuntimeError(f"Salt and Theta NaN patterns differ: t={t} face={f} "
                                   f"tile=({j0},{i0})")
            east, north = rotate_to_east_north(u[:, tj, ti], v[:, tj, ti], cs[f, tj, ti], sn[f, tj, ti])
            den1 = box_sum(den, ONE // QUARTER)
            r1, c1 = tile_slices(j0, i0, ONE)
            for n, field in enumerate((east, north, theta, salt)):
                num = weighted_box_sum(field, area, wet, QUARTER)
                c = slice(n * LEVELS, (n + 1) * LEVELS)
                out[QUARTER][c, f, rows, cols] = ratio(num, den)
                out[ONE][c, f, r1, c1] = ratio(box_sum(num, ONE // QUARTER), den1)
            return chunk_bytes(key("Theta", f, j0, i0)) + chunk_bytes(key("Salt", f, j0, i0))

        with timed(timing, "tiles"):
            nbytes += sum(pool.map(tile, TILES))
        del u, v
    write_step(t, out, timing)
    return nbytes


def read_plane(var: str, t: int) -> tuple[np.ndarray, int]:
    """One whole-globe 2D chunk, filled from its neighbours when it is missing."""
    array, source_path = _W["src"][var], Path(_W["cfg"]["source"])
    missing = _W["missing"][var]
    steps = fill_sources(t, array.shape[0], missing) if t in missing else [t]
    plane = np.asarray(array[steps[0]], dtype=F32)
    for s in steps[1:]:
        plane += array[s]
    if len(steps) > 1:
        plane *= F32(1 / len(steps))
    if np.isnan(plane).all():
        raise RuntimeError(f"{var} t={t} is all NaN but not listed as missing")
    return plane, sum(chunk_bytes(source_path / var / f"{s}/0/0/0") for s in steps)


def boundary_step(t: int, timing: dict) -> int:
    pool, wet0 = _W["pool"], _W["wet0"]
    ra, cs, sn = (_W["grid"][k] for k in ("rA", "CS", "SN"))
    with timed(timing, "read"):
        read = dict(zip(VARS_2D, pool.map(lambda var: read_plane(var, t), VARS_2D)))
    planes = {var: plane for var, (plane, _) in read.items()}
    for var in ("Eta", "oceQnet", "oceFWflx"):
        if np.isnan(planes[var][wet0]).any():
            raise RuntimeError(f"{var} t={t} has NaN on surface-wet cells")

    def put(job):
        var, f, (j0, i0) = job
        _W["rechunk"][var][t, f, j0:j0 + TILE, i0:i0 + TILE] = planes[var][f, j0:j0 + TILE, i0:i0 + TILE]

    with timed(timing, "rechunk"):
        list(pool.map(put, [(var, f, o) for var in VARS_2D for f in range(N_FACES) for o in TILES]))

    # Only after the raw tiles are written: the wind stress is centred in place.
    taux, tauy = planes["oceTAUX"], planes["oceTAUY"]
    list(pool.map(lambda f: center_inplace(taux[f], axis=-1), range(N_FACES)))
    list(pool.map(lambda f: center_inplace(tauy[f], axis=-2), range(N_FACES)))
    out = {b: np.full((len(VARS_2D), N_FACES, FACE_SIZE // b, FACE_SIZE // b), np.nan, F32)
           for b in BOXES}

    def face(f):
        east, north = rotate_to_east_north(taux[f], tauy[f], cs[f], sn[f])
        fields = (planes["Eta"][f], east, north, planes["oceQnet"][f], planes["oceFWflx"][f])
        den = weighted_box_sum(F32(1), ra[f], wet0[f], QUARTER)
        den1 = box_sum(den, ONE // QUARTER)
        for n, field in enumerate(fields):
            num = weighted_box_sum(field, ra[f], wet0[f], QUARTER)
            out[QUARTER][n, f] = ratio(num, den)
            out[ONE][n, f] = ratio(box_sum(num, ONE // QUARTER), den1)

    with timed(timing, "coarsen"):
        list(pool.map(face, range(N_FACES)))
    write_step(t, out, timing)
    return sum(nbytes for _, nbytes in read.values())


def pending_steps(args, n_time: int) -> list[int]:
    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        edges = np.linspace(0, n_time, args.time_splits + 1).astype(int)
        index = args.time_split_index or 0
        steps = list(range(edges[index], edges[index + 1]))
        if args.stride:
            steps = [t for t in steps if t % args.stride == args.offset]
    done = done_steps(args, args.part)
    pending = [t for t in steps if t not in done]
    return pending[: args.limit] if args.limit else pending


def fill(args) -> None:
    if args.part is None:
        raise ValueError("fill needs --part boundary or --part prognostic")
    group = zarr.open_group(str(coarse_path(args, QUARTER)), mode="r")
    n_time = group["vars_3d"].shape[0]
    pending = pending_steps(args, n_time)
    logger.info("%s: %d pending step(s), %d workers x %d threads", args.part, len(pending),
                args.workers, args.read_threads)
    if not pending:
        return
    cfg = {
        "part": args.part,
        "source": str(args.source),
        "coarse": {b: str(coarse_path(args, b)) for b in BOXES},
        "rechunk": str(rechunk_path(args)),
        "markers": str(marker_dir(args, args.part)),
        "missing": group.attrs["filled_time_steps_json"],
        "read_threads": args.read_threads,
    }
    marker_dir(args, args.part).mkdir(parents=True, exist_ok=True)
    started = _time.perf_counter()
    total_bytes, step_seconds, max_rss, split = 0, [], 0.0, {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(cfg,)) as pool:
        futures = {pool.submit(run_step, t): t for t in pending}
        try:
            for count, future in enumerate(as_completed(futures), 1):
                t, seconds, nbytes, rss, timing = future.result()
                total_bytes += nbytes
                step_seconds.append(seconds)
                max_rss = max(max_rss, rss)
                for name, value in timing.items():
                    split[name] = split.get(name, 0.0) + value
                if count % args.log_every == 0 or count == len(pending):
                    elapsed = _time.perf_counter() - started
                    logger.info(
                        "%d/%d done (last t=%d) | %.1f s/step wall, %.0f s/step per worker (%s) | "
                        "%.2f GB/s read | worker peak %.1f GB | eta %.1f h",
                        count, len(pending), t, elapsed / count, np.mean(step_seconds),
                        ", ".join(f"{k} {v / count:.0f}" for k, v in split.items()),
                        total_bytes / elapsed / 1e9, max_rss,
                        elapsed / count * (len(pending) - count) / 3600)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


# ---------------------------------------------------------------- stats


def stats(args) -> None:
    both = done_steps(args, "boundary") & done_steps(args, "prognostic")
    steps = sorted(t for t in both if t < TRAIN_STEPS)
    if len(steps) < 96:
        raise RuntimeError(f"only {len(steps)} train steps are done for both parts; need 96")
    steps = [steps[i] for i in np.unique(np.linspace(0, len(steps) - 1, 96).round().astype(int))]
    for b in BOXES:
        group = zarr.open_group(str(coarse_path(args, b)), mode="r+")
        area = group["rA"][:].astype(np.float64)
        for name, dim in (("vars_3d", "channel_3d"), ("vars_2d", "channel_2d")):
            n_ch = group[name].shape[1]
            s0, s1, s2 = np.zeros(n_ch), np.zeros(n_ch), np.zeros(n_ch)
            for t in steps:
                x = group[name][t].astype(np.float64)
                ok = np.isfinite(x)
                w = np.where(ok, area, 0.0)
                xw = np.where(ok, x, 0.0) * w
                s0 += w.sum(axis=(1, 2, 3))
                s1 += xw.sum(axis=(1, 2, 3))
                s2 += (xw * np.where(ok, x, 0.0)).sum(axis=(1, 2, 3))
            mean = s1 / s0
            std = np.sqrt(np.maximum(s2 / s0 - mean**2, 0.0))
            create_array(group, f"{name}_mean", (n_ch,), (n_ch,), "f4", (dim,))[:] = mean
            create_array(group, f"{name}_std", (n_ch,), (n_ch,), "f4", (dim,))[:] = std
        group.attrs["stats_steps_json"] = json.dumps(steps)
        zarr.consolidate_metadata(str(coarse_path(args, b)))
        logger.info("stats written for %s from %d steps", COARSE_NAMES[b], len(steps))


# ---------------------------------------------------------------- verify


def conservation_error(source: zarr.Group, coarse: zarr.Group, wet0: np.ndarray | None,
                       var: str, t: int, face: int, b: int) -> float:
    """Worst relative mismatch between sum(x*rA) on raw wet pixels and on coarse cells.

    Covers one face: every level of a 3D tracer, or a 2D C-point field with the
    surface-wet mask `wet0` (from the rechunk store's hFacC_2d > 0).
    """
    ra = np.asarray(source["rA"][face], dtype=F32)
    if var in VARS_3D:
        x = np.asarray(source[var][t, :, face], dtype=F32)
        wet = np.isfinite(x)
        c = CHANNELS_3D.index(f"{var}_0")
        coarse_x = coarse["vars_3d"][t, c:c + LEVELS, face]
    else:
        x = np.asarray(source[var][t, face], dtype=F32)
        wet = wet0
        coarse_x = coarse["vars_2d"][t, VARS_2D.index(var), face]
    fine = (np.where(wet, x, F32(0)) * ra).sum(axis=(-2, -1), dtype=np.float64)
    den = weighted_box_sum(F32(1), ra, wet, b)
    coarse_total = np.nansum(coarse_x.astype(np.float64) * den, axis=(-2, -1))
    return float(np.max(np.abs(coarse_total - fine) / np.maximum(np.abs(fine), 1e-30)))


def verify(args) -> None:
    source = zarr.open_group(str(args.source), mode="r")
    coarse = {b: zarr.open_group(str(coarse_path(args, b)), mode="r") for b in BOXES}
    n_time = coarse[QUARTER]["vars_3d"].shape[0]
    problems = []
    complete = True
    for part, name in (("boundary", "vars_2d"), ("prognostic", "vars_3d")):
        done = done_steps(args, part)
        todo = sorted(set(range(n_time)) - done)
        logger.info("%s: %d/%d steps marked done", part, len(done), n_time)
        complete &= not todo
        for b in BOXES:
            lost = [t for t in done if not (coarse_path(args, b) / name / f"{t}.0.0.0.0").exists()]
            if lost:
                problems.append(f"{COARSE_NAMES[b]}/{name}: {len(lost)} marked steps lack a chunk, "
                                f"e.g. {lost[:5]}")
    for b in BOXES:
        for entry in os.scandir(coarse_path(args, b) / "vars_3d"):
            if entry.name.endswith(".partial"):
                os.remove(entry.path)
        for entry in os.scandir(coarse_path(args, b) / "vars_2d"):
            if entry.name.endswith(".partial"):
                os.remove(entry.path)

    rng = np.random.default_rng(0)
    wet0 = None
    for part, var in (("boundary", "Eta"), ("prognostic", "Theta")):
        done = sorted(done_steps(args, part))
        for t in rng.choice(done, size=min(2, len(done)), replace=False) if done else []:
            face = int(rng.integers(N_FACES))
            if var == "Eta":
                wet0 = np.asarray(zarr.open_array(str(rechunk_path(args) / "hFacC_2d"), mode="r")[face]) > 0
            for b in BOXES:
                err = conservation_error(source, coarse[b], wet0, var, int(t), face, b)
                logger.info("conservation %s t=%d face=%d box=%d: rel err %.2e", var, t, face, b, err)
                if err > 1e-5:
                    problems.append(f"{var} t={t} face={face} box={b}: conservation error {err:.2e}")

    for problem in problems:
        logger.error(problem)
    if problems:
        raise SystemExit(1)
    if complete:
        for path in [coarse_path(args, b) for b in BOXES] + [rechunk_path(args)]:
            zarr.consolidate_metadata(str(path))
        logger.info("verify passed; metadata consolidated")
    else:
        logger.info("verify found no errors, but steps are still pending")


# ---------------------------------------------------------------- main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("init", "fill", "stats", "verify"))
    p.add_argument("--part", choices=("boundary", "prognostic"))
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--coarse-root", type=Path, default=COARSE_ROOT)
    p.add_argument("--rechunk-root", type=Path, default=RECHUNK_ROOT)
    p.add_argument("--workers", type=int, default=8, help="Processes; one time step each")
    p.add_argument("--read-threads", type=int, default=8, help="Threads per process")
    p.add_argument("--time-splits", type=int, default=1)
    p.add_argument("--time-split-index", type=int, default=None)
    p.add_argument("--stride", type=int, default=None, help="Only steps with t %% stride == offset")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--steps", default=None, help="Comma-separated steps; overrides the split")
    p.add_argument("--limit", type=int, default=None, help="At most this many pending steps")
    p.add_argument("--clevel", type=int, default=5, help="zstd level for the coarse stores")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--force", action="store_true", help="init: erase existing stores and markers")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    numcodecs.blosc.use_threads = False
    {"init": init, "fill": fill, "stats": stats, "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
