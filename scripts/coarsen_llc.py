#!/usr/bin/env python
"""Regrid LLC4320 onto Samudra's 1/4- and 1-degree lat/lon grid, box-average it to
1/3 degree on the native LLC faces, and rechunk its 2D fields into 720 tiles, in
one pass per hour.

Products (--products): `samudra` = rechunk + 1/4 + 1 degree, `native` = 1/3 degree
on the 13 faces. Each has its own markers, so an hour already done for one only
computes the other.

Plan and decisions: docs/10-15-26-coarsen_LLC_plan.md. Modes:

  init     create the three stores; build the regrid weights and static fields
  fill     for each pending hour: rechunk the 2D fields, regrid every field
  stats    per-variable, per-level mean/std over 96 finished train hours
  prepare  add the `repaired` arrays to stores made before they existed
  report   merge the per-hour repair records into repairs.log and store attrs
  verify   markers, chunk files, conservation spot checks, consolidate
           (run it only when no fill is running: it deletes *.partial files)

An hour is done for a product once `<coarse-root>/_progress[_native]/<t>` exists.
The marker is written after every chunk of that hour, and zarr writes chunks by
rename, so a killed fill just redoes its unfinished hours.

Bad raw pixels are replaced by the same pixel at the nearest earlier hour where it
is clean: a pixel is bad if it is outside LIMITS or infinite, if Theta/Salt's
NaN-ness disagrees with mask_c there, or if a 2D field is NaN on a wet cell. A
missing raw chunk or 2D step is all bad, so it copies the whole field. Hour 0
uses the next hour. Each repair
is recorded in `<coarse-root>/_repairs/<t>.json`, merged into the single log
`<coarse-root>/repairs.log` by `--mode report` (and at the end of every fill),
and noted per hour and variable in every store's `repaired` array.
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
    POLAR_LAT,
    POLAR_SUBCELLS,
    SUBCELLS,
    aligned_weights,
    apply_weights,
    bottom_hfac,
    box_sum,
    cell_area,
    center_inplace,
    corner_grid,
    lonlat_from_vector,
    nearest,
    repair_candidates,
    ratio,
    rotate_to_east_north,
    split_edges,
    subcell_weights,
    tile_matrix,
    unit_vector,
)

logger = logging.getLogger(__name__)

SOURCE = Path("/orcd/data/abodner/003/LLC4320/LLC4320")
TARGET_GRID = Path("/orcd/data/abodner/002/cody/OM4.zarr")
COARSE_ROOT = Path("/orcd/scratch/seedfund/001/llc_emulator/coarse")
RECHUNK_ROOT = Path("/orcd/data/abodner/002/cody/LLC_boundary_rechunk")
QUARTER, ONE = "0p25deg", "1deg"
RESOLUTIONS = (QUARTER, ONE)
SPLIT = 4  # 1/4-degree cells per 1-degree cell, along each axis
STORE_NAMES = {res: f"LLC4320_samudra_{res}.zarr" for res in RESOLUTIONS}
RECHUNK_NAME = "LLC4320_2d_720.zarr"
LEVELS = 51
VARS_3D = ("U", "V", "Theta", "Salt")
VARS_2D = ("Eta", "oceTAUX", "oceTAUY", "oceQnet", "oceFWflx")
TRAIN_STEPS = 8808  # 2011-09-13 .. 2012-09-13, as in the face-1 cache
#: (face, j0, i0) of every 720 tile; a tile's index is its position here.
TILES = [(f, j, i) for f in range(N_FACES)
         for j in range(0, FACE_SIZE, TILE) for i in range(0, FACE_SIZE, TILE)]
TILE_PIXELS = TILE * TILE
FILL_LAT = 88.0  # fine cells poleward of this are candidates for the polar fill
F32 = np.float32
SAMUDRA, NATIVE = "samudra", "native"
PRODUCTS = (SAMUDRA, NATIVE)
NATIVE_BOX = 16  # 1/48-degree cells per native 1/3-degree cell, along each axis
NATIVE_NAME = "LLC4320_native_0p33deg.zarr"
REPAIR_RULE = ("each bad raw pixel (outside LIMITS, infinite, NaN-ness disagreeing with mask_c, or a 2D NaN on "
               "a wet cell) is replaced by the same pixel at the nearest earlier hour where it is clean; a missing "
               "chunk or 2D step copies the whole field. Hour 0 uses the next hour. See repairs.log and `repaired`.")
#: Raw values outside these are corrupted data, replaced by the previous clean hour.
LIMITS = {"Theta": (-5, 50), "Salt": (0, 100), "U": (-20, 20), "V": (-20, 20), "Eta": (-50, 50),
          "oceTAUX": (-50, 50), "oceTAUY": (-50, 50), "oceQnet": (-1e6, 1e6), "oceFWflx": (-1, 1)}


def compressor() -> numcodecs.Blosc:
    """zstd-3 with byte shuffle, as chosen for the 1-face cache."""
    return numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)


def coarse_path(args, res: str) -> Path:
    return args.coarse_root / STORE_NAMES[res]


def rechunk_path(args) -> Path:
    return args.rechunk_root / RECHUNK_NAME


def native_path(args) -> Path:
    return args.coarse_root / NATIVE_NAME


def marker_dir(args, product: str = SAMUDRA) -> Path:
    return args.coarse_root / ("_progress" if product == SAMUDRA else f"_progress_{product}")


def done_steps(args, product: str = SAMUDRA) -> set[int]:
    path = marker_dir(args, product)
    return {int(n) for n in os.listdir(path)} if path.exists() else set()


def bad_pixels(var: str, x: np.ndarray, wet: np.ndarray | None = None,
               wet_finite: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Masks of bad pixels by reason. `wet`: where x must be finite (and NaN elsewhere);
    `wet_finite`: cells that must be finite (2D fields, whose land may be 0 or NaN)."""
    lo, hi = LIMITS[var]
    masks = {"out_of_range": np.isinf(x) | (x < lo) | (x > hi)}
    if wet is not None:
        masks["nan_pattern"] = np.isfinite(x) != wet
    if wet_finite is not None:
        masks["nan_on_wet"] = wet_finite & np.isnan(x)
    return masks


def repair_pixels(x: np.ndarray, bad: np.ndarray, t: int, n_time: int, fetch, check) -> list[int]:
    """Replace `x[bad]` in place with the same pixels at the nearest hour where each is clean.

    `fetch(s)` gives the raw field at hour s (None if that chunk is missing too);
    `check(y)` gives y's bad-pixel mask. Returns the hours copied from.
    """
    remaining, sources = bad.copy(), []
    for s in repair_candidates(t, n_time):
        y = fetch(s)
        if y is None:
            continue
        ok = remaining & ~check(y)
        if ok.any():
            x[ok] = y[ok]
            remaining &= ~ok
            sources.append(s)
        if not remaining.any():
            return sources
    raise RuntimeError(f"t={t}: {int(remaining.sum())} bad pixels have no clean copy nearby")


def record(repairs: list, t: int, var: str, masks: dict, sources: list, **where) -> None:
    """One repair record: why, how many pixels, and which hours they came from."""
    kinds = [k for k, m in masks.items() if m.any()]
    repairs.append({"t": t, "var": var, "kind": "+".join(kinds),
                    "pixels": int(np.logical_or.reduce(list(masks.values())).sum()),
                    "copied_from": sources} | where)


def create_repaired(group: zarr.Group, n_time: int, variables) -> None:
    """`repaired[t, v]`: -1 not written yet, 0 original data, 1 some of it copied from another hour."""
    a = create_array(group, "repaired", (n_time, len(variables)), (1, len(variables)), "i1",
                     ("time", "variable"), compressor(), fill_value=-1)
    a.attrs.update({"variables": list(variables), "meaning": "-1 not written, 0 original, 1 repaired",
                    "rule": REPAIR_RULE})


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


def target_grids(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Samudra's 1-degree grid, and the 1/4-degree grid that splits each of its cells 4x4."""
    g = zarr.open_group(str(path), mode="r")
    lat_b, lon_b = g["lat_b"][:, 0], g["lon_b"][0, :]
    q_lat_b, q_lon_b = split_edges(lat_b, SPLIT), split_edges(lon_b, SPLIT)
    if not np.allclose(q_lon_b, np.arange(q_lon_b.size) * 360 / (q_lon_b.size - 1)):
        raise ValueError("target longitudes must be equal columns starting at 0E")
    return {
        ONE: {"lat": g["lat"][:, 0], "lon": g["lon"][0, :], "lat_b": lat_b, "lon_b": lon_b},
        QUARTER: {"lat": (q_lat_b[:-1] + q_lat_b[1:]) / 2, "lon": (q_lon_b[:-1] + q_lon_b[1:]) / 2,
                  "lat_b": q_lat_b, "lon_b": q_lon_b},
    }


def to_one_degree(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """1/4-degree sums `[..., cells]` -> 1-degree sums `[..., 180, 360]`."""
    return box_sum(x.reshape(*x.shape[:-1], *shape), SPLIT)


# ---------------------------------------------------------------- init


def find_missing(source_path: Path, n_time: int) -> dict[str, list[int]]:
    """2D steps whose whole-globe chunk is absent."""
    missing = {}
    for var in VARS_2D:
        present = {int(n) for n in os.listdir(source_path / var) if n.isdigit()}
        steps = sorted(set(range(n_time)) - present)
        missing[var] = steps
    return missing


def build_weights(source: zarr.Group, grid: dict, threads: int):
    """Per-tile `(pixel, cell, area)` and method, plus polar candidates and total LLC area."""
    lat_edges, n_lon = grid["lat_b"], grid["lon"].size
    weights, methods, polar, total_area = [None] * len(TILES), np.zeros(len(TILES), np.int8), [], 0.0
    started = _time.perf_counter()
    for f in range(N_FACES):
        lon_c, lat_c = corner_grid(source["XG"][f], source["YG"][f])
        xc, yc, ra = (np.asarray(source[n][f]) for n in ("XC", "YC", "rA"))
        total_area += ra.sum(dtype=np.float64)

        def job(n, f=f, lon_c=lon_c, lat_c=lat_c, xc=xc, yc=yc, ra=ra):
            _, j0, i0 = TILES[n]
            tj, ti, cj, ci = (slice(j0, j0 + TILE), slice(i0, i0 + TILE),
                              slice(j0, j0 + TILE + 1), slice(i0, i0 + TILE + 1))
            w = aligned_weights(lon_c[cj, ci], lat_c[cj, ci], xc[tj, ti], ra[tj, ti], lat_edges, n_lon)
            method = 0
            if w is None:
                n_sub = POLAR_SUBCELLS if np.abs(yc[tj, ti]).max() > POLAR_LAT else SUBCELLS
                w = subcell_weights(lon_c[cj, ci], lat_c[cj, ci], ra[tj, ti], lat_edges, n_lon, n_sub)
                method = 1 if n_sub == SUBCELLS else 2
            pixels = np.flatnonzero(np.abs(yc[tj, ti]) > FILL_LAT)
            xyz = unit_vector(xc[tj, ti].ravel()[pixels], yc[tj, ti].ravel()[pixels])
            return n, w, method, (np.full(pixels.size, n), pixels, xyz)

        with ThreadPoolExecutor(threads) as pool:
            for n, w, method, cand in pool.map(job, range(f * 36, (f + 1) * 36)):
                weights[n], methods[n] = w, method
                if cand[1].size:
                    polar.append(cand)
        logger.info("weights: face %d done (%d sub-cell tiles so far, %.0f s)", f,
                    int((methods > 0).sum()), _time.perf_counter() - started)
    return weights, methods, polar, total_area


def polar_fill(weights, polar, grid, radius) -> dict[str, np.ndarray]:
    """Give each target cell with no area one entry from the fine cell nearest its centre."""
    n_lon = grid["lon"].size
    n_cells = (grid["lat"].size) * n_lon
    cover = np.zeros(n_cells)
    for pixel, cell, area in weights:
        cover += np.bincount(cell, weights=area, minlength=n_cells)
    empty = np.flatnonzero(cover == 0)
    fill = {"tile": [], "pixel": [], "cell": [], "area": []}
    if empty.size:
        tiles, pixels, xyz = (np.concatenate(p, axis=-1) for p in zip(*polar))
        row, col = np.divmod(empty, n_lon)
        centres = unit_vector(grid["lon"][col], grid["lat"][row])
        index, chord = nearest(centres, xyz)
        if np.max(chord) * radius > 5e3:
            raise RuntimeError(f"polar fill: nearest fine cell is {np.max(chord) * radius:.0f} m away")
        area = cell_area(grid["lat_b"], grid["lon_b"], radius).ravel()[empty]
        fill = {"tile": tiles[index], "pixel": pixels[index], "cell": empty, "area": area}
        for n in np.unique(fill["tile"]):
            sel = fill["tile"] == n
            pixel, cell, a = weights[n]
            weights[n] = (np.r_[pixel, fill["pixel"][sel]].astype(np.int32),
                          np.r_[cell, fill["cell"][sel]].astype(np.int32),
                          np.r_[a, fill["area"][sel]].astype(F32))
    logger.info("polar fill: %d empty 1/4-degree cells filled", empty.size)
    return {k: np.asarray(v) for k, v in fill.items()}


def write_weights(group: zarr.Group, weights, methods, fill, tile_wet) -> None:
    counts = np.array([w[0].size for w in weights])
    start = np.r_[0, np.cumsum(counts)]
    regrid = group.require_group("regrid")
    comp = compressor()
    for name, dtype in (("pixel", "i4"), ("cell", "i4"), ("area", "f4")):
        k = ("pixel", "cell", "area").index(name)
        data = np.concatenate([w[k] for w in weights]).astype(dtype)
        create_array(regrid, name, data.shape, (4_194_304,), dtype, ("entry",), comp)[:] = data
    create_array(regrid, "tile_start", start.shape, start.shape, "i8", ("tile_edge",))[:] = start
    create_array(regrid, "tile_method", methods.shape, methods.shape, "i1", ("tile",))[:] = methods
    create_array(regrid, "tile_wet", tile_wet.shape, tile_wet.shape, "i1", ("tile",))[:] = tile_wet
    for name, value in fill.items():
        dtype = "f4" if name == "area" else "i4"
        create_array(regrid, f"fill_{name}", value.shape, (max(value.size, 1),), dtype,
                     ("fill",))[:] = value.astype(dtype)
    regrid.attrs.update({
        "description": "1/48-degree pixel -> 1/4-degree cell overlap areas, grouped by 720 tile",
        "pixel": "index within the tile, j*720 + i",
        "cell": "flat 1/4-degree index, y*1440 + x",
        "tile_order": "face-major, then tile row, then tile column",
        "tile_method": "0 = exact lon/lat-rectangle overlap, 1 = 4x4 sub-cells, "
                       "2 = 16x16 sub-cells (tiles reaching 85 degrees)",
        "fill": "polar cells with no overlap, given the fine cell nearest their centre",
    })


def read_weights(group: zarr.Group) -> list[tuple[np.ndarray, object]]:
    """Per-tile `(cells, area matrix)` from a 1/4-degree store's `regrid/` group."""
    regrid = group["regrid"]
    start = regrid["tile_start"][:]
    pixel, cell, area = (regrid[k][:] for k in ("pixel", "cell", "area"))
    return [tile_matrix(pixel[a:b], cell[a:b], area[a:b], TILE_PIXELS)
            for a, b in zip(start[:-1], start[1:])]


def static_sums(source: zarr.Group, matrices, hf2_out, n_cells: int, threads: int):
    """Area sums of hFacC, wet cells, Depth and hFacC_2d over every tile."""
    hfac_src, depth_src, cs_src, sn_src = (open_source_array(source, n)
                                           for n in ("hFacC", "Depth", "CS", "SN"))
    sums = {"cover": np.zeros(n_cells), "hfac_area": np.zeros((LEVELS, n_cells)),
            "wet_area": np.zeros((LEVELS, n_cells)), "Depth": np.zeros(n_cells),
            "hFacC_2d": np.zeros(n_cells)}
    tile_wet = np.zeros(len(TILES), np.int8)

    def job(n):
        f, j0, i0 = TILES[n]
        cells, matrix = matrices[n]
        hfac = np.nan_to_num(hfac_src.read(None, f, j0, j0 + TILE, i0, i0 + TILE))
        cs, sn = (s.read(None, f, j0, j0 + TILE, i0, i0 + TILE) for s in (cs_src, sn_src))
        wet = hfac > 0
        if np.any(np.abs(cs * cs + sn * sn - 1)[wet[0]] > 1e-3):
            raise ValueError(f"CS^2 + SN^2 != 1 on wet cells of tile {TILES[n]}")
        hf2 = bottom_hfac(hfac)
        hf2_out[f, j0:j0 + TILE, i0:i0 + TILE] = hf2
        depth = np.nan_to_num(depth_src.read(None, f, j0, j0 + TILE, i0, i0 + TILE))
        wet0 = wet[0].ravel()
        return n, cells, {
            "cover": np.asarray(matrix.sum(axis=1)).ravel(),
            "hfac_area": apply_weights(matrix, hfac.reshape(LEVELS, -1)),
            "wet_area": apply_weights(matrix, wet.reshape(LEVELS, -1).astype(F32)),
            "Depth": apply_weights(matrix, np.where(wet0, depth.ravel(), F32(0))),
            "hFacC_2d": apply_weights(matrix, np.where(wet0, hf2.ravel(), F32(0))),
        }, bool(wet.any())

    started = _time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        for count, (n, cells, part, any_wet) in enumerate(pool.map(job, range(len(TILES))), 1):
            for name, value in part.items():
                sums[name][..., cells] += value
            tile_wet[n] = any_wet
            if count % 52 == 0:
                logger.info("statics %d/%d tiles (%.0f s)", count, len(TILES), _time.perf_counter() - started)
    return sums, tile_wet


def init(args) -> None:
    products = args.products
    paths = {SAMUDRA: [coarse_path(args, res) for res in RESOLUTIONS] + [rechunk_path(args)],
             NATIVE: [native_path(args)]}
    existing = [p for prod in products for p in paths[prod] if p.exists()]
    if existing and not args.force:
        raise FileExistsError(f"{existing} exist; init would erase them. Pass --force.")
    for prod in products:
        for p in paths[prod] + [marker_dir(args, prod)]:
            shutil.rmtree(p, ignore_errors=True)

    source = zarr.open_group(str(args.source), mode="r")
    n_time = source["Theta"].shape[0]
    missing = find_missing(args.source, n_time)
    logger.info("missing 2D steps: %s", {v: len(s) for v, s in missing.items()})
    time_values = source["time"][:]
    time_attrs = {k: v for k, v in source["time"].attrs.items() if k != "_ARRAY_DIMENSIONS"}
    common = {
        "built_by": "coarsen_llc.py",
        "plan": "docs/10-15-26-coarsen_LLC_plan.md",
        "source_path": str(args.source),
        "filled_time_steps_json": json.dumps(missing),
        "fill_rule": REPAIR_RULE,
        "train_steps": [0, TRAIN_STEPS],
        "val_steps": [TRAIN_STEPS, n_time],
        "compression": "blosc zstd level 3, byte shuffle",
    }
    if SAMUDRA in products:
        init_samudra(args, source, n_time, time_values, time_attrs, common)
    if NATIVE in products:
        init_native(args, source, n_time, time_values, time_attrs, common)
    for prod in products:
        marker_dir(args, prod).mkdir(parents=True, exist_ok=True)


def init_samudra(args, source, n_time, time_values, time_attrs, common) -> None:
    started = _time.perf_counter()
    grids = target_grids(args.target_grid)
    q = grids[QUARTER]
    shape_q, shape_1 = (q["lat"].size, q["lon"].size), (grids[ONE]["lat"].size, grids[ONE]["lon"].size)
    n_cells = shape_q[0] * shape_q[1]

    weights, methods, polar, total_area = build_weights(source, q, args.read_threads)
    radius = float(np.sqrt(total_area / (4 * np.pi)))
    fill = polar_fill(weights, polar, q, radius)
    matrices = [tile_matrix(*w, TILE_PIXELS) for w in weights]

    common = common | {"target_grid_path": str(args.target_grid)}

    rechunk = zarr.open_group(str(rechunk_path(args)), mode="w")
    for var in VARS_2D:
        raw = source[var]
        create_array(rechunk, var, raw.shape, (1, 1, TILE, TILE), "f4",
                     raw.attrs["_ARRAY_DIMENSIONS"], compressor(), dimension_separator="/")
    hf2_out = create_array(rechunk, "hFacC_2d", (N_FACES, FACE_SIZE, FACE_SIZE), (1, TILE, TILE),
                           "f4", ("face", "j", "i"), compressor(), dimension_separator="/")
    hf2_out.attrs["description"] = "open-water fraction of each column's bottom cell (deep=1, land=0)"
    create_array(rechunk, "time", (n_time,), (1024,), time_values.dtype, ("time",))[:] = time_values
    rechunk["time"].attrs.update(time_attrs)
    create_repaired(rechunk, n_time, VARS_2D)
    rechunk.attrs.update(common | {"description": "LLC4320 2D fields rechunked to 720x720 tiles; raw values"})

    sums, tile_wet = static_sums(source, matrices, hf2_out, n_cells, args.read_threads)
    analytic = cell_area(q["lat_b"], q["lon_b"], radius).ravel()
    rel = np.abs(sums["cover"] - analytic) / analytic
    logger.info("cover vs spherical area: median %.1e, 99.9%% %.1e, max %.1e (radius %.1f m)",
                np.median(rel), np.quantile(rel, 0.999), rel.max(), radius)

    for res in RESOLUTIONS:
        grid, shape = grids[res], (shape_q if res == QUARTER else shape_1)
        s = {k: (v.reshape(*v.shape[:-1], *shape) if res == QUARTER else to_one_degree(v, shape_q))
             for k, v in sums.items()}
        lat_b, lon_b, lat = grid["lat_b"], grid["lon_b"], grid["lat"]
        dlon, dlat = np.deg2rad(np.diff(lon_b)), np.deg2rad(np.diff(lat_b))
        statics = {
            "XC": np.broadcast_to(grid["lon"][None, :], shape),
            "YC": np.broadcast_to(lat[:, None], shape),
            "rA": cell_area(lat_b, lon_b, radius),
            "dxF": radius * np.cos(np.deg2rad(lat))[:, None] * dlon[None, :],
            "dyF": np.broadcast_to((radius * dlat)[:, None], shape),
            "Depth": ratio(s["Depth"], s["wet_area"][0]),
            "hFacC_2d": ratio(s["hFacC_2d"], s["wet_area"][0]),
            "land_frac": 1 - ratio(s["hfac_area"], s["cover"]),
            "wet_area": s["wet_area"],
        }
        group = zarr.open_group(str(coarse_path(args, res)), mode="w")
        for var in VARS_3D:
            create_array(group, var, (n_time, LEVELS, *shape), (1, LEVELS, *shape), "f4",
                         ("time", "k", "y", "x"), compressor())
        for var in VARS_2D:
            create_array(group, var, (n_time, *shape), (1, *shape), "f4", ("time", "y", "x"), compressor())
        for name in ("lat", "lon", "lat_b", "lon_b"):
            value = grid[name]
            create_array(group, name, value.shape, value.shape, "f8", (name.replace("lat", "y").replace("lon", "x"),),
                         compressor())[:] = value
        for name, value in statics.items():
            dims = ("k", "y", "x") if value.ndim == 3 else ("y", "x")
            create_array(group, name, value.shape, value.shape, "f4", dims, compressor())[:] = value
        create_array(group, "time", (n_time,), (1024,), time_values.dtype, ("time",))[:] = time_values
        group["time"].attrs.update(time_attrs)
        create_repaired(group, n_time, VARS_3D + VARS_2D)
        group.attrs.update(common | {
            "resolution": res,
            "grid": "Samudra OM4 1-degree grid" + (" split 4x4" if res == QUARTER else ""),
            "earth_radius_m": radius,
            "variables_3d_json": json.dumps(VARS_3D),
            "variables_2d_json": json.dumps(VARS_2D),
            "velocity_frame": "east/north: U, V, oceTAUX, oceTAUY centred and rotated with CS/SN at 1/48 degree",
            "face_edge_velocity": "edge value copied (xgcm boundary='extend')",
            "regrid": "conservative: exact on lon/lat-aligned tiles, 4x4 sub-cells elsewhere "
                      "(16x16 poleward of 85 degrees), polar fill",
            "land_fill": "NaN",
            "land_frac": "1 - overlap-area-weighted mean of 1/48-degree hFacC; 1 = all land",
            "wet_area": "area of wet 1/48-degree cells in the cell (m^2); the averaging weight",
        })
        if res == QUARTER:
            write_weights(group, weights, methods, fill, tile_wet)
        zarr.consolidate_metadata(str(coarse_path(args, res)))
    zarr.consolidate_metadata(str(rechunk_path(args)))
    logger.info("samudra init done in %.0f s: %d sub-cell tiles (%d at 16x16), %d regrid entries",
                _time.perf_counter() - started, int((methods > 0).sum()), int((methods == 2).sum()),
                sum(w[0].size for w in weights))


def init_native(args, source, n_time, time_values, time_attrs, common) -> None:
    """The 1/3-degree native-face store: 16x16 boxes of 1/48-degree cells, 270x270 per face."""
    started, b = _time.perf_counter(), NATIVE_BOX
    n, nt = FACE_SIZE // b, TILE // b
    srcs = {name: open_source_array(source, name) for name in ("XC", "YC", "rA", "dxF", "dyF", "Depth", "hFacC")}
    sums = {"xyz": np.zeros((3, N_FACES, n, n)), "hfac_area": np.zeros((LEVELS, N_FACES, n, n)),
            "wet_area": np.zeros((LEVELS, N_FACES, n, n))}
    sums |= {k: np.zeros((N_FACES, n, n)) for k in ("rA", "dxF", "dyF", "Depth", "hFacC_2d")}

    def job(k):
        f, j0, i0 = TILES[k]
        xc, yc, ra, dx, dy, depth, hfac = (np.nan_to_num(srcs[name].read(None, f, j0, j0 + TILE, i0, i0 + TILE))
                                           for name in srcs)
        wet = hfac > 0
        return k, {
            "xyz": box_sum(unit_vector(xc, yc) * ra, b), "rA": box_sum(ra, b),
            "dxF": box_sum(dx, b), "dyF": box_sum(dy, b),
            "Depth": box_sum(np.where(wet[0], depth * ra, F32(0)), b),
            "hFacC_2d": box_sum(np.where(wet[0], bottom_hfac(hfac) * ra, F32(0)), b),
            "wet_area": box_sum(np.where(wet, ra, F32(0)), b), "hfac_area": box_sum(hfac * ra, b),
        }

    with ThreadPoolExecutor(args.read_threads) as pool:
        for k, part in pool.map(job, range(len(TILES))):
            f, j0, i0 = TILES[k]
            for name, value in part.items():
                sums[name][..., f, j0 // b:j0 // b + nt, i0 // b:i0 // b + nt] = value
    xc, yc = lonlat_from_vector(sums["xyz"])
    statics = {
        "XC": xc, "YC": yc, "rA": sums["rA"], "dxF": sums["dxF"] / b, "dyF": sums["dyF"] / b,
        "Depth": ratio(sums["Depth"], sums["wet_area"][0]),
        "hFacC_2d": ratio(sums["hFacC_2d"], sums["wet_area"][0]),
        "land_frac": 1 - ratio(sums["hfac_area"], sums["rA"]),
        "wet_area": sums["wet_area"],
    }
    group = zarr.open_group(str(native_path(args)), mode="w")
    for var in VARS_3D:
        create_array(group, var, (n_time, LEVELS, N_FACES, n, n), (1, LEVELS, N_FACES, n, n), "f4",
                     ("time", "k", "face", "y", "x"), compressor())
    for var in VARS_2D:
        create_array(group, var, (n_time, N_FACES, n, n), (1, N_FACES, n, n), "f4",
                     ("time", "face", "y", "x"), compressor())
    for name, value in statics.items():
        dims = ("k", "face", "y", "x") if value.ndim == 4 else ("face", "y", "x")
        create_array(group, name, value.shape, value.shape, "f4", dims, compressor())[:] = value
    create_array(group, "time", (n_time,), (1024,), time_values.dtype, ("time",))[:] = time_values
    group["time"].attrs.update(time_attrs)
    create_repaired(group, n_time, VARS_3D + VARS_2D)
    group.attrs.update(common | {
        "resolution": "0p33deg",
        "grid": "native LLC4320 faces (face, j, i), 16x16 boxes of 1/48-degree cells; nominal 1/3 degree",
        "box_size": b,
        "variables_3d_json": json.dumps(VARS_3D),
        "variables_2d_json": json.dumps(VARS_2D),
        "velocity_frame": "east/north: U, V, oceTAUX, oceTAUY centred and rotated with CS/SN at 1/48 degree",
        "face_edge_velocity": "edge value copied (xgcm boundary='extend')",
        "weights": "rA over wet 1/48-degree cells",
        "land_fill": "NaN",
        "land_frac": "1 - rA-weighted mean of 1/48-degree hFacC; 1 = all land",
        "wet_area": "area of wet 1/48-degree cells in the cell (m^2); the averaging weight",
        "XC_YC": "area-weighted mean of the cell's unit vectors, converted back to lon/lat",
        "dxF_dyF": "box width/height along the face's own i/j axes",
    })
    zarr.consolidate_metadata(str(native_path(args)))
    logger.info("native init done in %.0f s", _time.perf_counter() - started)


# ---------------------------------------------------------------- fill

#: Per-process state, set once by `init_worker`.
_W: dict = {}


def init_worker(cfg: dict) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    numcodecs.blosc.use_threads = False
    source = zarr.open_group(cfg["source"], mode="r")
    pool = ThreadPoolExecutor(cfg["read_threads"])
    quarter = zarr.open_group(cfg["coarse"][QUARTER], mode="r")
    rechunk = Path(cfg["rechunk"])
    _W.update(cfg=cfg, source=source, pool=pool)
    names = ("CS", "SN", "rA") if NATIVE in cfg["products"] else ("CS", "SN")
    _W["grid"] = {name: np.stack(list(pool.map(lambda f, a=source[name]: a[f], range(N_FACES))))
                  for name in names}
    _W["wet0"] = zarr.open_array(str(rechunk / "hFacC_2d"), mode="r")[:] > 0
    _W["tile_wet"] = quarter["regrid/tile_wet"][:].astype(bool)
    _W["src3d"] = {v: open_source_array(source, v) for v in VARS_3D}
    _W["mask"] = open_source_array(source, "mask_c")
    _W["src2d"] = {v: source[v] for v in VARS_2D}
    _W["missing"] = {v: set(s) for v, s in json.loads(cfg["missing"]).items()}
    if SAMUDRA in cfg["products"]:
        _W["matrices"] = read_weights(quarter)
        _W["wet_area"] = quarter["wet_area"][:].reshape(LEVELS, -1).astype(np.float64)
        _W["shape"] = quarter["wet_area"].shape[1:]
        _W["out"] = {res: {v: zarr.open_array(str(Path(cfg["coarse"][res]) / v), mode="r+")
                           for v in VARS_3D + VARS_2D} for res in RESOLUTIONS}
        _W["rechunk"] = {v: zarr.open_array(str(rechunk / v), mode="r+", write_empty_chunks=False)
                         for v in VARS_2D}
        _W["out"][QUARTER]["repaired"] = zarr.open_array(str(Path(cfg["coarse"][QUARTER]) / "repaired"), mode="r+")
        _W["out"][ONE]["repaired"] = zarr.open_array(str(Path(cfg["coarse"][ONE]) / "repaired"), mode="r+")
        _W["rechunk"]["repaired"] = zarr.open_array(str(rechunk / "repaired"), mode="r+")
    if NATIVE in cfg["products"]:
        _W["out_native"] = {v: zarr.open_array(str(Path(cfg["native"]) / v), mode="r+")
                            for v in VARS_3D + VARS_2D + ("repaired",)}


def run_step(job: tuple[int, tuple[str, ...]]) -> tuple[int, float, int, float, dict]:
    t, needs = job
    started = _time.perf_counter()
    timing: dict[str, float] = {}
    nbytes = hour(t, needs, timing)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return t, _time.perf_counter() - started, nbytes, rss_gb, timing


def read_planes(t: int, repairs: list) -> tuple[dict[str, np.ndarray], int]:
    """The five whole-globe 2D fields at hour t, bad pixels taken from the nearest clean hour."""
    pool, wet0, source_path = _W["pool"], _W["wet0"], Path(_W["cfg"]["source"])

    def one(var):
        array, missing = _W["src2d"][var], _W["missing"][var]
        n_time = array.shape[0]
        cache: dict[int, np.ndarray] = {}

        def plane(s):
            if s not in cache:
                cache[s] = np.asarray(array[s], dtype=F32)
            return cache[s]

        if t in missing:  # every pixel is missing: copy the whole field
            out = np.full((N_FACES, FACE_SIZE, FACE_SIZE), np.nan, F32)
            masks = {"missing": np.ones(out.shape, bool)}
            sources = repair_pixels(out, masks["missing"], t, n_time,
                                    lambda s: None if s in missing else plane(s), lambda y: np.zeros(y.shape, bool))
            record(repairs, t, var, masks, sources)
        else:
            out = plane(t)
        for f, j0, i0 in TILES:
            tj, ti = slice(j0, j0 + TILE), slice(i0, i0 + TILE)
            wet_finite = wet0[f, tj, ti] if var in ("Eta", "oceQnet", "oceFWflx") else None
            masks = bad_pixels(var, out[f, tj, ti], wet_finite=wet_finite)
            bad = np.logical_or.reduce(list(masks.values()))
            if not bad.any():
                continue
            sources = repair_pixels(
                out[f, tj, ti], bad, t, n_time,
                lambda s: None if s in missing else plane(s)[f, tj, ti],
                lambda y: np.logical_or.reduce(list(bad_pixels(var, y, wet_finite=wet_finite).values())))
            record(repairs, t, var, masks, sources, face=f, j0=j0, i0=i0)
        return var, out, sum(chunk_bytes(source_path / var / f"{s}/0/0/0") for s in cache)

    planes, nbytes = {}, 0
    for var, plane_, size in pool.map(one, VARS_2D):
        planes[var] = plane_
        nbytes += size
    return planes, nbytes


def read_tile(var: str, t: int, n: int, repairs: list, wet: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """One raw (51, 720, 720) tile, bad pixels taken from the nearest clean hour."""
    src, source_path = _W["src3d"][var], Path(_W["cfg"]["source"])
    f, j0, i0 = TILES[n]
    n_time = src.array.shape[0]

    def fetch(s):
        if chunk_bytes(source_path / var / f"{s}/0/{f}/{j0 // TILE}/{i0 // TILE}") == 0 and _W["tile_wet"][n]:
            return None  # a missing chunk reads as NaN, which centring would turn into 0
        return src.read(s, f, j0, j0 + TILE, i0, i0 + TILE)

    def check(y):
        return np.logical_or.reduce(list(bad_pixels(var, y, wet=wet).values()))

    size = chunk_bytes(source_path / var / f"{t}/0/{f}/{j0 // TILE}/{i0 // TILE}")
    cube = fetch(t)
    if cube is None:
        cube = np.full((LEVELS, TILE, TILE), np.nan, F32)
        masks = {"missing": np.ones(cube.shape, bool)}
    else:
        masks = bad_pixels(var, cube, wet=wet)
    bad = np.logical_or.reduce(list(masks.values()))
    if bad.any():
        sources = repair_pixels(cube, bad, t, n_time, fetch, check)
        record(repairs, t, var, masks, sources, face=f, j0=j0, i0=i0)
    return cube, size


def regrid_2d(t: int, products, timing: dict, repairs: list):
    """The five 2D fields (repaired), their 1/4-degree area sums, and native box sums."""
    pool, wet0 = _W["pool"], _W["wet0"]
    cs, sn = _W["grid"]["CS"], _W["grid"]["SN"]
    with timed(timing, "read_2d"):
        planes, nbytes = read_planes(t, repairs)
    raw = {var: planes[var] for var in VARS_2D}
    planes = {var: (p.copy() if var in ("oceTAUX", "oceTAUY") else p) for var, p in planes.items()}
    taux, tauy = planes["oceTAUX"], planes["oceTAUY"]
    list(pool.map(lambda f: center_inplace(taux[f], axis=-1), range(N_FACES)))
    list(pool.map(lambda f: center_inplace(tauy[f], axis=-2), range(N_FACES)))

    def tile(n):
        f, j0, i0 = TILES[n]
        tj, ti = slice(j0, j0 + TILE), slice(i0, i0 + TILE)
        east, north = rotate_to_east_north(taux[f, tj, ti], tauy[f, tj, ti], cs[f, tj, ti], sn[f, tj, ti])
        fields = (planes["Eta"][f, tj, ti], east, north, planes["oceQnet"][f, tj, ti],
                  planes["oceFWflx"][f, tj, ti])
        wet = wet0[f, tj, ti]
        x = np.stack([np.where(wet, a, F32(0)) for a in fields])
        part = apply_weights(_W["matrices"][n][1], x.reshape(len(fields), -1)) if SAMUDRA in products else None
        nat = None
        if NATIVE in products:
            ra = _W["grid"]["rA"][f, tj, ti]
            nat = (box_sum(x * ra, NATIVE_BOX), box_sum(np.where(wet, ra, F32(0)), NATIVE_BOX))
        return n, part, nat

    num = np.zeros((len(VARS_2D), _W["wet_area"].shape[1])) if SAMUDRA in products else None
    nat_num = np.zeros((len(VARS_2D), N_FACES, FACE_SIZE // NATIVE_BOX, FACE_SIZE // NATIVE_BOX))
    nat_den = np.zeros(nat_num.shape[1:])
    with timed(timing, "regrid_2d"):
        for n, part, nat in pool.map(tile, range(len(TILES))):
            if part is not None:
                num[:, _W["matrices"][n][0]] += part
            if nat is not None:
                f, j0, i0 = TILES[n]
                cj, ci = native_slices(j0, i0)
                nat_num[:, f, cj, ci], nat_den[f, cj, ci] = nat
    return raw, num, (nat_num, nat_den), nbytes


def native_slices(j0: int, i0: int) -> tuple[slice, slice]:
    """Native-grid rows/cols covered by the 720 tile at (j0, i0)."""
    nt = TILE // NATIVE_BOX
    return slice(j0 // NATIVE_BOX, j0 // NATIVE_BOX + nt), slice(i0 // NATIVE_BOX, i0 // NATIVE_BOX + nt)


def regrid_3d(t: int, products, timing: dict, repairs: list):
    """Area sums of U (east), V (north), Theta, Salt and the wet area: 1/4-degree and native."""
    pool = _W["pool"]
    cs, sn = _W["grid"]["CS"], _W["grid"]["SN"]
    num = den = None
    if SAMUDRA in products:
        n_cells = _W["wet_area"].shape[1]
        num = np.zeros((len(VARS_3D), LEVELS, n_cells))
        den = np.zeros((LEVELS, n_cells))
    nn = FACE_SIZE // NATIVE_BOX
    nat_num = np.zeros((len(VARS_3D), LEVELS, N_FACES, nn, nn)) if NATIVE in products else None
    nat_den = np.zeros((LEVELS, N_FACES, nn, nn)) if NATIVE in products else None
    nbytes = 0

    for f in range(N_FACES):
        tiles = range(f * 36, (f + 1) * 36)

        def read_face(var):
            face = np.empty((LEVELS, FACE_SIZE, FACE_SIZE), F32)

            def one(n):
                _, j0, i0 = TILES[n]
                cube, size = read_tile(var, t, n, repairs)
                face[:, j0:j0 + TILE, i0:i0 + TILE] = cube
                return size

            return face, sum(pool.map(one, tiles))

        with timed(timing, "read_uv"):
            (u, bu), (v, bv) = read_face("U"), read_face("V")
        nbytes += bu + bv
        # One level per thread: a whole-face in-place centring allocates a hidden copy.
        with timed(timing, "centre"):
            list(pool.map(lambda k: center_inplace(u[k], axis=-1), range(LEVELS)))
            list(pool.map(lambda k: center_inplace(v[k], axis=-2), range(LEVELS)))

        def tile(n):
            _, j0, i0 = TILES[n]
            tj, ti = slice(j0, j0 + TILE), slice(i0, i0 + TILE)
            wet = _W["mask"].read(None, f, j0, j0 + TILE, i0, i0 + TILE) > 0
            theta, size_t = read_tile("Theta", t, n, repairs, wet)
            salt, size_s = read_tile("Salt", t, n, repairs, wet)
            east, north = rotate_to_east_north(u[:, tj, ti], v[:, tj, ti], cs[f, tj, ti], sn[f, tj, ti])
            masked = [np.where(wet, x, F32(0)) for x in (east, north, theta, salt)]
            sam = nat = None
            if SAMUDRA in products:
                matrix = _W["matrices"][n][1]
                sam = (np.stack([apply_weights(matrix, x) for x in masked]),
                       apply_weights(matrix, wet.reshape(LEVELS, -1).astype(F32)))
            if NATIVE in products:
                ra = _W["grid"]["rA"][f, tj, ti]
                nat = (np.stack([box_sum(x * ra, NATIVE_BOX) for x in masked]),
                       box_sum(np.where(wet, ra, F32(0)), NATIVE_BOX))
            return n, sam, nat, size_t + size_s

        with timed(timing, "tiles"):
            for n, sam, nat, size in pool.map(tile, tiles):
                if sam is not None:
                    cells = _W["matrices"][n][0]
                    num[:, :, cells] += sam[0]
                    den[:, cells] += sam[1]
                if nat is not None:
                    _, j0, i0 = TILES[n]
                    cj, ci = native_slices(j0, i0)
                    nat_num[:, :, f, cj, ci], nat_den[:, f, cj, ci] = nat
                nbytes += size
        del u, v
    return num, den, (nat_num, nat_den), nbytes


def write_repairs(t: int, repairs: list) -> None:
    """This hour's repair records, `<coarse-root>/_repairs/<t>.json` (written before any marker)."""
    path = Path(_W["cfg"]["repairs"]) / f"{t:05d}.json"
    if not repairs:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(sorted(repairs, key=lambda r: json.dumps(r, sort_keys=True)), indent=0))
    tmp.replace(path)


def hour(t: int, needs, timing: dict) -> int:
    """Process hour t for every product of the run; write the products in `needs`, and any
    product already done if this hour needed repairs (its earlier output had the bad data)."""
    pool, repairs, products = _W["pool"], [], _W["cfg"]["products"]
    raw2, num2, nat2, bytes2 = regrid_2d(t, products, timing, repairs)
    num3, den, nat3, bytes3 = regrid_3d(t, products, timing, repairs)
    write = [p for p in products if p in needs or repairs]
    codes = {v: int(any(r["var"] == v for r in repairs)) for v in VARS_3D + VARS_2D}
    out = {}
    if SAMUDRA in write:
        wet_area, shape = _W["wet_area"], _W["shape"]
        # Theta's wet cells must be hFacC's (bad tiles were replaced): same weights, mask and sums.
        if not (np.array_equal(den > 0, wet_area > 0) and np.allclose(den, wet_area, rtol=1e-5)):
            bad = np.argwhere((den > 0) != (wet_area > 0))
            raise RuntimeError(f"t={t}: Theta's wet area differs from hFacC's, e.g. at (k, cell) {bad[:3].tolist()}")
        den1, wet1 = to_one_degree(den, shape), to_one_degree(wet_area[0], shape)
        for n, var in enumerate(VARS_3D):
            out[(QUARTER, var)] = ratio(num3[n], den).reshape(LEVELS, *shape)
            out[(ONE, var)] = ratio(to_one_degree(num3[n], shape), den1)
        for n, var in enumerate(VARS_2D):
            out[(QUARTER, var)] = ratio(num2[n], wet_area[0]).reshape(shape)
            out[(ONE, var)] = ratio(to_one_degree(num2[n], shape), wet1)
        row = np.array([codes[v] for v in VARS_3D + VARS_2D], np.int8)
        out[(QUARTER, "repaired")] = out[(ONE, "repaired")] = row
        out[("rechunk", "repaired")] = np.array([codes[v] for v in VARS_2D], np.int8)
    if NATIVE in write:
        for n, var in enumerate(VARS_3D):
            out[(NATIVE, var)] = ratio(nat3[0][n], nat3[1])
        for n, var in enumerate(VARS_2D):
            out[(NATIVE, var)] = ratio(nat2[0][n], nat2[1])
        out[(NATIVE, "repaired")] = np.array([codes[v] for v in VARS_3D + VARS_2D], np.int8)

    def put(item):
        (res, var), value = item
        store = {NATIVE: _W.get("out_native"), "rechunk": _W.get("rechunk")}.get(res) or _W["out"][res]
        store[var][t] = value

    def put_tile(job):
        var, (f, j0, i0) = job
        _W["rechunk"][var][t, f, j0:j0 + TILE, i0:i0 + TILE] = raw2[var][f, j0:j0 + TILE, i0:i0 + TILE]

    with timed(timing, "write"):
        # A rewrite of a finished hour drops its marker first, so nothing (the mover) reads it half-written.
        for product in write:
            if product not in needs:
                (Path(_W["cfg"]["markers"][product]) / f"{t:05d}").unlink(missing_ok=True)
        write_repairs(t, repairs)
        if SAMUDRA in write:
            list(pool.map(put_tile, [(var, tile) for var in VARS_2D for tile in TILES]))
        list(pool.map(put, out.items()))
        for product in write:
            (Path(_W["cfg"]["markers"][product]) / f"{t:05d}").touch()
    timing["repairs"] = timing.get("repairs", 0.0) + len(repairs)
    return bytes2 + bytes3


def pending_steps(args, n_time: int) -> list[tuple[int, tuple[str, ...]]]:
    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        # Every N-th hour, so tasks stay balanced however earlier runs left the markers.
        steps = list(range(args.time_split_index or 0, n_time, args.time_splits))
        if args.stride:
            steps = [t for t in steps if t % args.stride == args.offset]
    done = {product: done_steps(args, product) for product in args.products}
    pending = [(t, needs) for t in steps
               if (needs := tuple(p for p in args.products if t not in done[p]))]
    return pending[: args.limit] if args.limit else pending


def fill(args) -> None:
    group = zarr.open_group(str(coarse_path(args, QUARTER)), mode="r")
    n_time = group["Theta"].shape[0]
    pending = pending_steps(args, n_time)
    logger.info("%d pending hour(s), %d workers x %d threads", len(pending), args.workers, args.read_threads)
    if not pending:
        return
    cfg = {
        "source": str(args.source),
        "products": list(args.products),
        "coarse": {res: str(coarse_path(args, res)) for res in RESOLUTIONS},
        "native": str(native_path(args)),
        "rechunk": str(rechunk_path(args)),
        "markers": {p: str(marker_dir(args, p)) for p in args.products},
        "repairs": str(args.coarse_root / "_repairs"),
        "missing": group.attrs["filled_time_steps_json"],
        "read_threads": args.read_threads,
    }
    for p in args.products:
        marker_dir(args, p).mkdir(parents=True, exist_ok=True)
    (args.coarse_root / "_repairs").mkdir(parents=True, exist_ok=True)
    logger.info("hours needing: %s", {k: sum(k in n for _, n in pending) for k in args.products})
    started = _time.perf_counter()
    total_bytes, step_seconds, max_rss, split = 0, [], 0.0, {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(cfg,)) as pool:
        futures = {pool.submit(run_step, job): job[0] for job in pending}
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
                        "%d/%d done (last t=%d) | %.1f s/hour wall, %.0f s/hour per worker (%s) | "
                        "%.2f GB/s read | worker peak %.1f GB | eta %.1f h",
                        count, len(pending), t, elapsed / count, np.mean(step_seconds),
                        ", ".join(f"{k} {v / count:.0f}" for k, v in split.items()),
                        total_bytes / elapsed / 1e9, max_rss,
                        elapsed / count * (len(pending) - count) / 3600)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    report(args)


def stores(args) -> list[Path]:
    """Every output store that exists: 1/4 and 1 degree, native, rechunk."""
    paths = [coarse_path(args, res) for res in RESOLUTIONS] + [native_path(args), rechunk_path(args)]
    return [p for p in paths if p.exists()]


def prepare(args) -> None:
    """Add `repaired` (and the current repair rule) to stores created before they existed."""
    n_time = zarr.open_group(str(coarse_path(args, QUARTER)), mode="r")["Theta"].shape[0]
    for path in stores(args):
        group = zarr.open_group(str(path), mode="r+")
        if "repaired" not in group:
            create_repaired(group, n_time, VARS_2D if path == rechunk_path(args) else VARS_3D + VARS_2D)
            logger.info("added repaired to %s", path.name)
        group.attrs["fill_rule"] = REPAIR_RULE
        zarr.consolidate_metadata(str(path))


def report(args) -> None:
    """Merge `_repairs/<t>.json` into the single log `repairs.log` and note it in every store."""
    folder = args.coarse_root / "_repairs"
    records = []
    for name in sorted(os.listdir(folder)) if folder.exists() else []:
        if name.endswith(".json"):
            records += json.loads((folder / name).read_text())
    columns = ("t", "var", "kind", "pixels", "copied_from", "face", "j0", "i0")

    def cell(r, c):
        v = r.get(c, "")
        return ",".join(map(str, v)) if isinstance(v, list) else str(v)

    lines = ["\t".join(columns)] + ["\t".join(cell(r, c) for c in columns)
                                     for r in sorted(records, key=lambda r: (r["t"], r["var"]))]
    log = args.coarse_root / "repairs.log"
    tmp = log.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(log)
    hours = sorted({r["t"] for r in records})
    for path in stores(args):
        group = zarr.open_group(str(path), mode="r+")
        group.attrs.update({"fill_rule": REPAIR_RULE, "repairs_log": str(log),
                            "repaired_hours_json": json.dumps(hours), "repair_count": len(records)})
    logger.info("repairs.log: %d records over %d hours", len(records), len(hours))


# ---------------------------------------------------------------- stats


def stats(args) -> None:
    stores = {SAMUDRA: [coarse_path(args, res) for res in RESOLUTIONS], NATIVE: [native_path(args)]}
    for product in args.products:
        steps = sorted(t for t in done_steps(args, product) if t < TRAIN_STEPS)
        if len(steps) < 96:
            raise RuntimeError(f"{product}: only {len(steps)} train hours are done; need 96")
        steps = [steps[i] for i in np.unique(np.linspace(0, len(steps) - 1, 96).round().astype(int))]
        for path in stores[product]:
            stats_for(path, steps)


def stats_for(path: Path, steps: list[int]) -> None:
    """wet_area-weighted per-level (3D) or scalar (2D) mean/std of every variable."""
    group = zarr.open_group(str(path), mode="r+")
    wet_area = group["wet_area"][:].astype(np.float64)
    for var in VARS_3D + VARS_2D:
        w = wet_area if var in VARS_3D else wet_area[0]
        axes = tuple(range(1 if var in VARS_3D else 0, w.ndim))
        s0 = s1 = s2 = 0.0
        for t in steps:
            x = group[var][t].astype(np.float64)
            ok = np.isfinite(x)
            xw = np.where(ok, x, 0.0) * w
            s0 = s0 + np.where(ok, w, 0.0).sum(axis=axes)
            s1 = s1 + xw.sum(axis=axes)
            s2 = s2 + (xw * np.where(ok, x, 0.0)).sum(axis=axes)
        mean = np.atleast_1d(s1 / s0)
        std = np.sqrt(np.maximum(np.atleast_1d(s2 / s0) - mean**2, 0.0))
        dim = ("k",) if var in VARS_3D else ("one",)
        create_array(group, f"{var}_mean", mean.shape, mean.shape, "f4", dim)[:] = mean
        create_array(group, f"{var}_std", std.shape, std.shape, "f4", dim)[:] = std
    group.attrs["stats_steps_json"] = json.dumps(steps)
    group.attrs["stats"] = "wet_area-weighted mean/std over the stats_steps_json hours"
    zarr.consolidate_metadata(str(path))
    logger.info("stats written for %s from %d hours", path.name, len(steps))


# ---------------------------------------------------------------- verify


def conservation_errors(source: zarr.Group, coarse: dict, wet0: np.ndarray, var: str, t: int) -> dict:
    """Worst relative mismatch, per resolution, between global sum(x * rA) on raw wet cells
    and sum(coarse x * wet_area), with the polar-fill entries added to the raw side.
    """
    regrid = coarse[QUARTER]["regrid"]
    fill_tile, fill_pixel, fill_area = (regrid[f"fill_{k}"][:] for k in ("tile", "pixel", "area"))
    levels = LEVELS if var in VARS_3D else 1
    fine = np.zeros(levels)
    plane = None if var in VARS_3D else np.asarray(source[var][t], F32)
    for f in range(N_FACES):
        ra = np.asarray(source["rA"][f], F32)
        if var in VARS_3D:
            x = np.asarray(source[var][t, :, f], F32)
            wet = np.isfinite(x)
        else:
            x, wet = plane[f][None], wet0[f][None]
        fine += (np.where(wet, x, F32(0)) * ra).sum(axis=(-2, -1), dtype=np.float64)
        for n, p, a in zip(fill_tile, fill_pixel, fill_area):
            tf, j0, i0 = TILES[n]
            if tf == f:
                j, i = divmod(int(p), TILE)
                value = x[:, j0 + j, i0 + i]
                fine += np.where(wet[:, j0 + j, i0 + i], value, 0) * a
    errors = {}
    for res, group in coarse.items():
        w = group["wet_area"][:].astype(np.float64)[:levels]
        c = np.asarray(group[var][t], np.float64).reshape(levels, *w.shape[1:])
        total = np.nansum(c * w, axis=(-2, -1))
        errors[res] = float(np.max(np.abs(total - fine) / np.maximum(np.abs(fine), 1e-30)))
    return errors


def verify(args) -> None:
    source = zarr.open_group(str(args.source), mode="r")
    coarse = {res: zarr.open_group(str(coarse_path(args, res)), mode="r") for res in RESOLUTIONS}
    n_time = coarse[QUARTER]["Theta"].shape[0]
    done = sorted(done_steps(args))
    logger.info("%d/%d hours marked done", len(done), n_time)
    problems = []
    for res in RESOLUTIONS:
        for var in VARS_3D + VARS_2D:
            suffix = ".0.0.0" if var in VARS_3D else ".0.0"
            lost = [t for t in done if not (coarse_path(args, res) / var / f"{t}{suffix}").exists()]
            if lost:
                problems.append(f"{STORE_NAMES[res]}/{var}: {len(lost)} marked hours lack a chunk, e.g. {lost[:5]}")
            for entry in os.scandir(coarse_path(args, res) / var):
                if entry.name.endswith(".partial"):
                    os.remove(entry.path)
    wet0 = zarr.open_array(str(rechunk_path(args) / "hFacC_2d"), mode="r")[:] > 0
    rng = np.random.default_rng(0)
    for t in (rng.choice(done, size=min(2, len(done)), replace=False) if done else []):
        for var in ("Eta", "Theta"):
            errors = conservation_errors(source, coarse, wet0, var, int(t))
            logger.info("conservation %s t=%d: %s", var, t, {k: f"{v:.1e}" for k, v in errors.items()})
            problems += [f"{var} t={t} {res}: conservation error {e:.1e}" for res, e in errors.items() if e > 1e-5]
    native_done = sorted(done_steps(args, NATIVE)) if native_path(args).exists() else []
    if native_path(args).exists():
        logger.info("%d/%d hours marked done for the native store", len(native_done), n_time)
        for var in VARS_3D + VARS_2D:
            suffix = ".0.0.0.0" if var in VARS_3D else ".0.0.0"
            lost = [t for t in native_done if not (native_path(args) / var / f"{t}{suffix}").exists()]
            if lost:
                problems.append(f"{NATIVE_NAME}/{var}: {len(lost)} marked hours lack a chunk, e.g. {lost[:5]}")
            for entry in os.scandir(native_path(args) / var):
                if entry.name.endswith(".partial"):
                    os.remove(entry.path)
        if len(native_done) == n_time:
            zarr.consolidate_metadata(str(native_path(args)))
    for problem in problems:
        logger.error(problem)
    if problems:
        raise SystemExit(1)
    if len(done) == n_time:
        for path in [coarse_path(args, res) for res in RESOLUTIONS] + [rechunk_path(args)]:
            zarr.consolidate_metadata(str(path))
        logger.info("verify passed; metadata consolidated")
    else:
        logger.info("verify found no errors, but %d hours are still pending", n_time - len(done))


# ---------------------------------------------------------------- main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=("init", "fill", "prepare", "report", "stats", "verify"))
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--target-grid", type=Path, default=TARGET_GRID)
    p.add_argument("--coarse-root", type=Path, default=COARSE_ROOT)
    p.add_argument("--rechunk-root", type=Path, default=RECHUNK_ROOT)
    p.add_argument("--workers", type=int, default=2, help="Processes; one hour each")
    p.add_argument("--read-threads", type=int, default=32, help="Threads per process")
    p.add_argument("--time-splits", type=int, default=1)
    p.add_argument("--time-split-index", type=int, default=None)
    p.add_argument("--stride", type=int, default=None, help="Only hours with t %% stride == offset")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--steps", default=None, help="Comma-separated hours; overrides the split")
    p.add_argument("--limit", type=int, default=None, help="At most this many pending hours")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--products", default=",".join(PRODUCTS),
                   help="Comma-separated subset of samudra,native")
    p.add_argument("--force", action="store_true", help="init: erase the listed products' stores and markers")
    args = p.parse_args(argv)
    args.products = tuple(x for x in args.products.split(",") if x)
    if not set(args.products) <= set(PRODUCTS):
        raise ValueError(f"--products must be a subset of {PRODUCTS}")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    numcodecs.blosc.use_threads = False
    {"init": init, "fill": fill, "prepare": prepare, "report": report, "stats": stats,
     "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
