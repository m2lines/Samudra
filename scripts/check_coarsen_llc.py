#!/usr/bin/env python
"""Checks of coarsen_llc.py output (plan: Verification items 2-3).

Always: store layout, and the regrid weights (each 1/48-degree cell shares out
exactly its rA; each 1/4-degree cell's total against its spherical area; no
empty cells). For each --steps hour:

- the rechunk equals the raw globe (or the mean of its neighbours on a filled
  step);
- Eta and Theta are conserved globally;
- one aligned cell is recomputed independently from raw edges;
- one Arctic cell is compared with an 8x8 sub-cell estimate;
- coarse NaNs match `wet_area == 0`;
- there are no jumps along LLC face boundaries;
- quick-look PNGs are saved.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import zarr

from coarsen_llc import (
    F32, LEVELS, NATIVE_BOX, NATIVE_NAME, ONE, QUARTER, RECHUNK_NAME, RESOLUTIONS, SOURCE, STORE_NAMES,
    TILE, TILES, TARGET_GRID, VARS_2D, VARS_3D, conservation_errors, target_grids,
)

from ocean_emulators.llc_coarse import (  # noqa: E402
    center_inplace, corner_grid, rotate_to_east_north, subcell_weights,
)

logger = logging.getLogger(__name__)
FAILURES: list[str] = []


def check(ok: bool, message: str) -> None:
    (logger.info if ok else logger.error)("%s %s", "PASS" if ok else "FAIL", message)
    if not ok:
        FAILURES.append(message)


def check_layout(coarse: dict, rechunk: zarr.Group, grids: dict) -> None:
    n_time = coarse[QUARTER]["Theta"].shape[0]
    for res, group in coarse.items():
        shape = (grids[res]["lat"].size, grids[res]["lon"].size)
        for var in VARS_3D + VARS_2D:
            a = group[var]
            want = (n_time, LEVELS, *shape) if var in VARS_3D else (n_time, *shape)
            ok = a.shape == want and a.chunks == (1, *want[1:]) and a.dtype == np.float32
            ok &= a.compressor.cname == "zstd" and a.compressor.clevel == 3 and a.compressor.shuffle == 1
            check(ok, f"{STORE_NAMES[res]}/{var} {a.shape} chunks {a.chunks} {a.compressor}")
        check(np.array_equal(group["lat"][:], grids[res]["lat"]) and np.array_equal(group["lon"][:], grids[res]["lon"]),
              f"{STORE_NAMES[res]} lat/lon equal the target grid's")
    for var in VARS_2D:
        a = rechunk[var]
        check(a.chunks == (1, 1, TILE, TILE) and a._dimension_separator == "/" and a.compressor.clevel == 3,
              f"rechunk/{var} chunks {a.chunks} separator {a._dimension_separator!r} {a.compressor}")


def check_weights(source: zarr.Group, quarter: zarr.Group) -> None:
    regrid = quarter["regrid"]
    start, method = regrid["tile_start"][:], regrid["tile_method"][:]
    pixel, cell, area = (regrid[k][:] for k in ("pixel", "cell", "area"))
    fill_tile, fill_pixel, fill_area = (regrid[f"fill_{k}"][:] for k in ("tile", "pixel", "area"))
    worst = 0.0
    for n, (a, b) in enumerate(zip(start[:-1], start[1:])):
        f, j0, i0 = TILES[n]
        share = np.bincount(pixel[a:b], weights=area[a:b], minlength=TILE * TILE)
        sel = fill_tile == n
        share -= np.bincount(fill_pixel[sel], weights=fill_area[sel], minlength=TILE * TILE)
        ra = np.asarray(source["rA"][f, j0:j0 + TILE, i0:i0 + TILE], np.float64).ravel()
        worst = max(worst, float(np.max(np.abs(share - ra) / ra)))
    check(worst < 1e-5, f"each 1/48-degree cell shares out its rA (worst relative error {worst:.1e})")
    n_cells = quarter["rA"].size
    cover = np.bincount(cell, weights=area, minlength=n_cells)
    rel = np.abs(cover - quarter["rA"][:].ravel()) / quarter["rA"][:].ravel()
    curved = np.zeros(n_cells, bool)
    for n in np.flatnonzero(method):
        curved[cell[start[n]:start[n + 1]]] = True
    if fill_tile.size:
        curved[regrid["fill_cell"][:]] = True
    check(cover.min() > 0, f"every 1/4-degree cell has area ({np.sum(cover == 0)} empty)")
    check(rel[~curved].max() < 1e-3,
          f"cells from aligned tiles: area vs spherical max rel error {rel[~curved].max():.1e}")
    logger.info("INFO cells touched by sub-cell tiles or polar fill (%d): rel area error median %.1e, "
                "99.9%% %.1e, max %.1e; %d sub-cell tiles (%d at 16x16), %d polar-fill cells", curved.sum(),
                np.median(rel[curved]), np.quantile(rel[curved], 0.999), rel[curved].max(),
                int((method > 0).sum()), int((method == 2).sum()), fill_tile.size)
    lat = np.repeat(quarter["lat"][:], quarter["lon"].size)
    for lo, hi in ((-90, 85), (85, 89), (89, 90.1)):
        band = curved & (lat >= lo) & (lat < hi) & (quarter["land_frac"][0].ravel() < 1)
        if band.any():
            logger.info("INFO ocean cells from sub-cells, lat %g..%g: rel area error max %.1e, 99%% %.1e",
                        lo, min(hi, 90), rel[band].max(), np.quantile(rel[band], 0.99))


def check_rechunk(source, coarse, rechunk, t: int, missing: dict) -> None:
    for var in VARS_2D:
        if t in missing[var]:
            s = t - 1 if t > 0 else t + 1  # the previous hour (assumed clean here)
            expected = np.asarray(source[var][s], dtype=F32)
            names = list(rechunk["repaired"].attrs["variables"])
            check(rechunk["repaired"][t, names.index(var)] == 1, f"t={t} {var} marked repaired in the rechunk")
        else:
            expected = np.asarray(source[var][t], dtype=F32)
        check(np.array_equal(rechunk[var][t], expected, equal_nan=True),
              f"t={t} rechunk/{var} equals raw{' (filled)' if t in missing[var] else ''}")


def hand_aligned_cell(source, quarter, grid, t: int) -> None:
    """Recompute one face-1 cell from raw edges: rows split by sin(lat), columns by centre."""
    f, y = 1, int(np.searchsorted(grid["lat"], -10.0))
    x = int(np.searchsorted(grid["lon"], 10.0))
    s, n, w, e = grid["lat_b"][y], grid["lat_b"][y + 1], grid["lon_b"][x], grid["lon_b"][x + 1]
    yg = np.asarray(source["YG"][f, :, 0], np.float64)   # face 1: rows of constant latitude
    xc = np.asarray(source["XC"][f, 0, :], np.float64)
    cols = np.flatnonzero((np.mod(xc, 360) >= w) & (np.mod(xc, 360) < e))
    south, north = yg, np.r_[yg[1:], 2 * yg[-1] - yg[-2]]
    overlap = np.clip(np.sin(np.deg2rad(np.minimum(north, n))) - np.sin(np.deg2rad(np.maximum(south, s))), 0, None)
    frac = overlap / (np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south)))
    rows = np.flatnonzero(frac > 0)
    jr, ic = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    ra = np.asarray(source["rA"][f, jr, ic], np.float64) * frac[rows][:, None]
    for k in (0, 20):
        theta = np.asarray(source["Theta"][t, k, f, jr, ic], np.float64)
        u = np.asarray(source["U"][t, k, f, jr, slice(cols[0], cols[-1] + 2)], F32)
        v = np.asarray(source["V"][t, k, f, slice(rows[0], rows[-1] + 2), ic], F32)
        uc, vc = center_inplace(u, -1)[:, :-1], center_inplace(v, -2)[:-1]
        east, north_v = rotate_to_east_north(uc, vc, source["CS"][f, jr, ic], source["SN"][f, jr, ic])
        wet = np.isfinite(theta)
        mean = {name: float((np.where(wet, a, 0) * ra).sum() / (ra * wet).sum())
                for name, a in (("Theta", theta), ("U", east), ("V", north_v))}
        got = {name: float(quarter[name][t, k, y, x]) for name in mean}
        check(all(np.isclose(got[m], mean[m], rtol=1e-4, atol=1e-6) for m in mean),
              f"t={t} k={k} aligned cell (lat {grid['lat'][y]:.2f}, lon {grid['lon'][x]:.3f}) on face 1: "
              + ", ".join(f"{m} {got[m]:.5g} vs {mean[m]:.5g}" for m in mean))
    eta = np.asarray(source["Eta"][t, f], np.float64)[jr, ic]
    wet0 = np.asarray(source["hFacC"][0, f, jr, ic]) > 0
    mean = float((np.where(wet0, eta, 0) * ra).sum() / (ra * wet0).sum())
    got = float(quarter["Eta"][t, y, x])
    check(np.isclose(got, mean, rtol=1e-4, atol=1e-6), f"t={t} aligned cell Eta {got:.6g} vs {mean:.6g}")


def hand_arctic_cell(source, quarter, grid, t: int) -> None:
    """Compare one cap cell with an 8x8 sub-cell estimate over the same neighbourhood."""
    regrid = quarter["regrid"]
    start = regrid["tile_start"][:]
    y = int(np.searchsorted(grid["lat"], 85.0))
    x = int(np.searchsorted(grid["lon"], 30.0))
    target = y * grid["lon"].size + x
    hits = []
    for n in range(6 * 36, 7 * 36):
        a, b = start[n], start[n + 1]
        cell = regrid["cell"][a:b]
        sel = np.flatnonzero(cell == target)
        if sel.size:
            hits += [(n, p) for p in regrid["pixel"][a:b][sel]]
    n0 = hits[0][0]
    f, j0, i0 = TILES[n0]
    js, is_ = zip(*(divmod(int(p), TILE) for n, p in hits if n == n0))
    ja, jb = max(min(js) - 3, 0), min(max(js) + 4, TILE)
    ia, ib = max(min(is_) - 3, 0), min(max(is_) + 4, TILE)
    xg = source["XG"][f, j0 + ja:j0 + jb + 1, i0 + ia:i0 + ib + 1]
    yg = source["YG"][f, j0 + ja:j0 + jb + 1, i0 + ia:i0 + ib + 1]
    lon_c, lat_c = corner_grid(xg, yg)
    lon_c, lat_c = lon_c[:jb - ja + 1, :ib - ia + 1], lat_c[:jb - ja + 1, :ib - ia + 1]
    ra = source["rA"][f, j0 + ja:j0 + jb, i0 + ia:i0 + ib]
    pixel, cell, area = subcell_weights(lon_c, lat_c, ra, grid["lat_b"], grid["lon"].size, n_sub=8)
    sel = cell == target
    theta = np.asarray(source["Theta"][t, 0, f, j0 + ja:j0 + jb, i0 + ia:i0 + ib], np.float64).ravel()
    wet = np.isfinite(theta[pixel[sel]])
    estimate = float((theta[pixel[sel]][wet] * area[sel][wet]).sum() / area[sel][wet].sum())
    got = float(quarter["Theta"][t, 0, y, x])
    spread = float(np.nanmax(theta[pixel[sel]]) - np.nanmin(theta[pixel[sel]]))
    check(abs(got - estimate) <= 0.02 * spread + 1e-6,
          f"t={t} Arctic cell (lat {grid['lat'][y]:.2f}, lon {grid['lon'][x]:.3f}) Theta_0 {got:.5f} vs "
          f"8x8 estimate {estimate:.5f} (spread {spread:.4f})")


def seam_ratios(field: np.ndarray, grid: dict) -> dict[str, float]:
    """Mean |jump| across LLC face boundaries over the mean |jump| a few cells away."""
    lat = grid["lat"]
    band = (lat > -55) & (lat < 60)
    out = {}
    for lon in (322.0, 52.0, 142.0, 232.0):  # -38, 52, 142, -128 degrees E
        c = int(round(lon / 360 * grid["lon"].size))
        a = field[band]
        seam = np.nanmean(np.abs(a[:, c] - a[:, c - 1]))
        inside = np.nanmean(np.abs(a[:, c + 5] - a[:, c + 4]))
        out[f"{lon - 360 if lon > 180 else lon:g}E"] = float(seam / inside)
    r = int(np.searchsorted(grid["lat_b"], 67.5)) - 1  # the row holding the cap boundary
    seam = max(np.nanmean(np.abs(field[r] - field[r - 1])), np.nanmean(np.abs(field[r + 1] - field[r])))
    out["67.5N"] = float(seam / np.nanmean(np.abs(field[r - 4] - field[r - 5])))
    return out


def check_hour(source, coarse, rechunk, grids, t: int, png_dir: Path) -> None:
    wet0 = rechunk["hFacC_2d"][:] > 0  # hFacC[0] > 0, without reading all 51 levels
    for var in ("Eta", "Theta"):
        errors = conservation_errors(source, coarse, wet0, var, t)
        check(all(e < 1e-5 for e in errors.values()),
              f"t={t} global {var} conservation: " + ", ".join(f"{r} {e:.1e}" for r, e in errors.items()))
    for res, group in coarse.items():
        land = group["wet_area"][:] == 0
        ok = all(np.array_equal(np.isnan(group[v][t]), land) for v in VARS_3D)
        ok &= all(np.array_equal(np.isnan(group[v][t]), land[0]) for v in VARS_2D)
        check(ok, f"t={t} {res}: coarse NaNs equal wet_area == 0 at every level")
        lf = group["land_frac"][:]
        check(np.array_equal(lf == 1, land), f"{res}: land_frac == 1 exactly where wet_area == 0")
    q, grid = coarse[QUARTER], grids[QUARTER]
    hand_aligned_cell(source, q, grid, t)
    hand_arctic_cell(source, q, grid, t)
    for var in ("U", "V", "Theta"):
        ratios = seam_ratios(q[var][t, 0], grid)
        check(all(r < 2.5 for r in ratios.values()),
              f"t={t} {var}_0 jump across face boundaries / jump inside: "
              + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items()))
    save_pngs(q, grid, t, png_dir)


def save_pngs(q: zarr.Group, grid: dict, t: int, png_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = {"Theta_0": q["Theta"][t, 0], "U_0 (east)": q["U"][t, 0], "V_0 (north)": q["V"][t, 0],
              "Eta": q["Eta"][t], "land_frac k=0": q["land_frac"][0], "land_frac k=40": q["land_frac"][40]}
    png_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    for ax, (title, field) in zip(axes.flat, panels.items()):
        lo, hi = np.nanpercentile(field, [2, 98])
        diverging = title.startswith(("U", "V"))
        if diverging:
            lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        im = ax.pcolormesh(grid["lon_b"], grid["lat_b"], field, vmin=lo, vmax=hi,
                           cmap="RdBu_r" if diverging else "viridis", shading="flat")
        ax.set_title(f"{title}, t={t}, Samudra 1/4 deg")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(png_dir / f"samudra_quicklook_t{t:05d}.png", dpi=80)
    plt.close(fig)

    north = grid["lat_b"] >= 60
    rows = slice(int(np.argmax(north)), None)
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    for k, (title, field) in enumerate((("Theta_0", q["Theta"][t, 0]), ("U_0 (east)", q["U"][t, 0]))):
        ax = fig.add_subplot(1, 2, k + 1, projection="polar")
        sub = field[rows.start:]
        lo, hi = np.nanpercentile(sub, [2, 98])
        if title.startswith("U"):
            lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        im = ax.pcolormesh(np.deg2rad(grid["lon_b"]), 90 - grid["lat_b"][rows.start:], sub, vmin=lo, vmax=hi,
                           cmap="RdBu_r" if title.startswith("U") else "viridis", shading="flat")
        ax.set_title(f"{title} north of 60N, t={t}")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(png_dir / f"samudra_arctic_t{t:05d}.png", dpi=80)
    plt.close(fig)
    logger.info("saved quick-looks to %s", png_dir)


def check_native(source, native: zarr.Group, rechunk: zarr.Group, t: int) -> None:
    """Per-face conservation, NaN pattern, and one face-9 box recomputed from raw."""
    wet_area = native["wet_area"][:].astype(np.float64)
    wet0 = rechunk["hFacC_2d"][:] > 0
    eta = np.asarray(source["Eta"][t], F32)
    worst = {"Eta": 0.0, "Theta": 0.0}
    for f in range(13):
        ra = np.asarray(source["rA"][f], F32)
        pairs = [("Eta", eta[f][None], wet0[f][None], 1)]
        if f in (1, 6, 9):
            theta = np.asarray(source["Theta"][t, :, f], F32)
            pairs.append(("Theta", theta, np.isfinite(theta), LEVELS))
        for var, x, wet, levels in pairs:
            fine = (np.where(wet, x, F32(0)) * ra).sum(axis=(-2, -1), dtype=np.float64)
            c = np.asarray(native[var][t], np.float64).reshape(levels, 13, *wet_area.shape[-2:])[:, f]
            total = np.nansum(c * wet_area[:levels, f], axis=(-2, -1))
            worst[var] = max(worst[var], float(np.max(np.abs(total - fine) / np.maximum(np.abs(fine), 1e-30))))
    check(all(e < 1e-5 for e in worst.values()),
          f"t={t} native per-face conservation: Eta {worst['Eta']:.1e} (13 faces), Theta {worst['Theta']:.1e} (faces 1,6,9)")
    land = wet_area == 0
    ok = all(np.array_equal(np.isnan(native[v][t]), land) for v in VARS_3D)
    ok &= all(np.array_equal(np.isnan(native[v][t]), land[0]) for v in VARS_2D)
    check(ok, f"t={t} native: NaNs equal wet_area == 0")
    # Face 9 is rotated (+i south, +j east); one 16x16 box, centred and rotated by hand.
    f, y, x, b = 9, 150, 150, NATIVE_BOX
    j0, i0 = y * b, x * b
    for k in (0, 20):
        u = np.nan_to_num(np.asarray(source["U"][t, k, f, j0:j0 + b, i0:i0 + b + 1], np.float64))
        v = np.nan_to_num(np.asarray(source["V"][t, k, f, j0:j0 + b + 1, i0:i0 + b], np.float64))
        uc, vc = (u[:, :-1] + u[:, 1:]) / 2, (v[:-1] + v[1:]) / 2
        cs, sn = (np.asarray(source[n][f, j0:j0 + b, i0:i0 + b], np.float64) for n in ("CS", "SN"))
        theta = np.asarray(source["Theta"][t, k, f, j0:j0 + b, i0:i0 + b], np.float64)
        w = np.where(np.isfinite(theta), np.asarray(source["rA"][f, j0:j0 + b, i0:i0 + b], np.float64), 0)
        want = {"U": uc * cs - vc * sn, "V": uc * sn + vc * cs, "Theta": np.nan_to_num(theta)}
        want = {n: float((a * w).sum() / w.sum()) for n, a in want.items()}
        got = {n: float(native[n][t, k, f, y, x]) for n in want}
        check(all(np.isclose(got[n], want[n], rtol=1e-4, atol=1e-6) for n in want),
              f"t={t} k={k} native face-9 box ({y},{x}): " + ", ".join(f"{n} {got[n]:.5g} vs {want[n]:.5g}" for n in want))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--target-grid", type=Path, default=TARGET_GRID)
    p.add_argument("--coarse-root", type=Path, required=True)
    p.add_argument("--rechunk-root", type=Path, required=True)
    p.add_argument("--steps", default="", help="Comma-separated hours to check")
    p.add_argument("--skip-weights", action="store_true")
    p.add_argument("--native-only", action="store_true", help="Only the native 1/3-degree checks")
    p.add_argument("--png-dir", type=Path, default=None)
    args = p.parse_args()
    source = zarr.open_group(str(args.source), mode="r")
    coarse = {res: zarr.open_group(str(args.coarse_root / STORE_NAMES[res]), mode="r") for res in RESOLUTIONS}
    rechunk = zarr.open_group(str(args.rechunk_root / RECHUNK_NAME), mode="r")
    grids = target_grids(args.target_grid)
    missing = {v: set(s) for v, s in json.loads(coarse[ONE].attrs["filled_time_steps_json"]).items()}
    native = zarr.open_group(str(args.coarse_root / NATIVE_NAME), mode="r") \
        if (args.coarse_root / NATIVE_NAME).exists() else None
    if not args.native_only:
        check_layout(coarse, rechunk, grids)
        if not args.skip_weights:
            check_weights(source, coarse[QUARTER])
    for t in (int(s) for s in args.steps.split(",") if s):
        if not args.native_only:
            check_rechunk(source, coarse, rechunk, t, missing)
            check_hour(source, coarse, rechunk, grids, t, args.png_dir or args.coarse_root / "_check")
        if native is not None:
            check_native(source, native, rechunk, t)
    logger.info("%d failure(s)", len(FAILURES))
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
