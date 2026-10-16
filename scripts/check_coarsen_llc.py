#!/usr/bin/env python
"""One-step checks of coarsen_llc.py output (plan: Verification items 2-3).

For each --boundary-steps step: the rechunk equals the raw globe (or the mean of
its neighbours on a filled step) tile for tile, Eta is conserved, coarse NaNs
match wet_frac, and the wind stress is hand-checked. For each
--prognostic-steps step: Theta is conserved on faces 1, 6, 9, one coarse cell
per face is recomputed independently from raw, raw Theta NaNs match mask_c,
coarse NaNs match wet_frac, surface U/V are continuous across face seams, and a
quick-look PNG is saved.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import zarr

from coarsen_llc import (
    BOXES, CHANNELS_3D, COARSE_NAMES, F32, LEVELS, ONE, QUARTER, RECHUNK_NAME, SOURCE, TILE,
    VARS_2D, conservation_error,
)

from ocean_emulators.llc_coarse import N_FACES, faces_to_latlon  # noqa: E402

logger = logging.getLogger(__name__)
CHECK_FACES = (1, 6, 9)
FAILURES: list[str] = []


def check(ok: bool, message: str) -> None:
    (logger.info if ok else logger.error)("%s %s", "PASS" if ok else "FAIL", message)
    if not ok:
        FAILURES.append(message)


def check_layout(coarse: dict, rechunk: zarr.Group, n_time: int) -> None:
    for b, group in coarse.items():
        n = 4320 // b
        expected = {
            "vars_3d": ((n_time, len(CHANNELS_3D), N_FACES, n, n), (1, len(CHANNELS_3D), N_FACES, n, n)),
            "vars_2d": ((n_time, len(VARS_2D), N_FACES, n, n), (1, len(VARS_2D), N_FACES, n, n)),
            "wet_frac": ((LEVELS, N_FACES, n, n), (LEVELS, 1, n, n)),
            "XC": ((N_FACES, n, n), (N_FACES, n, n)),
        }
        for name, (shape, chunks) in expected.items():
            a = group[name]
            check(a.shape == shape and a.chunks == chunks and a.dtype == np.float32,
                  f"{COARSE_NAMES[b]}/{name} shape {a.shape} chunks {a.chunks} {a.dtype}")
    for var in VARS_2D:
        a = rechunk[var]
        check(a.chunks == (1, 1, TILE, TILE) and a._dimension_separator == "/",
              f"rechunk/{var} chunks {a.chunks} separator {a._dimension_separator!r}")


def hand_cell(source, t: int, f: int, k: int, y: int, x: int, b: int,
              u_name: str, v_name: str) -> tuple[float, float]:
    """East/north mean of one coarse cell, from raw arrays with plain numpy."""
    j0, i0 = y * b, x * b
    j1, i1 = j0 + b, i0 + b
    sel3 = (t, k, f) if u_name == "U" else (t, f)

    def raw(name, rows, cols):
        return np.asarray(source[name][sel3 + (rows, cols)], dtype=np.float64)

    # One extra column of U and row of V, copying the edge at the face boundary.
    u = np.nan_to_num(raw(u_name, slice(j0, j1), slice(i0, min(i1 + 1, 4320))))
    v = np.nan_to_num(raw(v_name, slice(j0, min(j1 + 1, 4320)), slice(i0, i1)))
    if u.shape[1] == b:
        u = np.concatenate([u, u[:, -1:]], axis=1)
    if v.shape[0] == b:
        v = np.concatenate([v, v[-1:]], axis=0)
    uc, vc = (u[:, :-1] + u[:, 1:]) / 2, (v[:-1] + v[1:]) / 2
    cs = np.asarray(source["CS"][f, j0:j1, i0:i1], dtype=np.float64)
    sn = np.asarray(source["SN"][f, j0:j1, i0:i1], dtype=np.float64)
    ra = np.asarray(source["rA"][f, j0:j1, i0:i1], dtype=np.float64)
    if u_name == "U":
        wet = np.isfinite(np.asarray(source["Theta"][t, k, f, j0:j1, i0:i1]))
    else:
        wet = np.asarray(source["hFacC"][0, f, j0:j1, i0:i1]) > 0
    w = np.where(wet, ra, 0)
    east, north = uc * cs - vc * sn, uc * sn + vc * cs
    return float((east * w).sum() / w.sum()), float((north * w).sum() / w.sum())


def pick_cell(wet_frac: np.ndarray, f: int, k: int) -> tuple[int, int]:
    """A wet 1/4-degree cell at a tile's north-east corner, so tile-edge centring is used."""
    ys, xs = np.nonzero(wet_frac[k, f] > 0.5)
    corner = (ys % 60 == 59) & (xs % 60 == 59) & (ys < 359) & (xs < 359)
    pool = np.flatnonzero(corner) if corner.any() else np.arange(ys.size)
    i = pool[len(pool) // 2]
    return int(ys[i]), int(xs[i])


def check_boundary_step(source, coarse, rechunk, t: int, missing: dict) -> None:
    for var in VARS_2D:
        if t in missing[var]:
            steps = [s for s in (t - 1, t + 1) if 0 <= s < source[var].shape[0]]
            expected = np.asarray(source[var][steps[0]], dtype=F32)
            for s in steps[1:]:
                expected += source[var][s]
            if len(steps) > 1:
                expected *= F32(1 / len(steps))
            check(all(t in json.loads(g.attrs["filled_time_steps_json"])[var]
                      for g in (*coarse.values(), rechunk)),
                  f"t={t} {var} listed as filled in all three stores")
        else:
            expected = np.asarray(source[var][t], dtype=F32)
        got = rechunk[var][t]
        check(np.array_equal(got, expected, equal_nan=True),
              f"t={t} rechunk/{var} equals raw{' (filled)' if t in missing[var] else ''}")
    hf2 = rechunk["hFacC_2d"][:]
    for f in range(N_FACES):
        if "Eta" in missing and t in missing["Eta"]:
            break
        for b in BOXES:
            err = conservation_error(source, coarse[b], hf2[f] > 0, "Eta", t, f, b)
            if f in CHECK_FACES or err > 1e-5:
                check(err < 1e-5, f"t={t} Eta conservation face {f} box {b}: {err:.1e}")
    for b in BOXES:
        nan = np.isnan(coarse[b]["vars_2d"][t])
        land = coarse[b]["wet_frac"][0] == 0
        check(all(np.array_equal(nan[c], land) for c in range(len(VARS_2D))),
              f"t={t} box {b} vars_2d NaNs match wet_frac[0] == 0")
    if any(t in missing[v] for v in ("oceTAUX", "oceTAUY")):
        return
    wet_frac = coarse[QUARTER]["wet_frac"][:]
    for f in CHECK_FACES:
        y, x = pick_cell(wet_frac, f, 0)
        east, north = hand_cell(source, t, f, 0, y, x, QUARTER, "oceTAUX", "oceTAUY")
        got = coarse[QUARTER]["vars_2d"][t, 1:3, f, y, x]
        check(np.allclose(got, [east, north], rtol=1e-4, atol=1e-7),
              f"t={t} face {f} cell ({y},{x}) wind stress east/north {got} vs {east:.4g}, {north:.4g}")


def seam_ratios(field: np.ndarray, n: int) -> dict[str, float]:
    """Mean |jump| across a face seam over the mean |jump| a few cells inside."""
    seams = {  # name: (axis, last index before the seam, slice along the seam)
        "11|1": (1, -1, slice(n, 2 * n)),
        "2|5": (1, n - 1, slice(2 * n, 3 * n)),
        "5|7": (1, 2 * n - 1, slice(2 * n, 3 * n)),
        "8|7": (0, 2 * n - 1, slice(2 * n, 3 * n)),
    }
    out = {}
    for name, (axis, last, along) in seams.items():
        a = np.moveaxis(field, axis, -1)[along]
        first = (last + 1) % a.shape[-1]
        seam = np.nanmean(np.abs(a[:, last] - a[:, first]))
        inside = np.nanmean(np.abs(a[:, first + 4] - a[:, first + 5]))
        out[name] = float(seam / inside)
    return out


def check_prognostic_step(source, coarse, t: int, png_dir: Path) -> None:
    wet_frac = {b: coarse[b]["wet_frac"][:] for b in BOXES}
    for f in CHECK_FACES:
        for b in BOXES:
            err = conservation_error(source, coarse[b], None, "Theta", t, f, b)
            check(err < 1e-5, f"t={t} Theta conservation face {f} box {b} (all levels): {err:.1e}")
    for f in CHECK_FACES:
        for k in (0, 20):
            y, x = pick_cell(wet_frac[QUARTER], f, k)
            east, north = hand_cell(source, t, f, k, y, x, QUARTER, "U", "V")
            c = [CHANNELS_3D.index(f"U_{k}"), CHANNELS_3D.index(f"V_{k}")]
            got = coarse[QUARTER]["vars_3d"][t, c, f, y, x]
            check(np.allclose(got, [east, north], rtol=1e-4, atol=1e-6),
                  f"t={t} face {f} k={k} cell ({y},{x}) U/V east/north {got} vs {east:.5g}, {north:.5g}")
        for (j0, i0) in ((0, 0), (1440, 2160), (3600, 3600)):
            theta = np.asarray(source["Theta"][t, :, f, j0:j0 + TILE, i0:i0 + TILE])
            mask = np.asarray(source["mask_c"][:, f, j0:j0 + TILE, i0:i0 + TILE]) > 0
            check(np.array_equal(np.isfinite(theta), mask),
                  f"t={t} face {f} tile ({j0},{i0}) Theta NaNs equal mask_c == 0")
    for b in BOXES:
        x = coarse[b]["vars_3d"][t]
        land = wet_frac[b] == 0
        ok = all(np.array_equal(np.isnan(x[n * LEVELS:(n + 1) * LEVELS]), land) for n in range(4))
        check(ok, f"t={t} box {b} vars_3d NaNs match wet_frac == 0 at every level")
    n = 4320 // QUARTER
    for name in ("U_0", "V_0"):
        grid, _ = faces_to_latlon(coarse[QUARTER]["vars_3d"][t, CHANNELS_3D.index(name)])
        ratios = seam_ratios(grid, n)
        check(all(r < 2.5 for r in ratios.values()),
              f"t={t} {name} seam jump / interior jump: "
              + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items()))
    save_png(coarse, t, png_dir)


def save_png(coarse, t: int, png_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group = coarse[QUARTER]
    panels = {
        "Theta_0": group["vars_3d"][t, CHANNELS_3D.index("Theta_0")],
        "U_0 (east)": group["vars_3d"][t, CHANNELS_3D.index("U_0")],
        "V_0 (north)": group["vars_3d"][t, CHANNELS_3D.index("V_0")],
        "Eta": group["vars_2d"][t, 0],
        "wet_frac k=0": group["wet_frac"][0],
        "wet_frac k=40": group["wet_frac"][40],
    }
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    for ax, (title, field) in zip(axes.flat, panels.items()):
        grid, _ = faces_to_latlon(field)
        lo, hi = np.nanpercentile(grid, [2, 98])
        if title.startswith(("U", "V")):
            lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        im = ax.imshow(grid, origin="lower", vmin=lo, vmax=hi,
                       cmap="RdBu_r" if title.startswith(("U", "V")) else "viridis")
        ax.set_title(f"{title}, t={t}, 1/4 deg stitched (no Arctic)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    png_dir.mkdir(parents=True, exist_ok=True)
    path = png_dir / f"coarse_quicklook_t{t:05d}.png"
    fig.savefig(path, dpi=80)
    logger.info("saved %s", path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--coarse-root", type=Path, required=True)
    p.add_argument("--rechunk-root", type=Path, required=True)
    p.add_argument("--boundary-steps", default="")
    p.add_argument("--prognostic-steps", default="")
    p.add_argument("--png-dir", type=Path, default=None)
    args = p.parse_args()
    source = zarr.open_group(str(args.source), mode="r")
    coarse = {b: zarr.open_group(str(args.coarse_root / COARSE_NAMES[b]), mode="r") for b in BOXES}
    rechunk = zarr.open_group(str(args.rechunk_root / RECHUNK_NAME), mode="r")
    missing = {v: set(s) for v, s in json.loads(coarse[ONE].attrs["filled_time_steps_json"]).items()}
    check_layout(coarse, rechunk, coarse[QUARTER]["vars_3d"].shape[0])
    for t in (int(s) for s in args.boundary_steps.split(",") if s):
        check_boundary_step(source, coarse, rechunk, t, missing)
    for t in (int(s) for s in args.prognostic_steps.split(",") if s):
        check_prognostic_step(source, coarse, t, args.png_dir or args.coarse_root / "_check")
    logger.info("%d failure(s)", len(FAILURES))
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
