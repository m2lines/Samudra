# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Maps, bias plots and spectra of one-step forecasts: ladder, Samudra 2, persistence.

    python onestep_figures.py extract CACHE PRED_DIR FIELDS.npz [--date 2016-07-21]
    python onestep_figures.py render FIELDS.npz OUTDIR

The figures follow the layout of the temperature-skyline report
(experiments/temperature_skyline/samudra-small-classifier on the
experiment/temperature-point-skyline branch), adapted to 1 degree.

Every forecast starts from the *true* state: the ladder's `lpc` rung (no
forcing) is stepped once from the true (t-1, t) pair, and Samudra 2's forecasts
come from one_step.py (PRED_DIR/test/pred.npy). Persistence predicts the state
at t. All three are scored at the target times Samudra 2 forecast (every other
five-day step of the test window, since it emits two steps per call), on the
same wet cells.

`extract` (heavy; run where the cache is) writes FIELDS.npz:
- regional patches on one target date, for the map figure;
- time-mean error (bias) maps for three depths, and the zonal-mean bias against
  depth and latitude, over all target times;
- radial spectra of field and error on fully wet tiles inside each region,
  averaged over all target times.
`render` (light) draws the PNGs and writes the plotted numbers as CSV.

Spectra follow the skyline report: remove a fitted plane, apply a periodic
Hann window, normalize power by the window's mean square, and sum into radial
shells of width 1/L cycles per cell (L = tile side), keeping nonzero
frequencies up to 0.5. A 1-degree cell is about 111 km times cos(latitude)
zonally, so the horizontal axis is in grid cells, not kilometres.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

VARS = {"thetao": "°C", "so": "psu", "uo": "m s⁻¹"}
DEPTH_LEVELS = (0, 7, 13)  # 2.5 m, 250 m, 1850 m, as in the skyline report
# (name, label, centre lat, centre lon in degrees east)
REGIONS = [
    ("north-pacific", "North Pacific", 35.0, 180.0),
    ("gulf-stream", "Gulf Stream", 38.0, 295.0),
    ("agulhas", "Agulhas", -40.0, 25.0),
]
PATCH = 32  # cells; 32 degrees at 1 degree
TILES = (32, 24, 16)  # fully wet spectral tiles, largest found first
MODELS = [  # key, label, colour, line style
    ("ladder", "Ladder (lpc)", "#d55e00", "-"),
    ("samudra2", "Samudra 2", "#0072b2", "-"),
    ("persistence", "Persistence", "#999900", ":"),
]


# ------------------------------------------------------------------ spectra
def radial_spectrum(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = a.shape[0]
    yy, xx = np.mgrid[:L, :L]
    G = np.column_stack([np.ones(L * L), yy.ravel(), xx.ravel()])
    coef, *_ = np.linalg.lstsq(G, a.ravel(), rcond=None)
    a = a - (G @ coef).reshape(L, L)
    h = np.hanning(L + 1)[:-1]
    w = np.outer(h, h)
    P = np.abs(np.fft.fft2(a * w)) ** 2 / (L * L) ** 2 / np.mean(w**2)
    f = np.fft.fftfreq(L)
    r = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    shell = np.rint(r * L).astype(int)
    k = np.arange(1, L // 2 + 1)
    return k / L, np.array([P[shell == s].sum() for s in k])


def wet_tile(wet: np.ndarray, y0: int, x0: int) -> tuple[int, int, int]:
    """Largest fully wet square tile inside the patch, nearest its centre."""
    for L in TILES:
        best = None
        for y in range(0, PATCH - L + 1):
            for x in range(0, PATCH - L + 1):
                win = wet[y0 + y : y0 + y + L, (x0 + x + np.arange(L)) % wet.shape[1]]
                if win.all():
                    d = abs(y + L / 2 - PATCH / 2) + abs(x + L / 2 - PATCH / 2)
                    if best is None or d < best[0]:
                        best = (d, y0 + y, x0 + x)
        if best:
            return best[1], best[2], L
    raise ValueError("no fully wet tile of 16 cells or more in this patch")


def take(a: np.ndarray, y: int, x: int, L: int) -> np.ndarray:
    return a[y : y + L, (x + np.arange(L)) % a.shape[1]]


# ------------------------------------------------------------------ extract
def extract(a: argparse.Namespace) -> None:
    import xarray as xr
    from rollout import Roller
    from run import Cache

    cache = Cache(a.cache)
    meta = json.loads((a.pred / "test" / "meta.json").read_text())
    pred = np.load(a.pred / "test" / "pred.npy", mmap_mode="r")
    s_ch = meta["channels"]
    stamps = cache.meta["splits"]["test"]
    targets = [t for t in meta["target_times"] if t in stamps]
    if len(targets) != len(meta["target_times"]):
        raise ValueError("some Samudra target times are not in the cache's test split")
    lon = xr.open_zarr(cache.meta["source"]).lon.values
    lat1 = cache.lat[:, 0]
    if not np.allclose(cache.lat, lat1[:, None]):
        raise ValueError("expects a rectilinear (Gaussian) grid")
    roller = Roller(cache, "lpc", forcing=False)
    print("[extract] ladder fitted", flush=True)
    H, W = cache.lat.shape
    nlev = 19
    sum_err = {(m, v): np.zeros((nlev, H, W)) for m, *_ in MODELS for v in VARS}
    sq_err = {(m, v): np.zeros(nlev) for m, *_ in MODELS for v in VARS}
    count = 0
    date_k = int(
        np.argmin([abs(np.datetime64(t[:10]) - np.datetime64(a.date)) for t in targets])
    )
    if a.max_targets:
        date_k = 0
    out: dict[str, np.ndarray] = {}
    patches = {}
    for name, _, clat, clon in REGIONS:
        y0 = int(np.argmin(abs(lat1 - clat))) - PATCH // 2
        x0 = int(np.argmin(abs((lon - clon + 180) % 360 - 180))) - PATCH // 2
        patches[name] = (y0, x0 % W)
    tiles = {}
    for name, (y0, x0) in patches.items():
        for z in DEPTH_LEVELS:
            tiles[name, z] = wet_tile(cache.masks[z], y0, x0)
    spec: dict[tuple, np.ndarray] = {}
    test = roller.test
    for k, t in enumerate(meta["target_times"]):
        if a.max_targets and k >= a.max_targets:
            break
        j = stamps.index(t)
        cur = {c: test[c][j - 1].astype(np.float64) for c in roller.chans}
        prev = {c: test[c][j - 2].astype(np.float64) for c in roller.chans}
        lad = roller.step(cur, prev, j - 1)
        for v in VARS:
            for z in range(nlev):
                c = f"{v}_{z}"
                wet = cache.masks[z]
                truth = test[c][j].astype(np.float64)
                fc = {
                    "ladder": lad[c],
                    "samudra2": pred[k, s_ch.index(c)].astype(np.float64),
                    "persistence": cur[c],
                }
                for m, f in fc.items():
                    e = np.where(wet, f - truth, 0.0)
                    sum_err[m, v][z] += e
                    sq_err[m, v][z] += np.sum(cache.area[wet] * e[wet] ** 2)
                    if z in DEPTH_LEVELS:
                        for name, (y0, x0) in patches.items():
                            ty, tx, L = tiles[name, z]
                            for kind, arr in (("error", f - truth), ("field", f)):
                                fr, p = radial_spectrum(take(arr, ty, tx, L))
                                key = (name, v, z, m, kind)
                                spec[key] = spec.get(key, 0) + p
                                spec[(name, v, z, "freq")] = fr
                            if k == date_k:
                                out[f"patch/{name}/{v}/{z}/{m}"] = take(
                                    f, y0, x0, PATCH
                                )
                if z in DEPTH_LEVELS:
                    for name, (y0, x0) in patches.items():
                        ty, tx, L = tiles[name, z]
                        fr, p = radial_spectrum(take(truth, ty, tx, L))
                        key = (name, v, z, "om4", "field")
                        spec[key] = spec.get(key, 0) + p
                        if k == date_k:
                            out[f"patch/{name}/{v}/{z}/om4"] = take(
                                truth, y0, x0, PATCH
                            )
                            out[f"patch/{name}/{v}/{z}/wet"] = take(wet, y0, x0, PATCH)
                            out[f"patch/{name}/{v}/{z}/area"] = take(
                                cache.area, y0, x0, PATCH
                            )
                            out[f"tile/{name}/{z}"] = np.array(
                                [ty - y0, (tx - x0) % W, L]
                            )
        count += 1
        if k % 25 == 0:
            print(f"[extract] {k + 1}/{len(meta['target_times'])} {t}", flush=True)
    area_tot = np.array([cache.area[cache.masks[z]].sum() for z in range(nlev)])
    for (m, v), s in sum_err.items():
        bias = s / count
        bias = np.where(cache.masks[:nlev], bias, np.nan)
        for z in DEPTH_LEVELS:
            out[f"bias/{v}/{z}/{m}"] = bias[z].astype(np.float32)
        # zonal-mean bias, area-weighted over wet cells in each latitude row
        wts = np.where(cache.masks[:nlev], cache.area[None], 0.0)
        num = np.nansum(np.nan_to_num(bias) * wts, axis=2)
        den = wts.sum(axis=2)
        out[f"zonal/{v}/{m}"] = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        out[f"rmse/{v}/{m}"] = np.sqrt(sq_err[m, v] / count / area_tot)
    for key, p in spec.items():
        if key[-1] == "freq":
            out["spec/" + "/".join(map(str, key))] = p
        else:
            out["spec/" + "/".join(map(str, key))] = p / count
    out["lat"] = lat1
    out["lon"] = lon
    out["depth"] = np.array([cache.meta["depth"][f"thetao_{z}"] for z in range(nlev)])
    out["date"] = np.array(meta["target_times"][date_k])
    out["n_targets"] = np.array(count)
    out["patch_origin"] = np.array([patches[n] for n, *_ in REGIONS])
    a.fields.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.fields, **out)
    print(f"[extract] wrote {a.fields}", flush=True)


# ------------------------------------------------------------------ render
def render(a: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d = np.load(a.fields)
    o = a.out
    o.mkdir(parents=True, exist_ok=True)
    depth = d["depth"]
    date = str(d["date"])[:10]
    n = int(d["n_targets"])
    rows = []

    # Regional maps, one figure per region and variable.
    for v, unit in VARS.items():
        for name, label, *_ in REGIONS:
            fig = plt.figure(figsize=(16, 10.6), dpi=100)
            for row, z in enumerate(DEPTH_LEVELS):
                p = lambda m: d[f"patch/{name}/{v}/{z}/{m}"]  # noqa: E731
                wet = p("wet").astype(bool)
                area = p("area")
                gold, lad, sam = p("om4"), p("ladder"), p("samudra2")
                fields = [gold, lad, sam]
                lo = min(float(np.nanmin(f[wet])) for f in fields)
                hi = max(float(np.nanmax(f[wet])) for f in fields)
                errs = [lad - gold, sam - gold]
                lim = max(float(np.nanmax(abs(e[wet]))) for e in errs)
                ty, tx, L = d[f"tile/{name}/{z}"]
                for col, vals in enumerate(fields + errs):
                    ax = fig.add_axes(
                        (
                            (45 + col * 300) / 1600,
                            (710 - row * 330) / 1060,
                            256 / 1600,
                            256 / 1060,
                        )
                    )
                    im = ax.imshow(
                        np.ma.masked_where(~wet, vals),
                        origin="lower",
                        interpolation="nearest",
                        cmap="viridis" if col < 3 else "RdBu_r",
                        vmin=lo if col < 3 else -lim,
                        vmax=hi if col < 3 else lim,
                    )
                    ax.set_facecolor("#d9d9d9")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row == 0:
                        ax.set_title(
                            [
                                "OM4",
                                "Ladder (lpc)",
                                "Samudra 2",
                                "Ladder − OM4",
                                "Samudra 2 − OM4",
                            ][col],
                            fontsize=11,
                        )
                    if col == 0:
                        ax.set_ylabel(f"{depth[z]:g} m", fontsize=12)
                    if col >= 3:
                        ax.add_patch(
                            Rectangle(
                                (tx - 0.5, ty - 0.5), L, L, fill=False, ls="--", lw=0.8
                            )
                        )
                        rmse = np.sqrt(np.average(vals[wet] ** 2, weights=area[wet]))
                        ax.set_xlabel(f"Area RMSE {rmse:.4g} {unit}", fontsize=10)
                        rows.append(
                            {
                                "figure": "map",
                                "region": name,
                                "variable": v,
                                "depth_m": depth[z],
                                "model": ["ladder", "samudra2"][col - 3],
                                "date": date,
                                "rmse": rmse,
                            }
                        )
                    if col in (2, 4):
                        cax = fig.add_axes(
                            (
                                (45 + col * 300 + 264) / 1600,
                                (710 - row * 330) / 1060,
                                9 / 1600,
                                256 / 1060,
                            )
                        )
                        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=8)
            fig.suptitle(
                f"{label} {v} · one step (+5 days) from the true state · target {date} · 1° official OM4\n"
                f"{PATCH}×{PATCH} cells · grey = land · dashed box = spectral tile · all scales in {unit}",
                y=0.985,
                fontsize=14,
            )
            fig.savefig(o / f"maps-{name}-{v}.png", dpi=90)
            plt.close(fig)

    # Global time-mean bias maps: rows = depth, columns = ladder, Samudra 2.
    lat, lon = d["lat"], d["lon"]
    for v, unit in VARS.items():
        fig, axes = plt.subplots(3, 2, figsize=(14, 10.5), layout="constrained")
        for r, z in enumerate(DEPTH_LEVELS):
            bl, bs = d[f"bias/{v}/{z}/ladder"], d[f"bias/{v}/{z}/samudra2"]
            lim = float(
                np.nanpercentile(np.abs(np.concatenate([bl.ravel(), bs.ravel()])), 99)
            )
            for c, (m, b) in enumerate((("Ladder (lpc)", bl), ("Samudra 2", bs))):
                ax = axes[r, c]
                im = ax.pcolormesh(
                    lon, lat, b, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest"
                )
                ax.set_facecolor("#d9d9d9")
                ax.set_title(
                    f"{m} · {depth[z]:g} m · global-mean |bias| {np.nanmean(np.abs(b)):.3g}",
                    fontsize=10,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            fig.colorbar(im, ax=axes[r, :], shrink=0.9, label=f"{unit} (±99th pct.)")
        fig.suptitle(
            f"{v} one-step bias: time-mean (forecast − OM4) over {n} target dates, 2014-10 to 2022-12\n"
            "Each forecast starts from the true state, so this is the per-step drift a rollout would accumulate",
            fontsize=13,
        )
        fig.savefig(o / f"bias-maps-{v}.png", dpi=80)
        plt.close(fig)

    # Zonal-mean bias against latitude and depth.
    fig, axes = plt.subplots(
        len(VARS), 2, figsize=(13, 11), layout="constrained", sharey=True
    )
    for r, (v, unit) in enumerate(VARS.items()):
        zl, zs = d[f"zonal/{v}/ladder"], d[f"zonal/{v}/samudra2"]
        lim = float(
            np.nanpercentile(np.abs(np.concatenate([zl.ravel(), zs.ravel()])), 99)
        )
        for c, (m, b) in enumerate((("Ladder (lpc)", zl), ("Samudra 2", zs))):
            ax = axes[r, c]
            im = ax.pcolormesh(
                lat,
                np.arange(len(depth)),
                b,
                cmap="RdBu_r",
                vmin=-lim,
                vmax=lim,
                shading="nearest",
            )
            ax.set_facecolor("#d9d9d9")
            ax.set_title(f"{m} · {v}", fontsize=11)
            ax.set_yticks(range(0, len(depth), 3), [f"{x:g}" for x in depth[::3]])
            if c == 0:
                ax.set_ylabel("depth (m)")
            if r == len(VARS) - 1:
                ax.set_xlabel("latitude")
        fig.colorbar(im, ax=axes[r, :], shrink=0.9, label=unit)
    axes[0, 0].invert_yaxis()
    fig.suptitle(
        f"Zonal-mean one-step bias by depth (levels evenly spaced) · {n} target dates · area-weighted over wet cells",
        fontsize=13,
    )
    fig.savefig(o / "bias-zonal.png", dpi=90)
    plt.close(fig)

    # RMSE by depth: ladder, Samudra 2, persistence.
    fig, axes = plt.subplots(1, len(VARS), figsize=(14, 5), layout="constrained")
    for ax, (v, unit) in zip(axes, VARS.items(), strict=True):
        for m, label, color, ls in MODELS:
            r_ = d[f"rmse/{v}/{m}"]
            ax.semilogx(r_, depth, color=color, ls=ls, label=label, lw=1.8)
            for z, val in enumerate(r_):
                rows.append(
                    {
                        "figure": "rmse_by_depth",
                        "region": "global",
                        "variable": v,
                        "depth_m": depth[z],
                        "model": m,
                        "date": "all",
                        "rmse": float(val),
                    }
                )
        ax.set_yscale("symlog", linthresh=10)
        ax.invert_yaxis()
        ax.set_title(v)
        ax.set_xlabel(f"area-weighted RMSE ({unit})")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("depth (m)")
    axes[0].legend()
    fig.suptitle(
        f"One-step RMSE by depth, global wet cells, {n} target dates", fontsize=13
    )
    fig.savefig(o / "rmse-by-depth.png", dpi=90)
    plt.close(fig)

    # Spectra: one figure per variable and kind; rows = region, cols = depth.
    spec_rows = []
    for v, unit in VARS.items():
        for kind in ("field", "error"):
            fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
            for r, (name, label, *_) in enumerate(REGIONS):
                for c, z in enumerate(DEPTH_LEVELS):
                    ax = axes[r, c]
                    f = d[f"spec/{name}/{v}/{z}/freq"]
                    series = list(MODELS)
                    if kind == "field":
                        series = [("om4", "OM4", "black", "-")] + series
                    for m, lab, color, ls in series:
                        p = d[f"spec/{name}/{v}/{z}/{m}/{kind}"]
                        ax.loglog(f, p, color=color, ls=ls, lw=1.7, label=lab)
                        for fi, pi in zip(f, p, strict=True):
                            spec_rows.append(
                                {
                                    "variable": v,
                                    "kind": kind,
                                    "region": name,
                                    "depth_m": depth[z],
                                    "model": m,
                                    "cycles_per_cell": fi,
                                    "power": pi,
                                }
                            )
                    L = int(d[f"tile/{name}/{z}"][2])
                    ax.set_title(f"{label} · {depth[z]:g} m · {L}×{L} tile")
                    ax.axvspan(0.25, 0.5, alpha=0.1, color="gray")
                    ax.grid(alpha=0.2)
                    if r == 2:
                        ax.set_xlabel("Cycles / 1° grid cell")
                    if c == 0:
                        ax.set_ylabel(f"Power per shell ({unit}²)")
            fig.suptitle(
                f"{v} {kind} spectra · one step from the true state · mean over {n} target dates\n"
                "Shaded wavelengths 2–4 cells · fully wet tiles · plane removed + periodic Hann",
                fontsize=14,
                y=0.99,
            )
            fig.legend(
                *axes[0, 0].get_legend_handles_labels(),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=4,
                frameon=False,
            )
            fig.subplots_adjust(
                top=0.86, bottom=0.07, hspace=0.3, wspace=0.28, left=0.08, right=0.98
            )
            fig.savefig(o / f"spectra-{kind}-{v}.png", dpi=90)
            plt.close(fig)

    for fname, data in (("figure-metrics.csv", rows), ("spectra.csv", spec_rows)):
        with (o / fname).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("cache", type=Path)
    e.add_argument("pred", type=Path)
    e.add_argument("fields", type=Path)
    e.add_argument("--date", default="2016-07-21")
    e.add_argument("--max-targets", type=int, default=0, help="smoke test only")
    r = sub.add_parser("render")
    r.add_argument("fields", type=Path)
    r.add_argument("out", type=Path)
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else render(a)


if __name__ == "__main__":
    main()
