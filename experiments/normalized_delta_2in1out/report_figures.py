# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Figures for the normalized-delta results reports (#984 and its stacked runs).

    python report_figures.py CONFIG.json OUT_DIR [FIGURE,...]

CONFIG lists the truth store, the one-step scores from score_onestep.py, and
the runs to compare. Each run has a `key` (its name in the scores file), a
`label`, a `color`, and any of: `eval` (an eval output dir holding
predictions.zarr and observation_metrics.csv), `onestep` (a one_step.py output
dir) and `logs` (training experiment.log files, in order, for runs that were
resumed). Every figure skips runs that lack its inputs.

Conventions: depth is drawn as level index with depth labels, since the 19
levels are spaced unevenly (2.5 m to 6000 m). Spatial means are weighted by
cell area over wet cells. "Year 1" and "last year" are the first and last 73
five-day steps of the rollout.
"""

import json
import re
import sys
from pathlib import Path

import cartopy.crs as ccrs  # type: ignore[import-untyped]
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#8a8984"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#9c2525"]
)
mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)
YEAR = 73


def depth_axis(ax, depths, label=True):
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{d:g}" for d in depths])
    ax.set_ylim(len(depths) - 0.5, -0.5)
    if label:
        ax.set_ylabel("depth (m)")


def validation(cfg, out):
    """Best validation score so far vs epoch (only improvements are logged)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, run in enumerate(cfg["runs"]):
        points, last = [], 0
        for log in run.get("logs", []):
            epoch = None
            for line in open(log, errors="replace"):
                if m := re.search(r"Training Epoch: \[(\d+)\]", line):
                    epoch = int(m.group(1))
                    last = max(last, epoch)
                elif (
                    m := re.search(r"checkpoint score \(([\d.]+)\)", line)
                ) and epoch is not None:
                    points.append((epoch, float(m.group(1))))
        if not points:
            continue
        e, s = zip(*sorted(dict(points).items()), strict=True)
        e, s = (*e, last), (*s, s[-1])
        ax.step(e, s, where="post", color=run["color"], label=run["label"])
        ax.annotate(
            f"{s[-1]:.3f}",
            (e[-1], s[-1]),
            xytext=(4, 5 - 10 * (i % 2)),
            textcoords="offset points",
            color=INK,
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("best validation rollout score")
    ax.set_ylim(0.24, 0.45)
    ax.set_title("Validation score (lower is better)")
    ax.legend()
    fig.savefig(out / "validation.png")
    plt.close(fig)


def onestep_by_depth(cfg, out):
    scores = json.loads(Path(cfg["onestep_scores"]).read_text())
    ch, depths = scores["channels"], scores["depths_m"]
    runs = [r for r in cfg["runs"] if r["key"] in ch["thetao_0"]]
    names = {
        "thetao": "temperature",
        "so": "salinity",
        "uo": "zonal velocity",
        "vo": "meridional velocity",
    }
    for metric, fname, xlabel in [
        ("ratio", "onestep_rmse_by_depth.png", "RMSE / persistence RMSE"),
        ("worse", "onestep_area_worse.png", "% of area worse than persistence"),
    ]:
        variables = list(names) if metric == "ratio" else ["thetao", "so"]
        fig, axes = plt.subplots(
            1, len(variables), figsize=(2.6 * len(variables), 4.2), sharey=True
        )
        for i, (ax, v) in enumerate(zip(axes, variables, strict=True)):
            for run in runs:
                c = [ch[f"{v}_{k}"] for k in range(len(depths))]
                if metric == "ratio":
                    x = [r[run["key"]]["rmse"] / r["persistence_rmse"] for r in c]
                else:
                    x = [100 * r[run["key"]]["area_worse_than_persistence"] for r in c]
                ax.plot(
                    x,
                    range(len(depths)),
                    "-o",
                    ms=3.5,
                    color=run["color"],
                    label=run["label"],
                )
            if metric == "ratio":
                ax.axvline(1, color=NEUTRAL, lw=1.2, ls="--")
            ax.set_title(names[v])
            ax.set_xlabel(xlabel)
            depth_axis(ax, depths, label=i == 0)
        axes[0].legend(loc="lower left", fontsize=7)
        fig.suptitle(
            f"One-step forecasts from the true state, {scores['n_dates']} test"
            " dates (left of 1 beats persistence)"
            if metric == "ratio"
            else f"One-step forecasts: where each model loses to persistence",
            fontsize=10,
        )
        fig.savefig(out / fname)
        plt.close(fig)


def onestep_maps(cfg, out, truth):
    """Per-cell RMSE ratio to persistence for channels where the runs differ."""
    runs = [r for r in cfg["runs"] if r.get("onestep")]
    if not runs:
        return
    channels = cfg.get("map_channels", ["so_13", "thetao_18"])
    metas = [json.loads((Path(r["onestep"]) / "meta.json").read_text()) for r in runs]
    preds = [np.load(Path(r["onestep"]) / "pred.npy", mmap_mode="r") for r in runs]
    common = sorted(set.intersection(*(set(m["target_times"]) for m in metas)))
    pos = {str(t): i for i, t in enumerate(truth.time.values)}
    target = np.array([pos[t] for t in common])
    lon, lat = truth.lon.values, truth.lat.values
    fig, axes = plt.subplots(
        len(channels),
        len(runs),
        figsize=(4.6 * len(runs), 2.6 * len(channels)),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=200)},
        squeeze=False,
    )
    for i, c in enumerate(channels):
        t = truth[c].isel(time=target).values.astype("f8")
        p = truth[c].isel(time=target - 1).values.astype("f8")
        mse_p = np.mean((p - t) ** 2, axis=0)
        for j, (run, m, pr) in enumerate(zip(runs, metas, preds, strict=True)):
            idx = [m["target_times"].index(x) for x in common]
            f = np.asarray(pr[idx, m["channels"].index(c)], dtype="f8")
            ratio = np.log2(np.sqrt(np.mean((f - t) ** 2, axis=0) / mse_p))
            ax = axes[i, j]
            im = ax.pcolormesh(
                lon,
                lat,
                ratio,
                cmap=DIVERGING,
                vmin=-1.5,
                vmax=1.5,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
            ax.coastlines(lw=0.3, color=MUTED)
            v, k = c.rsplit("_", 1)
            name = {"thetao": "temperature", "so": "salinity"}.get(v, v)
            depth = truth.lev.values[int(k)] if "lev" in truth else k
            ax.set_title(f"{run['label']}: {name}, {depth:g} m", fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    cb.set_ticklabels(["0.35×", "0.5×", "0.71×", "1×", "1.4×", "2×", "2.8×"])
    cb.set_label("RMSE / persistence RMSE (blue beats persistence)")
    fig.savefig(out / "onestep_maps.png")
    plt.close(fig)


def rollout(cfg, out, truth):
    runs = [r for r in cfg["runs"] if r.get("eval")]
    if not runs:
        return
    series, maps, variance = {}, {}, {}
    for run in runs:
        z = xr.open_zarr(Path(run["eval"]) / "predictions.zarr")
        area = z.areacello.values
        times = z.time
        depths = z.lev.values
        sst = z.thetao.isel(lev=0).values.astype("f8")
        sss = z.so.isel(lev=0).values.astype("f8")
        t_sst = truth.thetao_0.sel(time=times).values.astype("f8")
        t_sss = truth.so_0.sel(time=times).values.astype("f8")
        w = np.where(np.isfinite(t_sst[0]), area, 0)

        def gmean(a, w=w):
            return np.nansum(a * w, axis=(1, 2)) / w.sum()

        wet = w > 0
        series[run["key"]] = {
            "sst_bias": gmean(sst) - gmean(t_sst),
            "sss_bias": gmean(sss) - gmean(t_sss),
            "sst_max": np.nanmax(np.where(wet, sst, np.nan), axis=(1, 2)),
            "truth_max": np.nanmax(np.where(wet, t_sst, np.nan), axis=(1, 2)),
        }
        hot = np.unravel_index(np.nanargmax(np.where(wet, sst, np.nan)), sst.shape)
        maps[run["key"]] = (
            np.where(wet, sst[-YEAR:].mean(0) - t_sst[-YEAR:].mean(0), np.nan),
            (z.lon.values[hot[1], hot[2]], z.lat.values[hot[1], hot[2]]),
        )
        rows = {}
        for v in ("thetao", "so", "uo"):
            raw, detr = [], []
            # Chunks hold every level, so read each variable once, not per level.
            full = z[v].values
            for k in range(len(depths)):
                x = full[:, k].astype("f8")
                t = truth[f"{v}_{k}"].sel(time=times).values.astype("f8")
                m = np.isfinite(t).all(0) & np.isfinite(x).all(0)
                wk = np.where(m, area, 0)
                raw.append(
                    np.sqrt(
                        (np.where(m, x.var(0), 0) * wk).sum()
                        / (np.where(m, t.var(0), 0) * wk).sum()
                    )
                )
                s = np.arange(len(times), dtype="f8")
                s -= s.mean()

                def detrend(a, s=s):
                    a = np.where(m, a, 0.0)
                    a = a - a.mean(0)
                    slope = np.tensordot(s, a, axes=(0, 0)) / (s @ s)
                    return a - s[:, None, None] * slope

                detr.append(
                    np.sqrt(
                        (detrend(x).var(0) * wk).sum() / (detrend(t).var(0) * wk).sum()
                    )
                )
            rows[v] = (raw, detr)
        variance[run["key"]] = rows
        print("rollout stats done for", run["key"], flush=True)
    lon, lat = z.lon.values, z.lat.values
    years = 2014.8 + np.arange(len(times)) / YEAR

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), layout="constrained")
    for run in runs:
        ser = series[run["key"]]
        axes[0].plot(
            years, ser["sst_bias"], color=run["color"], lw=1.4, label=run["label"]
        )
        axes[1].plot(years, ser["sss_bias"], color=run["color"], lw=1.4)
        axes[2].plot(years, ser["sst_max"], color=run["color"], lw=1.4)
        i = int(np.nanargmax(ser["sst_max"]))
        axes[2].annotate(
            f"{ser['sst_max'][i]:.1f}",
            (years[i], ser["sst_max"][i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=INK,
        )
    axes[2].plot(
        years, series[runs[0]["key"]]["truth_max"], color=NEUTRAL, lw=1.2, label="OM4"
    )
    for ax, title, unit in [
        (axes[0], "Global mean SST bias", "°C"),
        (axes[1], "Global mean surface salinity bias", "psu"),
        (axes[2], "Hottest SST anywhere", "°C"),
    ]:
        ax.set_title(title)
        ax.set_ylabel(unit)
        if ax is not axes[2]:
            ax.axhline(0, color=NEUTRAL, lw=1, ls="--")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.suptitle("Eight-year rollout from 2014-10, against OM4", fontsize=10)
    fig.savefig(out / "rollout_drift.png")
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(runs),
        figsize=(4.6 * len(runs), 2.8),
        squeeze=False,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=200)},
    )
    for ax, run in zip(axes[0], runs, strict=True):
        bias, (hx, hy) = maps[run["key"]]
        im = ax.pcolormesh(
            lon,
            lat,
            bias,
            cmap=DIVERGING,
            vmin=-3,
            vmax=3,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
        ax.coastlines(lw=0.3, color=MUTED)
        ax.plot(
            hx,
            hy,
            marker="o",
            ms=9,
            mfc="none",
            mec=INK,
            mew=1.5,
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(f"{run['label']}", fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02, extend="both")
    cb.set_label("°C")
    fig.suptitle(
        "SST bias in the rollout's last year (circle: hottest cell of the"
        " whole rollout)",
        fontsize=10,
    )
    fig.savefig(out / "sst_bias_last_year.png")
    plt.close(fig)

    names = {"thetao": "temperature", "so": "salinity", "uo": "zonal velocity"}
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.2), sharey=True)
    for i, (ax, v) in enumerate(zip(axes, names, strict=True)):
        for run in runs:
            raw, detr = variance[run["key"]][v]
            ax.plot(
                raw,
                range(len(depths)),
                "-o",
                ms=3,
                color=run["color"],
                label=f"{run['label']}, raw",
            )
            ax.plot(
                detr,
                range(len(depths)),
                "--",
                color=run["color"],
                lw=1.4,
                label=f"{run['label']}, linear trend removed",
            )
        ax.axvline(1, color=NEUTRAL, lw=1.2)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.set_title(names[v])
        ax.set_xlabel("temporal std / OM4's")
        depth_axis(ax, depths, label=i == 0)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        "Variability over the rollout, relative to OM4 (1 = matches)", fontsize=10
    )
    fig.savefig(out / "variance_by_depth.png")
    plt.close(fig)
    (out / "rollout_stats.json").write_text(
        json.dumps(
            {
                k: {
                    v: {"raw": list(map(float, r)), "detrended": list(map(float, d))}
                    for v, (r, d) in rows.items()
                }
                for k, rows in variance.items()
            },
            indent=1,
        )
    )


def observations(cfg, out):
    runs = [r for r in cfg["runs"] if r.get("eval")]
    if not runs:
        return
    labels = {
        ("surface_sst_total_rmse", ""): "SST vs OISST",
        (
            "surface_geostrophic_velocity_vector_total_rmse",
            "",
        ): "surface geostrophic\nvelocity vs DUACS",
        ("instantaneous_surface_eke_total_rmse", ""): "surface EKE vs DUACS",
        ("ohc_per_area_total_rmse", "Surface (0-700m)"): "OHC 0–700 m vs IAP",
        (
            "ohc_per_area_total_rmse",
            "Intermediate (700-2000m)",
        ): "OHC 700–2000 m vs IAP",
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    offsets = np.linspace(-0.15, 0.15, len(runs))
    for off, run in zip(offsets, runs, strict=True):
        df = pd.read_csv(Path(run["eval"]) / "observation_metrics.csv")
        df = df[df.period_kind == "primary_complete_years"].fillna({"depth": ""})
        om4 = df[df.model == "om4"].set_index(["metric", "depth"])["value"]
        mod = df[df.model != "om4"].set_index(["metric", "depth"])
        cols = {
            c: mod[c].astype(float).to_dict() for c in ("value", "ci_low", "ci_high")
        }
        for i, key in enumerate(labels):
            base = om4.astype(float).to_dict()[key]
            value, low, high = (cols[c][key] for c in ("value", "ci_low", "ci_high"))
            ax.errorbar(
                value / base,
                i + off,
                xerr=[[(value - low) / base], [(high - value) / base]],
                fmt="o",
                ms=6,
                color=run["color"],
                capsize=2,
                label=run["label"] if i == 0 else None,
            )
    ax.axvline(1, color=NEUTRAL, lw=1.2, ls="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels.values())
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("model's RMSE against observations / OM4's own RMSE")
    ax.set_title("Rollout skill against observations, 2015–2022 (1 = as good as OM4)")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(out / "observation_metrics.png")
    plt.close(fig)


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    truth = xr.open_zarr(cfg["truth"])
    figures = {
        "validation": lambda: validation(cfg, out),
        "onestep": lambda: onestep_by_depth(cfg, out),
        "observations": lambda: observations(cfg, out),
        "maps": lambda: onestep_maps(cfg, out, truth),
        "rollout": lambda: rollout(cfg, out, truth),
    }
    for name, make in figures.items():
        if only is None or name in only:
            make()
            print(name, "done", flush=True)
    print("all figures in", out, flush=True)


if __name__ == "__main__":
    main()
