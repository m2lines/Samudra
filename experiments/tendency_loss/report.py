# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render figures and tables for the tendency-loss verification report.

Run from the repository root after run.py has trained the arms:
    PYTHONPATH=experiments/tendency_loss python experiments/tendency_loss/report.py \
        SKYLINE_PER_CHANNEL_CSV_GZ experiments/tendency_loss/report

SKYLINE_PER_CHANNEL_CSV_GZ is
experiments/temperature_skyline/all-fields-scaling/report/validation-per-channel.csv.gz
from the experiment/temperature-point-skyline branch.
"""

import csv
import gzip
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import run
import torch

from samudra.constants import build_om4_layout

DEPTHS = list(build_om4_layout().depth_levels)

HERE = Path(__file__).parent
WORK, RESULTS = run.WORK, run.RESULTS
FIELDS = ["uo", "vo", "thetao", "so", "zos"]
LABEL = {"uo": "u", "vo": "v", "thetao": "T", "so": "S", "zos": "SSH"}
# Validated categorical slots 1-5 (fields keep their colour in every figure).
COLOR = dict(zip(FIELDS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ARM_LABEL = {
    "equal_abs": "Equal MSE, absolute head",
    "dynamic_abs": "Dynamic (limit 20), absolute head\n= Samudra production",
    "tendency_abs": "Tendency (cap 1000), absolute head",
    "equal_res": "Equal MSE, residual head",
    "dynamic_res": "Dynamic (limit 20), residual head",
    "tendency20_res": "Tendency (cap 20), residual head",
    "tendency_res": "Tendency (cap 1000), residual head",
    "tendency_res_scaled": "Tendency (cap 1000), residual head\nscaled by change std = GraphCast",
    "dynamic_res_scaled": "Dynamic (limit 20), residual head\nscaled by change std",
    "tonly_res": "T only, residual head\n(no-mixing reference)",
}
ARM_ORDER = list(ARM_LABEL)
# Arms shown as lines in the depth and spectra figures (validated slots + ink).
KEY_ARMS = [
    "dynamic_abs",
    "dynamic_res",
    "tendency_res",
    "tendency_res_scaled",
    "tonly_res",
]
KEY_STYLE = {
    "dynamic_abs": ("#e34948", "s"),
    "dynamic_res": ("#2a78d6", "o"),
    "tendency_res": ("#eda100", "v"),
    "tendency_res_scaled": ("#1baf7a", "D"),
    "tonly_res": ("#52514e", "^"),
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    }
)


def field_of(name):
    return "zos" if name == "zos" else name.rsplit("_", 1)[0]


# -- Figure 1: loss shares on the quarter-degree skyline validation set ---------


def loss_shares(csv_path, out):
    rows = [
        r
        for r in csv.DictReader(gzip.open(csv_path, "rt"))
        if r["blocks"] == "16" and r["weights"] == "ema" and r["step"] == "25000"
    ]
    rows.sort(key=lambda r: int(r["channel_index"]))
    names = [r["channel"] for r in rows]
    L = np.array([float(r["full_grid_nmse"]) for r in rows])
    P = np.array([float(r["persistence_full_grid_nmse"]) for r in rows])
    field = np.array([field_of(n) for n in names])

    def dynamic(limit):
        w = 1 / L
        return np.clip(w, w.min(), w.min() * limit)

    def tendency(cap):
        w = 1 / P
        return np.clip(w, None, w.min() * cap) if cap else w

    schemes = {
        "Equal MSE (skyline)": np.ones_like(L),
        "Dynamic, limit 20\n(Samudra production)": dynamic(20),
        "Tendency, no cap": tendency(None),
        "Tendency, cap 1000\n(proposed)": tendency(1000),
    }
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, kind in zip(axes, ["loss", "gradient"]):
        for y, (label, w) in enumerate(schemes.items()):
            part = w * L if kind == "loss" else (w * np.sqrt(L)) ** 2
            share = {f: part[field == f].sum() / part.sum() for f in FIELDS}
            table.append(
                {
                    "scheme": label.replace("\n", " "),
                    "quantity": kind,
                    **{LABEL[f]: round(share[f], 4) for f in FIELDS},
                }
            )
            left = 0.0
            for f in FIELDS:
                ax.barh(
                    y,
                    share[f],
                    left=left,
                    color=COLOR[f],
                    height=0.62,
                    edgecolor=SURFACE,
                    linewidth=2,
                )
                if share[f] >= 0.06:
                    ax.text(
                        left + share[f] / 2,
                        y,
                        f"{LABEL[f]} {share[f]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=INK,
                    )
                left += share[f]
        ax.set_xlim(0, 1)
        ax.set_yticks(range(len(schemes)), list(schemes))
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(
            "Share of loss" if kind == "loss" else "Share of squared gradient norm",
            loc="left",
            fontsize=10,
        )
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[f]) for f in FIELDS]
    fig.legend(
        handles,
        [LABEL[f] for f in FIELDS],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Which fields drive training? Quarter-degree skyline validation errors "
        "re-weighted under each loss",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "loss-shares.png", dpi=120)
    plt.close(fig)
    write_csv(out / "loss-shares.csv", table)


# -- Figures 2-4: the 2-degree training comparison ------------------------------


def load_results():
    """Return {arm: [result per seed]}, seeds sorted."""
    res: dict[str, list[dict]] = {}
    for arm in ARM_ORDER:
        # Exact match: a prefix glob would also catch e.g. tendency_res_scaled_*.
        for p in sorted(RESULTS.glob("*_s*.json")):
            if p.stem.rsplit("_s", 1)[0] != arm:
                continue
            r = json.loads(p.read_text())
            r["field"] = np.array([field_of(n) for n in r["channels"]])
            for k in ("model_mse", "persistence_mse", "roughness_ratio"):
                r[k] = np.array(r[k])
            res.setdefault(arm, []).append(r)
    return res


def skill(r, f):
    m = r["field"] == f
    return (
        r["model_mse"][m].sum() / r["persistence_mse"][m].sum() if m.any() else np.nan
    )


def skill_table(res, out):
    rows = []
    for arm, runs in res.items():
        row = {"arm": ARM_LABEL[arm].replace("\n", " "), "seeds": len(runs)}
        for f in FIELDS:
            vals = [skill(r, f) for r in runs]
            if np.isfinite(vals).all():
                row[f"{LABEL[f]} skill mean"] = round(float(np.mean(vals)), 3)
                row[f"{LABEL[f]} skill per seed"] = " ".join(f"{v:.3f}" for v in vals)
                rough = [
                    float(np.mean(r["roughness_ratio"][r["field"] == f])) for r in runs
                ]
                row[f"{LABEL[f]} roughness"] = round(float(np.mean(rough)), 3)
        rows.append(row)
    write_csv(out / "skill-by-field.csv", rows)
    return rows


def skill_figure(res, out):
    arms = [a for a in ARM_ORDER if a in res]
    fig, axes = plt.subplots(1, 5, figsize=(14, 5.6), sharey=True)
    y = np.arange(len(arms))
    for ax, f in zip(axes, FIELDS):
        means = []
        for yi, a in zip(y, arms):
            vals = np.array([skill(r, f) for r in res[a]])
            mean = float(np.mean(vals)) if np.isfinite(vals).all() else np.nan
            means.append(mean)
            if np.isfinite(mean):
                ax.barh(yi, mean, color=COLOR[f], height=0.62)
                if len(vals) > 1:  # each seed as a dot
                    ax.plot(
                        vals,
                        [yi] * len(vals),
                        "o",
                        color=INK,
                        markersize=3.5,
                        markeredgecolor=SURFACE,
                        markeredgewidth=0.8,
                    )
                ax.text(
                    max(mean, vals.max()),
                    yi,
                    f"  {mean:.2f}",
                    va="center",
                    fontsize=8,
                    color=INK,
                )
            else:
                ax.text(
                    0.32, yi, " not predicted", va="center", fontsize=7.5, color=MUTED
                )
        ax.axvline(1.0, color=INK, linewidth=1, linestyle="--")
        ax.set_xscale("log")
        finite = [m for m in means if np.isfinite(m)]
        lo, hi = min(0.3, min(finite) * 0.7), max(finite) * 4
        ax.set_xlim(lo, hi)
        candidates = (0.5, 1, 2, 5) if hi <= 10 else (0.5, 1, 10, 100)
        ticks = [t for t in candidates if lo <= t <= hi]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(LABEL[f], loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [ARM_LABEL[a] for a in arms], fontsize=8)
    axes[0].invert_yaxis()
    fig.supxlabel(
        "Validation MSE ÷ persistence MSE (log scale; dashed = persistence; lower is better; "
        "dots = individual seeds)",
        fontsize=9,
    )
    fig.suptitle(
        "One-step skill relative to persistence, 2° OM4, Oct 2013 – Oct 2014 validation",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "skill-by-field.png", dpi=120)
    plt.close(fig)


def depth_figure(res, out):
    arms = [a for a in KEY_ARMS if a in res]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    depths = DEPTHS
    for a in arms:
        ratio = np.mean(
            [
                r["model_mse"][r["field"] == "thetao"]
                / r["persistence_mse"][r["field"] == "thetao"]
                for r in res[a]
            ],
            axis=0,
        )
        c, mk = KEY_STYLE[a]
        ax.plot(
            range(19),
            ratio,
            color=c,
            marker=mk,
            markersize=5,
            linewidth=2,
            label=ARM_LABEL[a].replace("\n", " "),
        )
    ax.axhline(1.0, color=INK, linewidth=1, linestyle="--")
    ax.text(18.3, 1.0, "persistence", va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_xticks(range(19), [f"{d:g}" for d in depths], rotation=60, fontsize=8)
    ax.set_xlabel("Depth level (m)")
    ax.set_ylabel("T validation MSE ÷ persistence MSE")
    ax.set_title("Temperature skill by depth", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    ax.set_ylim(top=max(ax.get_ylim()[1], 1e5))
    fig.tight_layout()
    fig.savefig(out / "temperature-skill-by-depth.png", dpi=120)
    plt.close(fig)


def predictions(arm, origin):
    """Reload a trained arm and predict one validation date."""
    loss_kind, head, outs = run.ARMS[arm]
    d = {
        k: np.load(WORK / f"{k}.npy", mmap_mode="r")
        for k in ("masks", "val_prog", "val_forc")
    }
    masks = torch.from_numpy(np.array(d["masks"]))
    out_idx = (
        list(range(77))
        if outs == "all"
        else [run.PROG.index(f"thetao_{z}") for z in range(19)]
    )
    net = run.Net(77 + 3 + 19, len(out_idx))
    net.load_state_dict(torch.load(WORK / f"{arm}_s0.pt"))
    net.eval()
    prog = torch.from_numpy(np.array(d["val_prog"][origin : origin + 1]))
    forc = torch.from_numpy(np.array(d["val_forc"][origin : origin + 1]))
    x = torch.cat([prog, forc, masks.float()[None]], 1)
    with torch.no_grad():
        y = net(x)
    if head == "res":
        y = prog[:, out_idx] + y
    elif head == "res_scaled":
        scale = np.load(WORK / "rms_norm.npy")[out_idx]
        y = prog[:, out_idx] + y * torch.from_numpy(scale).float().view(1, -1, 1, 1)
    names = [run.PROG[i] for i in out_idx]
    target = np.array(d["val_prog"][origin + 1])
    return names, y[0].numpy(), target, np.array(d["masks"])


def spectra_figure(res, out, origin=36):
    """Zonal power spectra of the T error along fully wet latitude rows
    (periodic in longitude, so no windowing), at three depths.
    """
    arms = [a for a in KEY_ARMS if a in res]
    styles = {a: KEY_STYLE[a][0] for a in arms}
    stds = np.load(WORK / "phys_stds.npy")
    levels = [0, 9, 12]  # 10, 8 and 7 fully wet rows at 2 degrees
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    band_rows = []
    for ax, z in zip(axes, levels):
        name = f"thetao_{z}"
        c = run.PROG.index(name)
        for a in arms:
            names, pred, target, masks = predictions(a, origin)
            k = names.index(name)
            rows = np.where(masks[z].all(axis=1))[0]
            err = (pred[k] - target[c])[rows] * stds[c]
            truth_change = (
                target[c]
                - np.array(np.load(WORK / "val_prog.npy", mmap_mode="r")[origin, c])
            )[rows] * stds[c]
            power = (np.abs(np.fft.rfft(err, axis=1)) ** 2).mean(0) / err.shape[1] ** 2
            ref = (np.abs(np.fft.rfft(truth_change, axis=1)) ** 2).mean(0) / err.shape[
                1
            ] ** 2
            kx = np.arange(power.size) / err.shape[1]
            ax.loglog(
                kx[1:],
                power[1:],
                color=styles[a],
                linewidth=2,
                linestyle=":" if a == "dynamic_abs" else "-",  # not colour alone
                label=ARM_LABEL[a].replace("\n", " "),
            )
            band = (kx >= 0.25) & (kx <= 0.5)
            band_rows.append(
                {
                    "depth_m": DEPTHS[z],
                    "arm": ARM_LABEL[a].replace("\n", " "),
                    "rows": len(rows),
                    "error_power_2to4cells": float(power[band].sum()),
                    "true_change_power_2to4cells": float(ref[band].sum()),
                }
            )
        ax.loglog(
            kx[1:],
            ref[1:],
            color=INK,
            linewidth=1,
            linestyle="--",
            label="True 5-day change",
        )
        ax.axvspan(0.25, 0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.set_title(f"{DEPTHS[z]:g} m ({len(rows)} rows)", loc="left", fontsize=10)
        ax.set_xlabel("Zonal wavenumber (cycles / grid cell)")
    axes[0].set_ylabel("Power (°C²)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Zonal spectra of one-step temperature error along fully wet latitude rows "
        "(shaded: 2–4-cell wavelengths)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "temperature-error-spectra.png", dpi=120)
    plt.close(fig)
    write_csv(out / "temperature-error-spectra.csv", band_rows)


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def skyline_floor_figure(csv_path, out):
    """Quarter-degree skyline (208M all-fields, EMA, 25k): temperature error vs
    the true 5-day change, by depth. A flat error under a falling change is the
    signature of an output noise floor, not of unpredictable physics.
    """
    rows = [
        r
        for r in csv.DictReader(gzip.open(csv_path, "rt"))
        if r["blocks"] == "16"
        and r["weights"] == "ema"
        and r["step"] == "25000"
        and r["channel"].startswith("thetao_")
    ]
    rows.sort(key=lambda r: int(r["depth"]))
    L = np.array([float(r["full_grid_nmse"]) for r in rows])
    P = np.array([float(r["persistence_full_grid_nmse"]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(19)
    ax.plot(
        x,
        L,
        color="#e34948",
        marker="s",
        markersize=5,
        linewidth=2,
        label="Skyline all-fields model error",
    )
    ax.plot(
        x,
        P,
        color=INK,
        marker="o",
        markersize=5,
        linewidth=2,
        linestyle="--",
        label="True 5-day change (persistence error)",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{d:g}" for d in DEPTHS], rotation=60, fontsize=8)
    ax.set_xlabel("Depth level (m)")
    ax.set_ylabel("Normalized MSE (full-grid)")
    ax.set_title(
        "Quarter-degree skyline: temperature error stays flat while the real "
        f"change spans {P.max() / P.min():,.0f}×",
        loc="left",
        fontsize=10.5,
    )
    ax.annotate(
        f"6,000 m: error {L[-1] / P[-1]:.0f}× persistence",
        (18, L[-1]),
        xytext=(12.5, 2e-6),
        fontsize=8.5,
        color=MUTED,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "skyline-temperature-error-floor.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    loss_shares(sys.argv[1], out)
    skyline_floor_figure(sys.argv[1], out)
    res = load_results()
    if res:
        skill_table(res, out)
        skill_figure(res, out)
        depth_figure(res, out)
        spectra_figure(res, out)
