# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Aggregate multi-date initializer diagnostics and plot selected velocity fields."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    grid = np.load(a.input / "grid.npz")
    names = grid["names"]
    lat = grid["lat"]
    lon = grid["lon"]
    mask = grid["mask"]
    sel = grid["selection"]
    rows = []
    for path in sorted(a.input.glob("*.json")):
        r = json.loads(path.read_text())
        if "metrics" not in r:
            continue
        for var, ids in [
            ("uo", range(19)),
            ("vo", range(19, 38)),
            ("thetao", range(38, 57)),
            ("so", range(57, 76)),
            ("zos", [76]),
        ]:
            m = {k: np.asarray(v)[list(ids)].mean() for k, v in r["metrics"].items()}
            row = dict(
                variant=r["variant"],
                step=r["step"],
                month=r["month"],
                variable=var,
                **m,
            )
            row["mean_rmse"] = np.sqrt(m["mean_mse"])
            row["member_rmse"] = np.sqrt(m["member_mse"])
            rows.append(row)
    with (a.output / "by-date.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for variant in sorted({r["variant"] for r in rows}):
        for step in sorted({r["step"] for r in rows if r["variant"] == variant}):
            for var in ["uo", "vo", "thetao", "so", "zos"]:
                selected = [
                    r
                    for r in rows
                    if r["variant"] == variant
                    and r["step"] == step
                    and r["variable"] == var
                ]
                agg = dict(
                    variant=variant, step=step, variable=var, dates=len(selected)
                )
                for key in selected[0]:
                    if key not in ["variant", "step", "month", "variable"]:
                        agg[key] = float(np.mean([r[key] for r in selected]))
                agg["mean_rmse"] = float(np.sqrt(agg["mean_mse"]))
                agg["member_rmse"] = float(np.sqrt(agg["member_mse"]))
                aggregate.append(agg)
    with (a.output / "aggregate.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    variants = [
        "A-control",
        "A-mae",
        "B-control",
        "B-member-mse",
        "B-crps-mean-mse",
        "B-multiscale-fair-crps",
        "B-initializer-only",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for vi, var in enumerate(["uo", "vo"]):
        for ai, key in enumerate(["mean_rmse", "absolute", "spread"]):
            for variant in variants:
                rr = [
                    r
                    for r in aggregate
                    if r["variant"] == variant and r["variable"] == var
                ]
                axes[vi, ai].plot(
                    [r["step"] for r in rr],
                    [r[key] for r in rr],
                    marker="o",
                    label=variant,
                )
            axes[vi, ai].set(
                title=f"{var}: {key}", xlabel="Mixed optimizer updates", ylabel="m/s"
            )
            axes[vi, ai].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(a.output / "learning-curves.png", dpi=150)
    plt.close(fig)
    rankrows = []
    spectra = []
    for month in sorted({r["month"] for r in rows}):
        for si, name in [(0, "uo_0"), (3, "vo_0")]:
            fig, axes = plt.subplots(
                7,
                4,
                figsize=(18, 15),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for ri, variant in enumerate(variants):
                z = np.load(a.input / f"{variant}-02000-{month}-maps.npz")
                x = z["members"][:, si]
                y = z["gold"][si]
                wet = mask[sel[si]]
                for ci, (title, field) in enumerate(
                    [
                        ("Gold", y),
                        ("Mean of 8", x.mean(0)),
                        ("Member 1", x[0]),
                        ("Mean minus gold", x.mean(0) - y),
                    ]
                ):
                    axes[ri, ci].pcolormesh(
                        lon,
                        lat,
                        np.where(wet, field, np.nan),
                        cmap="RdBu_r",
                        vmin=-0.5,
                        vmax=0.5,
                        shading="nearest",
                        rasterized=True,
                    )
                    if ri == 0:
                        axes[ri, ci].set_title(title)
                    if ci == 0:
                        axes[ri, ci].set_ylabel(variant, fontsize=8)
                axes[ri, 0].set_ylim(-65, 65)
            fig.suptitle(
                f"{month} OM4 initialization {name}; common ±0.5 m/s color scale"
            )
            fig.savefig(a.output / f"maps-{month}-{name}.png", dpi=140)
            plt.close(fig)
        for variant in variants:
            z = np.load(a.input / f"{variant}-02000-{month}-maps.npz")
            for si, c in enumerate(sel):
                x = z["members"][:, si]
                y = z["gold"][si]
                wet = mask[c]
                w = wet * np.cos(np.deg2rad(lat))[:, None]
                lower = (x < y).sum(0)
                ties = (x == y).sum(0)
                hist = np.array(
                    [
                        np.sum(w * ((lower <= i) & (i <= lower + ties)) / (ties + 1))
                        / w.sum()
                        for i in range(9)
                    ]
                )
                for rank, value in enumerate(hist):
                    rankrows.append(
                        dict(
                            variant=variant,
                            month=month,
                            channel=str(names[c]),
                            rank=rank,
                            area_fraction=value,
                        )
                    )
                # Common fixed Pacific window; FFT uses identical wet mask and Hann taper.
                iy = np.where((lat >= 20) & (lat <= 40))[0]
                ix = np.where((lon >= 150) & (lon <= 190))[0]
                mm = wet[np.ix_(iy, ix)]
                window = np.outer(np.hanning(len(iy)), np.hanning(len(ix)))
                stack = np.concatenate([y[None], x.mean(0)[None], x], axis=0)[
                    :, iy[:, None], ix
                ]
                stack = (
                    (stack - (stack * mm).sum((-2, -1), keepdims=True) / mm.sum())
                    * mm
                    * window
                )
                power = np.abs(np.fft.rfft2(stack)) ** 2 / (
                    window.square().sum()
                    if hasattr(window, "square")
                    else (window**2).sum()
                )
                ky = np.fft.fftfreq(len(iy))[:, None]
                kx = np.fft.rfftfreq(len(ix))[None, :]
                k = np.sqrt(ky**2 + kx**2)
                for lo, hi in zip(
                    np.linspace(0, 0.71, 15)[:-1], np.linspace(0, 0.71, 15)[1:]
                ):
                    keep = (k >= lo) & (k < hi)
                    if not keep.any():
                        continue
                    vals = power[:, keep].mean(1)
                    spectra.append(
                        dict(
                            variant=variant,
                            month=month,
                            channel=str(names[c]),
                            k_low=lo,
                            k_high=hi,
                            wet_fraction=mm.mean(),
                            gold=vals[0],
                            mean8=vals[1],
                            member1=vals[2],
                            mean_member_power=vals[2:].mean(),
                        )
                    )
    for fname, data in [
        ("rank-histograms.csv", rankrows),
        ("pacific-spectra.csv", spectra),
    ]:
        with (a.output / fname).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (a.output / "methods.txt").write_text(
        "Equal-date, equal-depth averages with cosine-latitude wet-cell weights. Rank cells are spatially dependent; not independent calibration samples. Tie ranks are uniformly divided. Pacific spectra: 150–190E,20–40N, physical grid FFT with common wet mask and Hann taper; cycles/grid cell, not isotropic physical-wavenumber or DEX spectra. Gold, mean, member1 and average member power saved separately. Maps saturate at ±0.5 m/s; raw arrays remain unclipped.\n"
    )


if __name__ == "__main__":
    main()
