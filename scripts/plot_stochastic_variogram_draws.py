# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot saved independent draws with references, means, and perturbations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANT = "B-shared-deep-wide-variogram"
LABELS = {
    "uo_0": "Eastward velocity, 2.5 m",
    "vo_0": "Northward velocity, 2.5 m",
    "vo_3": "Northward velocity, 40 m",
    "vo_8": "Northward velocity, 375 m",
    "vo_10": "Northward velocity, 775 m",
}
LICENSE = (
    "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
)


def load(folder, month):
    with np.load(folder / "grid.npz") as source:
        grid = {key: source[key].copy() for key in source.files}
    stem = f"{VARIANT}-02000-{month}"
    with np.load(folder / f"{stem}-maps.npz") as source:
        members, gold = source["members"].copy(), source["gold"].copy()
    proof = json.loads((folder / f"input-{month}.json").read_text())
    record = json.loads((folder / f"{stem}.json").read_text())
    assert proof["members"] == 8 and proof["weights"] == "ema"
    assert record["variant"] == VARIANT and record["step"] == 2000
    assert record["weights"] == "ema" and record["month"] == month
    names = grid["names"][grid["selection"]].tolist()
    wet = grid["mask"][grid["selection"]].astype(bool)
    assert members.shape == (8, len(names), len(grid["lat"]), len(grid["lon"]))
    assert gold.shape == members.shape[1:]
    assert np.isfinite(members[:, wet]).all() and np.isfinite(gold[wet]).all()
    return grid, names, members, gold, wet, proof, record


def weighted_stats(field, gold, mask, lat):
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    weights = weights / weights.sum()
    x, y = (
        np.where(mask, field, 0).astype(np.float64),
        np.where(mask, gold, 0).astype(np.float64),
    )
    xc, yc = x - (weights * x).sum(), y - (weights * y).sum()
    corr = (weights * xc * yc).sum() / np.sqrt(
        (weights * xc**2).sum() * (weights * yc**2).sum()
    )
    return float(np.sqrt((weights * (x - y) ** 2).sum())), float(corr)


def panel(ax, field, wet, grid, limit, *, positive=False):
    ax.set_facecolor("#d5d5d5")
    image = ax.pcolormesh(
        grid["lon"],
        grid["lat"],
        np.where(wet, field, np.nan),
        cmap="viridis" if positive else "RdBu_r",
        vmin=0 if positive else -limit,
        vmax=limit,
        shading="nearest",
        rasterized=True,
    )
    ax.set(
        xlim=(0, 360), ylim=(-90, 90), xticks=[0, 120, 240, 360], yticks=[-60, 0, 60]
    )
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    return image


def save(fig, path):
    for extension in ("png", "pdf"):
        target = path.with_suffix("." + extension)
        fig.savefig(target, dpi=150, bbox_inches="tight")
        Path(str(target) + ".license").write_text(LICENSE)
    plt.close(fig)


def fields(data, selected, out, month, suffix, metrics):
    grid, names, members, gold, wet, proof, _ = data
    fig, axes = plt.subplots(
        len(selected), 6, figsize=(24, 2.8 * len(selected)), squeeze=False
    )
    fig.subplots_adjust(
        top=0.80 if len(selected) == 2 else 0.85,
        bottom=0.07,
        left=0.07,
        right=0.90,
        hspace=0.33,
        wspace=0.13,
    )
    for row, name in enumerate(selected):
        index = names.index(name)
        predicted, truth, mask = members[:, index], gold[index], wet[index]
        mean = predicted.mean(0)
        limit = float(
            np.quantile(
                np.abs(np.concatenate([truth[mask], predicted[:, mask].ravel()])), 0.995
            )
        )
        values = [truth, mean, *predicted[:4]]
        for column, (field, label) in enumerate(
            zip(
                values,
                [
                    "OM4 reference",
                    "Mean of all 8",
                    "Draw 1",
                    "Draw 2",
                    "Draw 3",
                    "Draw 4",
                ],
                strict=True,
            )
        ):
            image = panel(axes[row, column], field, mask, grid, limit)
            rmse, corr = weighted_stats(field, truth, mask, grid["lat"])
            axes[row, column].set_title(
                label + (f"\nr = {corr:+.2f}" if column else "\n"), fontsize=10
            )
            if column == 0:
                axes[row, column].set_ylabel(LABELS[name] + "\nLatitude", fontsize=10)
            else:
                axes[row, column].tick_params(labelleft=False)
            metrics.append(
                dict(
                    case=month,
                    channel=name,
                    output=label,
                    rmse_m_s=rmse,
                    correlation=corr,
                )
            )
        fig.colorbar(
            image,
            ax=list(axes[row]),
            fraction=0.008,
            pad=0.01,
            label="m/s; 0.5% tails clipped",
        )
    fig.suptitle(
        f"Shared deep + wide + variogram · 2,000 updates · EMA · {month} case\n"
        f"OM4 initialization; state midpoint {proof['history_midpoints'][-1][:10]}; independent draws, no smoothing",
        fontsize=13,
    )
    save(fig, out / f"draws-{suffix}-{month}")


def perturbations(data, out, month):
    grid, names, members, _, wet, _, _ = data
    fig, axes = plt.subplots(2, 5, figsize=(22, 6), squeeze=False)
    fig.subplots_adjust(
        top=0.80, bottom=0.07, left=0.08, right=0.95, hspace=0.32, wspace=0.18
    )
    for row, name in enumerate(("uo_0", "vo_0")):
        i = names.index(name)
        mask = wet[i]
        deviations = members[:, i] - members[:, i].mean(0)
        limit = float(np.quantile(np.abs(deviations[:, mask]), 0.995))
        for member in range(4):
            im = panel(axes[row, member], deviations[member], mask, grid, limit)
            axes[row, member].set_title(f"Draw {member + 1} − mean of 8", fontsize=10)
        spread = members[:, i].std(0, ddof=0)
        spread_limit = float(np.quantile(spread[mask], 0.995))
        spread_image = panel(
            axes[row, 4], spread, mask, grid, spread_limit, positive=True
        )
        axes[row, 4].set_title("Std. dev. of all 8", fontsize=10)
        axes[row, 0].set_ylabel(LABELS[name] + "\nLatitude", fontsize=10)
        for ax in axes[row, 1:]:
            ax.tick_params(labelleft=False)
        fig.colorbar(im, ax=list(axes[row, :4]), fraction=0.008, pad=0.01, label="m/s")
        fig.colorbar(
            spread_image, ax=axes[row, 4], fraction=0.025, pad=0.02, label="m/s"
        )
    fig.suptitle(
        f"Surface draw differences · {month} case · Shared deep + wide + variogram\n"
        "Independent draws, EMA, 2,000 updates; row-specific scales; no smoothing",
        fontsize=13,
    )
    save(fig, out / f"draw-perturbations-{month}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, str | float]] = []
    for month in ("2014-01", "2014-04"):
        surface = load(args.root / "production" / VARIANT / "initializer-ema", month)
        interior = load(args.root / "depth-audit", month)
        for key in ("lat", "lon", "names", "mask"):
            np.testing.assert_array_equal(surface[0][key], interior[0][key])
        assert surface[5] == interior[5]
        assert surface[6]["checkpoint_sha256"] == interior[6]["checkpoint_sha256"]
        fields(surface, ["uo_0", "vo_0"], args.output, month, "surface", metrics)
        fields(
            interior, ["vo_3", "vo_8", "vo_10"], args.output, month, "interior", metrics
        )
        perturbations(surface, args.output, month)
    path = args.output / "draw-metrics.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(metrics[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(metrics)
    Path(str(path) + ".license").write_text(LICENSE)


if __name__ == "__main__":
    main()
