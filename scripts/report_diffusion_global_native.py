#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verified global OM4 member maps, persistence comparisons and temporal spectra proxies."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_latent_maps import grid
from samudra.experiments.diffusion_latent_native_summary import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = summarize(args.root)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with np.load(args.grid) as data:
        lat = data["lat"]
    for path in sorted(args.root.glob("fields-*.npz")):
        with np.load(path) as saved:
            members, truth = saved["members"][:, 0, -1], saved["truth"][0, -1]
            for channel, name in enumerate(saved["channels"]):
                values = members[:, channel]
                grid(
                    [truth[channel], values.mean(0), *values[:4]],
                    ["OM4 reference", "Eight-member mean"]
                    + [f"Member {i + 1}" for i in range(4)],
                    saved["mask"][channel],
                    lat,
                    args.output / f"{path.stem}-day30-{name}.png",
                    f"OM4 initialization; {path.stem}; day30; global wet support",
                    str(name),
                    show_loss_boundary=False,
                    distinguish_missing=True,
                )
    errors = result["errors"]
    names = errors["channels"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for group in ("thetao", "so", "uo", "vo", "zos"):
        indices = [i for i, name in enumerate(names) if name.split("_")[0] == group]
        for ax, key, label in (
            (axes[0], "normalized_mse", "Mean RMSE (standardized)"),
            (axes[1], "persistence_normalized_mse", "Mean / persistence RMSE"),
        ):
            values = np.sqrt(np.array(errors["normalized_mse"])[:, indices].mean(1))
            if key.startswith("persistence"):
                persistence = np.sqrt(np.array(errors[key])[:, indices].mean(1))
                values = np.divide(
                    values,
                    persistence,
                    out=np.full_like(values, np.nan),
                    where=persistence > 0,
                )
            ax.plot(errors["leads"], values, marker="o", label=group)
            ax.set(xlabel="Lead (days)", ylabel=label)
            ax.legend()
            ax.grid(alpha=0.2)
    axes[1].axhline(1, color="black", linewidth=0.7)
    save_png(fig, args.output / "native-errors.png", dpi=150)
    plt.close(fig)
    temporal = result["temporal"]
    chosen = [38, 76, 47, 66, 9]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(len(chosen))
    truth = np.array(temporal["truth_increment_mse"])[chosen]
    for offset, key, label in (
        (-0.18, "mean_increment_mse", "Ensemble mean"),
        (0.18, "member_increment_mse", "Individual members"),
    ):
        ratio = np.sqrt(np.array(temporal[key])[chosen] / truth)
        axes[0].bar(x + offset, ratio, width=0.36, label=label)
    axes[0].axhline(1, color="black")
    axes[0].set(ylabel="Five-day increment RMS / OM4 RMS", yscale="log")
    axes[0].legend()
    axes[1].bar(x, np.array(temporal["adjacent_member_correlation"])[chosen])
    axes[1].set(ylabel="Adjacent member-deviation correlation", ylim=(-1, 1))
    for ax in axes:
        ax.set_xticks(x, [names[i] for i in chosen])
        ax.grid(axis="y", alpha=0.2)
    save_png(fig, args.output / "native-temporal.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
