# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Plot all archived U-global and early variants separately by update budget."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.input.open() as stream:
        for row in csv.DictReader(stream):
            model = row["model"]
            if model.startswith(("U-global", "U-multitask-early")):
                key = (int(row["budget"]), model, row["wave"], int(row["seed"]))
                groups[key].append((int(row["update"]), float(row["rmse_blend"])))
    if not groups:
        raise ValueError("No matching archived curves")
    colors = {
        "U-global": "#202020",
        "U-multitask-early": "#0072B2",
        "U-multitask-early-latent": "#009E73",
        "U-multitask-early-fine": "#D55E00",
        "U-multitask-early-fine-latent": "#CC79A7",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for budget in sorted({key[0] for key in groups}):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.subplots_adjust(bottom=0.34, top=0.86, wspace=0.18)
        handles, labels = [], []
        late_values: list[float] = []
        for key, points in sorted(groups.items()):
            if key[0] != budget:
                continue
            _, model, wave, seed = key
            points.sort()
            updates, scores = zip(*points, strict=True)
            if len(set(updates)) != len(updates) or updates[-1] != budget:
                raise ValueError(f"Duplicate or incomplete curve: {key}")
            family = "U-global" if model.startswith("U-global") else model
            label = f"{model} · seed {seed}"
            if budget == 4000:
                label += " · original" if wave == "original 4k" else " · reproduction"
            style = "--" if wave == "original 4k" or seed == 1730 else "-"
            for ax in axes:
                (line,) = ax.plot(
                    updates,
                    scores,
                    color=colors[family],
                    linestyle=style,
                    linewidth=1.6,
                    marker="o",
                    markersize=2.4,
                    alpha=0.9,
                )
            handles.append(line)
            labels.append(label)
            late_values.extend(y for x, y in points if x >= budget - 1000)
            inventory.append(
                dict(
                    budget=budget, model=model, wave=wave, seed=seed, events=len(points)
                )
            )
        for ax in axes:
            ax.axhline(1, color="0.55", linewidth=0.8, linestyle=":")
            ax.set_xlabel("Total optimizer updates (OM4 + observations)")
            ax.set_ylabel("RMSE-only validation blend ↓")
            ax.grid(alpha=0.2)
        axes[0].set(title="Full training", xlim=(0, budget), ylim=(0.78, 2.4))
        margin = max(0.012, (max(late_values) - min(late_values)) * 0.08)
        axes[1].set(
            title="Final 1,000 updates · expanded vertical scale",
            xlim=(budget - 1000, budget),
            ylim=(min(late_values) - margin, max(late_values) + margin),
        )
        fig.suptitle(
            f"U-global and early-data variants · {budget:,}-update schedule",
            fontsize=17,
        )
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            ncol=2,
            fontsize=9,
            frameon=False,
        )
        fig.text(
            0.5,
            0.018,
            "Raw validation events; no smoothing. Dotted reference: seasonal climatology = 1. Same-seed reruns are separate executions.",
            ha="center",
            fontsize=9,
        )
        for extension in ("png", "pdf"):
            path = args.output / f"rmse-validation-{budget // 1000}k.{extension}"
            fig.savefig(path, dpi=110)
            Path(str(path) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
        plt.close(fig)
    (args.output / "provenance.json.gz").write_bytes(
        gzip.compress(
            (
                json.dumps(
                    dict(
                        source=str(args.input),
                        source_sha256=hashlib.sha256(
                            args.input.read_bytes()
                        ).hexdigest(),
                        curves=inventory,
                        smoothing=False,
                        selection_changed=False,
                    ),
                    indent=2,
                )
                + "\n"
            ).encode(),
            mtime=0,
        )
    )
    (args.output / "provenance.json.gz.license").write_text(
        "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
    )
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
