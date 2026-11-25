# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Per-cell bits saved beside the estimated joint-coding gap, by depth, for four runs.

    python joint_gap_plot.py OUT.png OUT.csv [SAMUDRA_ERROR_CORR.csv]

Reads results.json from the 2° and 1° v2 runs, with and without forcing. Top
row: bits per value saved by the best per-cell model (`mix`) over persistence.
Bottom row: the estimated joint-coding gap, -1/2 log2(1 - rho^2), where rho is
the zonal correlation between neighbouring cells' `lpc` errors. That is how many
more bits per value a coder that sees whole fields could save over any per-cell
model, under a Gaussian approximation. If SAMUDRA_ERROR_CORR.csv (from
error_corr.py) is given, the bottom row also shows the same estimate for the
paper Samudra 2's one-step errors at 1° (black). Bits are comparable within a run; the
ε (bin width) differs between resolutions, so compare shapes across them.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = [
    ("om4_twodeg_v2_results", "2°, no forcing", "#1f6fb8", "-"),
    ("om4_twodeg_v2_forcing_results", "2°, with forcing", "#1f6fb8", "--"),
    ("om4_onedeg_v2_results", "1°, no forcing", "#c4520f", "-"),
    ("om4_onedeg_v2_forcing_results", "1°, with forcing", "#c4520f", "--"),
]
VARS = [
    ("thetao", "Temperature"),
    ("so", "Salinity"),
    ("uo", "Eastward velocity"),
    ("vo", "Northward velocity"),
    ("zos", "Sea surface height"),
]


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    data: dict[tuple[str, str], list[dict]] = {}
    for d, label, _, _ in RUNS:
        L = json.loads((HERE / d / "results.json").read_text())
        for c in L["channels"]:
            b = c["bits"]
            r = {
                "run": label,
                "channel": c["channel"],
                "depth_m": c["depth"],
                "bits_saved_mix": b["persistence"] - b["mix"],
                "joint_gap_bits_est": c["joint_coding_gap_bits_est"],
            }
            rows.append(r)
            data.setdefault((label, c["channel"].rsplit("_", 1)[0]), []).append(r)
    with open(sys.argv[2], "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s2 = {}
    if len(sys.argv) > 3:
        for r in csv.DictReader(open(sys.argv[3])):
            s2[r["channel"]] = float(r["model_gap_bits"])
    depth: dict[str, float] = {r["channel"]: r["depth_m"] for r in rows}

    fig, axes = plt.subplots(
        2, len(VARS), figsize=(16, 8.5), sharey=True, layout="constrained"
    )
    for col, (v, name) in enumerate(VARS):
        for row, key in enumerate(("bits_saved_mix", "joint_gap_bits_est")):
            ax = axes[row, col]
            for _, label, color, ls in RUNS:
                rs = sorted(data[(label, v)], key=lambda r: r["depth_m"])
                x = [r[key] for r in rs]
                y = [r["depth_m"] for r in rs]
                ax.plot(
                    x,
                    y,
                    ls,
                    color=color,
                    lw=2,
                    label=label,
                    marker="o" if v == "zos" else None,
                    ms=8,
                )
            if row == 1 and s2:
                cs = sorted(
                    (c for c in s2 if c.rsplit("_", 1)[0] == v), key=lambda c: depth[c]
                )
                ax.plot(
                    [s2[c] for c in cs],
                    [depth[c] for c in cs],
                    "-",
                    color="black",
                    lw=2,
                    label="Samudra 2 (1°, paper checkpoint)",
                    marker="o" if v == "zos" else None,
                    ms=8,
                )
            ax.set_yscale("log")
            ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(name, fontsize=12)
            if col == 0:
                ax.set_ylabel("Depth (m)")
        axes[0, col].set_xlabel("bits / value saved over persistence")
        axes[1, col].set_xlabel("est. joint-coding gap (bits / value)")
    axes[0, 0].invert_yaxis()  # shared y: once flips all
    fig.legend(
        *axes[1, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Top: what the best per-cell model (mix) saves over persistence.  "
        "Bottom: what a whole-field coder could save on top, estimated from neighbouring errors' correlation.\n"
        "One step (5 days), OM4 test window 2014–2022. Forcing = tauuo, tauvo, hfds. "
        "Sea surface height is a single level (dot).",
        fontsize=11,
    )
    fig.savefig(out, dpi=80)


if __name__ == "__main__":
    main()
