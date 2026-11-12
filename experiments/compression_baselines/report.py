# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Render figures and markdown tables from run.py output.

python report.py OUT   # writes OUT/*.png and OUT/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = ["ar1", "lpc", "lpc_ctx", "analog", "lpc_analog", "mix"]
VARS = ["thetao", "so", "uo", "vo"]


def main(out: Path):
    R = json.loads((out / "results.json").read_text())
    ch = {c["channel"]: c for c in R["channels"]}
    var_of = lambda c: c.rsplit("_", 1)[0] if c != "zos" else "zos"  # noqa: E731

    # bits saved vs persistence, and RMSE skill vs persistence, as a function of depth
    for key, fname, label in (
        ("bits", "bits_gain_vs_depth.png", "bits/value saved vs persistence"),
        (
            "rmse",
            "rmse_skill_vs_depth.png",
            "RMSE skill vs persistence (1 - RMSE/RMSE_p)",
        ),
    ):
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)
        for ax, v in zip(axes, VARS):
            cs = sorted([c for c in ch if var_of(c) == v], key=lambda c: ch[c]["depth"])
            if not cs:
                continue
            depth = [ch[c]["depth"] for c in cs]
            for m in MODELS:
                if key == "bits":
                    val = [ch[c]["bits"]["persistence"] - ch[c]["bits"][m] for c in cs]
                else:
                    val = [
                        1 - ch[c]["rmse"][m] / max(ch[c]["rmse"]["persistence"], 1e-30)
                        for c in cs
                    ]
                ax.plot(val, depth, marker="o", ms=3, label=m)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(v)
            ax.set_xlabel(label, fontsize=8)
            ax.set_yscale("symlog", linthresh=50)
            ax.invert_yaxis()
        axes[0].set_ylabel("depth (m)")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)

    # maps of bits saved by lpc_ctx over persistence
    mp = np.load(out / "maps.npz")
    keys = [k for k in mp.files if k.endswith("_gain")]
    if keys:
        fig, axes = plt.subplots(
            1, len(keys), figsize=(4.5 * len(keys), 3), squeeze=False
        )
        for ax, k in zip(axes[0], keys):
            g = mp[k]
            lim = np.nanpercentile(np.abs(g), 98) if np.isfinite(g).any() else 1
            im = ax.imshow(
                g, origin="lower", cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto"
            )
            ax.set_title(f"{k[:-5]}: bits saved (lpc_ctx)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out / "bits_gain_maps.png", dpi=90)
        # 256-colour palette keeps this figure under the repo's 250 KB file limit
        from PIL import Image

        img = Image.open(out / "bits_gain_maps.png").convert("RGB")
        img.quantize(colors=256).save(out / "bits_gain_maps.png", optimize=True)
        plt.close(fig)

    # headline: three numbers per variable, as mean over levels (surface level in brackets)
    def skill(c, m="mix"):
        return 1 - ch[c]["rmse"][m] / max(ch[c]["rmse"]["persistence"], 1e-30)

    lines = [
        "Headline, per variable: mean over its levels, with the surface level in brackets.",
        "",
        "| variable | levels | persistence bits | mix bits | bits saved by mix | RMSE skill of mix vs persistence | est. joint-coding gap (bits) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARS + ["zos"]:
        cs = sorted([c for c in ch if var_of(c) == v], key=lambda c: ch[c]["depth"])
        if not cs:
            continue
        s = cs[0]

        def fmt(f, nd=2, cs=cs, s=s):
            return f"{np.mean([f(c) for c in cs]):.{nd}f} [{f(s):.{nd}f}]"

        lines.append(
            f"| {v} | {len(cs)} | {fmt(lambda c: ch[c]['bits']['persistence'])} "
            f"| {fmt(lambda c: ch[c]['bits']['mix'])} "
            f"| {fmt(lambda c: ch[c]['bits']['persistence'] - ch[c]['bits']['mix'])} "
            f"| {fmt(skill, 3)} | {fmt(lambda c: ch[c]['joint_coding_gap_bits_est'])} |"
        )
    lines.append("")

    # tables
    lines += [
        "| channel | depth (m) | bin width | persistence | ar1 | lpc | lpc_ctx | analog | lpc_analog | mix | blosc residual bound |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(ch, key=lambda c: (var_of(c), ch[c]["depth"])):
        b = ch[c]["bits"]
        lines.append(
            f"| {c} | {ch[c]['depth']:g} | {ch[c]['bin_width']:.3g} | "
            + " | ".join(f"{b[m]:.2f}" for m in ["persistence"] + MODELS)
            + f" | {ch[c]['blosc_bits_per_value']['residual_vs_persistence']:.2f} |"
        )
    lines += [
        "",
        "| variable | channels | persistence bits | mix bits | saved | mean RMSE skill (mix) | mean RMSE skill (lpc) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARS + ["zos"]:
        cs = [c for c in ch if var_of(c) == v]
        if not cs:
            continue
        p = np.mean([ch[c]["bits"]["persistence"] for c in cs])
        mx = np.mean([ch[c]["bits"]["mix"] for c in cs])

        def sk(m, cs=cs):
            return np.mean(
                [
                    1 - ch[c]["rmse"][m] / max(ch[c]["rmse"]["persistence"], 1e-30)
                    for c in cs
                ]
            )

        lines.append(
            f"| {v} | {len(cs)} | {p:.2f} | {mx:.2f} | {p - mx:.2f} | {sk('mix'):.3f} | {sk('lpc'):.3f} |"
        )
    lines += [
        "",
        "| channel | truth field (blosc bits/value) | ar1 forecast | lpc forecast | ratio lpc/truth | lpc error zonal corr | est. joint-coding gap (bits) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(ch, key=lambda c: (var_of(c), ch[c]["depth"])):
        bb = ch[c]["blosc_bits_per_value"]
        lines.append(
            f"| {c} | {bb['truth']:.2f} | {bb['ar1_forecast']:.2f} | {bb['lpc_forecast']:.2f} | {bb['lpc_forecast'] / max(bb['truth'], 1e-9):.3f} | {ch[c]['lpc_error_zonal_corr']:.2f} | {ch[c]['joint_coding_gap_bits_est']:.2f} |"
        )
    for f in sorted(out.glob("ncd_*.json")):
        n = json.loads(f.read_text())
        lines += [
            "",
            f"NCD ablation, {n['channel']} ({n['library']} library patches, {n['queries']} queries, k={n['k']}, {n['bytes_per_patch']} bytes/patch):",
            "",
            "| neighbour search | bits/value | RMSE | seconds |",
            "|---|---:|---:|---:|",
        ]
        for m, r in n["methods"].items():
            lines.append(
                f"| {m} | {r['bits']:.3f} | {r['rmse']:.4g} | {r['seconds']:.1f} |"
            )
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
