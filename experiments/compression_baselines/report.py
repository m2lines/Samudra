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

    # headline: three numbers per variable. Aggregate = bits per wet value over all
    # levels (each level weighted by its wet area); surface level in brackets.
    wet_area, dz = R["meta"]["wet_area"], R["meta"]["dz"]

    def skill(c, m="mix"):
        return 1 - ch[c]["rmse"][m] / max(ch[c]["rmse"]["persistence"], 1e-30)

    def saved(c):
        return ch[c]["bits"]["persistence"] - ch[c]["bits"]["mix"]

    lines = [
        f"Bin tolerance eps = {ch[next(iter(ch))]['eps']:g} sigma_delta (Keisler-style, per variable).",
        "",
        "| variable | sigma_delta (5-day change) | bin width |",
        "|---|---:|---:|",
    ]
    for v, sg in R["meta"]["sigma_delta"].items():
        w = next(ch[c]["bin_width"] for c in ch if var_of(c) == v)
        lines.append(f"| {v} | {sg:.4g} | {w:.4g} |")
    lines += [
        "",
        "Headline, per variable: bits per wet value over all levels (levels weighted by",
        "wet area), with the surface level in brackets. The last column weights levels",
        "by wet area x layer thickness instead.",
        "",
        "| variable | levels | persistence bits | mix bits | bits saved by mix | RMSE skill of mix vs persistence | est. joint-coding gap (bits) | bits saved, thickness-weighted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARS + ["zos"]:
        cs = sorted([c for c in ch if var_of(c) == v], key=lambda c: ch[c]["depth"])
        if not cs:
            continue
        s = cs[0]

        def fmt(f, nd=2, cs=cs, s=s, wts=None):
            w = [wet_area[c] for c in cs] if wts is None else wts
            return f"{np.average([f(c) for c in cs], weights=w):.{nd}f} [{f(s):.{nd}f}]"

        thick = [wet_area[c] * dz[c] for c in cs]
        lines.append(
            f"| {v} | {len(cs)} | {fmt(lambda c: ch[c]['bits']['persistence'])} "
            f"| {fmt(lambda c: ch[c]['bits']['mix'])} | {fmt(saved)} "
            f"| {fmt(skill, 3)} | {fmt(lambda c: ch[c]['joint_coding_gap_bits_est'])} "
            f"| {fmt(saved, wts=thick)} |"
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
