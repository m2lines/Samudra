# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU radial field/error spectra on saved, matched temperature patches."""

import csv
import gzip
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "quantization", HERE.parent / "soft-priors-25k/quantization/render.py"
)
assert spec and spec.loader
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
MODELS = [
    ("gold", "OM4", "black", "-"),
    ("new-temperature", "Small T-only classifier", "#009e73", "-"),
    ("new-all-fields", "Small all-fields classifier", "#d55e00", "-"),
    ("new-regression", "Small all-fields regression", "#0072b2", "-"),
    ("original-temperature-regression", "Large T-only regression", "#cc79a7", "--"),
    ("original-all-fields-regression", "Large all-fields regression", "#8b4513", "--"),
    ("samudra", "Samudra", "#777777", ":"),
    ("persistence", "Persistence", "#999900", ":"),
]


def main():
    rows = []
    figures = {}
    for kind in ("field", "error"):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
        figures[kind] = (fig, axes)
    with np.load(HERE / "report/source-fields.npz") as archive:
        for row, (region, label) in enumerate(
            [
                ("north-pacific", "North Pacific"),
                ("gulf-stream", "Gulf Stream"),
                ("agulhas", "Agulhas"),
            ]
        ):
            for col, (z, depth) in enumerate([(0, 2.5), (7, 250), (13, 1850)]):
                prefix = f"{region}-z{z:02d}/thetao/"
                wet = archive[prefix + "wet"]
                tile = helper.helpers.wet_patch(wet)
                assert tile is not None
                y, x, side = tile
                sl = np.s_[y : y + side, x : x + side]
                gold = archive[prefix + "gold"].astype(float)[sl]
                edges = np.linspace(0, 0.5, side // 2 + 1)
                for kind, (_, axes) in figures.items():
                    ax = axes[row, col]
                    for key, name, color, style in MODELS:
                        if kind == "error" and key == "gold":
                            continue
                        values = archive[prefix + key].astype(float)[sl]
                        assert np.isfinite(values).all()
                        if kind == "error":
                            values = values - gold
                        ft, freq = helper.fft(values)
                        power = abs(ft) ** 2
                        xx, yy = [], []
                        for i in range(len(edges) - 1):
                            mask = (freq > edges[i]) & (freq <= edges[i + 1])
                            if not mask.any():
                                continue
                            f = float(freq[mask].mean())
                            p = float(power[mask].sum())
                            xx.append(f)
                            yy.append(p)
                            rows.append(
                                dict(
                                    region=region,
                                    depth_m=depth,
                                    kind=kind,
                                    model=key,
                                    cycles_per_cell=f,
                                    power_C2=p,
                                    tile_y=y,
                                    tile_x=x,
                                    tile_side=side,
                                )
                            )
                        # Shell sums cover every nonzero mode up to radial Nyquist.
                        np.testing.assert_allclose(
                            sum(yy), power[(freq > 0) & (freq <= 0.5)].sum(), rtol=1e-12
                        )
                        ax.loglog(xx, yy, color=color, ls=style, lw=1.6, label=name)
                    ax.axvspan(0.25, 0.5, color="gray", alpha=0.10)
                    ax.set_title(f"{label} · {depth:g} m · {side}×{side} tile")
                    ax.grid(alpha=0.2, which="major")
                    ax.set_xlim(1 / 64, 0.5)
                    if row == 2:
                        ax.set_xlabel("Spatial frequency (cycles / native grid cell)")
                    if col == 0:
                        ax.set_ylabel("Power per radial shell (°C²)")
    for kind, (fig, axes) in figures.items():
        title = (
            "Temperature field spectra"
            if kind == "field"
            else "Temperature error spectra (prediction − OM4)"
        )
        fig.suptitle(
            title
            + "\nTarget 2016-07-21 · planar detrend + periodic Hann · shaded: wavelengths 2–4 cells",
            y=0.99,
            fontsize=16,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=4,
            frameon=False,
            fontsize=10,
        )
        fig.subplots_adjust(
            top=0.86, bottom=0.07, left=0.07, right=0.98, hspace=0.28, wspace=0.25
        )
        fig.savefig(HERE / f"report/temperature-{kind}-spectra.png", dpi=140)
        plt.close(fig)
    with gzip.open(HERE / "report/temperature-spectra.csv.gz", "wt") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
