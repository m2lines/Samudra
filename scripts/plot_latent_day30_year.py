# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Twelve-origin day30 mean-field spectra with native observation context."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_physical_spectra import REGIONS  # type: ignore[import-not-found]
from plot_latent_stopped_spectra import regional  # type: ignore[import-not-found]

from samudra.experiments.observation_pilot import digest


def mean_curve(fields, lat, lon, region, support):
    curves = [regional(field, lat, lon, region, support) for field in fields]
    if any(c is None for c in curves):
        raise ValueError(f"Insufficient common support: {region[0]}")
    for c in curves[1:]:
        np.testing.assert_array_equal(c["k_rad_km"], curves[0]["k_rad_km"])
    return dict(
        k_rad_km=curves[0]["k_rad_km"],
        power=np.mean([c["power"] for c in curves], axis=0).tolist(),
        shape=curves[0]["shape"],
        finite_fraction=curves[0]["finite_fraction"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("stopped-runs", "baseline", "references", "native", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path):
        sources[str(path)] = digest(path)
        return dict(np.load(path))

    grid = read(a.references / "grid.npz")
    lat, lon = grid["lat"], grid["lon"]
    om4 = read(a.references / "om4-day30.npz")["surface"]
    context = read(a.references / "obs08000-monthly.npz")
    roots = [a.baseline] + [
        a.stopped_runs / f"D-{seed}/report-v1" for seed in (1729, 1730)
    ]
    predictions = []
    for root in roots:
        data = read(root / "point-fields.npz")
        indices = np.flatnonzero(np.char.startswith(data["origins"], "2015-"))
        np.testing.assert_array_equal(data["origins"][indices], context["origins"])
        np.testing.assert_allclose(
            data["reference"][indices, 5], context["reference"], equal_nan=True
        )
        predictions.append(data["prediction"][indices, 5])
    result: dict = dict(
        scope="Average of separate day30 power spectra over all twelve monthly origins in2015. Diffusion curves use each origin's ensemble-mean field, not average member power; see separate member figures. Fixed coarse support over all twelve dates and all models. Native support is that geographic support projected by coarse-cell boundaries, intersected with native finite observations over all twelve dates.",
        sources=sources,
        curves={},
    )
    fig, axes = plt.subplots(2, len(REGIONS), figsize=(22, 8), layout="constrained")
    for row, (channel, source, unit) in enumerate(
        (("SST", "oisst", "°C² km"), ("SSH", "duacs", "m² km"))
    ):
        fields = [context["reference"][:, row], om4[:, row]] + [
            z[:, row] for z in predictions
        ]
        native_grid = read(a.native / f"{source}-grid.npz")
        native = []
        for month in range(1, 13):
            z = read(a.native / f"{source}-2015-{month:02}.npz")
            np.testing.assert_allclose(
                z["coarse"], fields[0][month - 1], equal_nan=True
            )
            native.append(z["native"])
        native = np.stack(native)
        support = grid["mask"][0 if row == 0 else 6].astype(
            bool
        ) & np.logical_and.reduce([np.isfinite(f).all(0) for f in fields])
        yi = np.searchsorted((lat[:-1] + lat[1:]) / 2, native_grid["lat"])
        xi = np.searchsorted((lon[:-1] + lon[1:]) / 2, native_grid["lon"])
        native_support = support[np.ix_(yi, xi)] & np.isfinite(native).all(0)
        for column, region in enumerate(REGIONS):
            ax = axes[row, column]
            curves = [mean_curve(f, lat, lon, region, support) for f in fields]
            curves.append(
                mean_curve(
                    native,
                    native_grid["lat"],
                    native_grid["lon"],
                    region,
                    native_support,
                )
            )
            names = [
                "Obs 1°",
                "OM4 1°",
                "Historical deterministic",
                "Stopped1729 mean",
                "Stopped1730 mean",
                "Obs native",
            ]
            for c, name, color, marker in zip(
                curves,
                names,
                ["black", "#c27700", "#9068a8", "#168445", "#206cb0", "#777777"],
                ["o", "^", "D", "P", "s", "v"],
                strict=True,
            ):
                ax.loglog(
                    c["k_rad_km"],
                    c["power"],
                    color=color,
                    marker=marker,
                    markevery=max(1, len(c["power"]) // 8),
                    label=name,
                )
            ax.set(
                title=f"{channel}: {region[0]}",
                xlabel="Wavenumber (rad/km)",
                ylabel=f"k P(k) ({unit})",
            )
            ax.grid(alpha=0.2)
            result["curves"][f"{channel}/{region[0]}"] = dict(
                zip(names, curves, strict=True)
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6)
    fig.suptitle(
        "Day30, all twelve monthly origins in2015: mean-field spectra; native observations on different grids"
    )
    fig.savefig(a.output / "day30-broad-spectra-2015.png", dpi=130)
    plt.close(fig)
    (a.output / "day30-broad-spectra-2015.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
