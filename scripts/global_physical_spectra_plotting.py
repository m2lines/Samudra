# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared plotting for day30 and day365 regional spectrum comparisons."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_physical_spectra import REGIONS  # type: ignore[import-not-found]

colors = {
    "obs00050": "#c94438",
    "obs00500": "#9c65c5",
    "obs02000": "#35a38f",
    "obs08000": "#2167ad",
    "scratch08000": "#c87526",
    "scratch16000": "#7b4caf",
    "OM4 (1°)": "#e29624",
    "Observations (1°)": "#222222",
    "Observations (native)": "#777777",
}


def plot_regional_spectra(
    spectra, stages, endpoints, output_root, description, prefix=""
):
    for variable, unit in [
        ("SST", "°C² km"),
        ("SSH", "m² km"),
        ("Geostrophic KE", "m² s⁻² km"),
    ]:
        for subset, chosen in [
            (
                "final",
                endpoints + ["OM4 (1°)", "Observations (1°)", "Observations (native)"],
            ),
            (
                "training",
                [k for k, _ in stages]
                + ["OM4 (1°)", "Observations (1°)", "Observations (native)"],
            ),
        ]:
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
            for ax, (region, xb, yb) in zip(axes.flat, REGIONS):
                for name in chosen:
                    curve = spectra[variable][region][name]
                    if "power" not in curve:
                        continue
                    label = dict(stages).get(name, name)
                    ax.loglog(
                        curve["k_rad_km"],
                        curve["power"],
                        label=label,
                        color=colors[name],
                        ls="--"
                        if name in ["OM4 (1°)", "Observations (native)"]
                        else "-",
                        lw=1.7 if name in ["obs08000", "Observations (1°)"] else 1.1,
                    )
                ax.set_title(
                    f"{region}: {xb[0]}–{xb[1]}°E, {yb[0]}–{yb[1]}°N", fontsize=10
                )
                ax.set_xlabel("Angular wavenumber (rad/km)")
                ax.set_ylabel("k × PSD (" + unit + ")")
                ax.grid(alpha=0.2, which="both")
                if variable == "Geostrophic KE" and region == "Niño 3.4":
                    ax.text(
                        0.5,
                        0.5,
                        "Unavailable:\ngeostrophy excludes ±5°",
                        ha="center",
                        transform=ax.transAxes,
                    )
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
            fig.suptitle(variable + ": " + description)
            fig.savefig(
                output_root
                / (
                    prefix
                    + subset
                    + "-"
                    + variable.lower().replace(" ", "-")
                    + "-spectra.png"
                ),
                dpi=140,
            )
            plt.close(fig)
            print("spectra", variable, subset, flush=True)
