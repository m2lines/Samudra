# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Day-30 broad regional spectra, with separate member and mean-field power."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_physical_spectra import REGIONS, spectrum  # type: ignore[import-not-found]

from samudra.experiments.observation_pilot import digest


def regional(field, lat, lon, region, support):
    _, xb, yb = region
    yi = np.flatnonzero((lat >= yb[0]) & (lat <= yb[1]))
    xi = np.flatnonzero((lon >= xb[0]) & (lon <= xb[1]))
    selection = np.ix_(yi, xi)
    return spectrum(field[selection], lat[yi], lon[xi], support[selection])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("stopped-runs", "references", "native", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    grid = dict(np.load(a.references / "grid.npz"))
    lat, lon = grid["lat"], grid["lon"]
    om4 = np.load(a.references / "om4-day30.npz")["surface"]
    ref = dict(np.load(a.references / "obs08000-monthly.npz"))
    result: dict = dict(
        scope="Day30 only. January2015 snapshot with eight individual readouts, power of their mean and mean of their powers. Regional plane-detrended windowed pseudo-spectra, not global spherical spectra or trajectory skill.",
        native_support="Native observations use the same coarse common geographic support projected by cell boundaries plus their native finite mask; native-versus-coarse differences include resolution, averaging and remaining support effects.",
        sources={},
        curves={},
    )
    for path in (
        a.references / "grid.npz",
        a.references / "om4-day30.npz",
        a.references / "obs08000-monthly.npz",
        a.native / "oisst-grid.npz",
        a.native / "duacs-grid.npz",
    ):
        result["sources"][str(path)] = digest(path)
    for seed in (1729, 1730):
        path = a.stopped_runs / f"D-{seed}/report-v1/members/2015-01.npz"
        data = dict(np.load(path))
        result["sources"][str(path)] = digest(path)
        np.testing.assert_allclose(lat, data["lat"])
        np.testing.assert_allclose(lon, data["lon"])
        fig, axes = plt.subplots(2, len(REGIONS), figsize=(22, 8), layout="constrained")
        for row, (channel, source, unit) in enumerate(
            (("thetao_0", "oisst", "°C² km"), ("zos", "duacs", "m² km"))
        ):
            ci = data["channel_names"].tolist().index(channel)
            members = data["states_at_leads"][:, 0, 2, ci]
            observed = data["observed_surface"][0, 5, row]
            np.testing.assert_allclose(
                observed, ref["reference"][0, row], equal_nan=True
            )
            native_path = a.native / f"{source}-2015-01.npz"
            native = dict(np.load(native_path))
            result["sources"][str(native_path)] = digest(native_path)
            native_grid = dict(np.load(a.native / f"{source}-grid.npz"))
            np.testing.assert_allclose(observed, native["coarse"], equal_nan=True)
            fields = [observed, om4[0, row], members.mean(0)]
            support = (
                data["mask"][ci].astype(bool)
                & np.logical_and.reduce([np.isfinite(f) for f in fields])
                & np.isfinite(members).all(0)
            )
            yi = np.searchsorted((lat[:-1] + lat[1:]) / 2, native_grid["lat"])
            xi = np.searchsorted((lon[:-1] + lon[1:]) / 2, native_grid["lon"])
            native_support = support[np.ix_(yi, xi)] & np.isfinite(native["native"])
            for column, region in enumerate(REGIONS):
                ax = axes[row, column]
                curves = [regional(f, lat, lon, region, support) for f in fields]
                member_curves = [
                    regional(f, lat, lon, region, support) for f in members
                ]
                native_curve = regional(
                    native["native"],
                    native_grid["lat"],
                    native_grid["lon"],
                    region,
                    native_support,
                )
                if (
                    any(c is None for c in curves + member_curves)
                    or native_curve is None
                ):
                    raise ValueError(f"Insufficient support: {region[0]} {channel}")
                for i, c in enumerate(member_curves):
                    ax.loglog(
                        c["k_rad_km"],
                        c["power"],
                        color="#bbbbbb",
                        alpha=0.65,
                        label="Individual readouts" if i == 0 else None,
                    )
                for c, name, color, marker in zip(
                    curves,
                    ["Obs 1°", "OM4 1°", "Spectrum of mean"],
                    ["black", "#c27700", "#206cb0"],
                    ["o", "^", "s"],
                    strict=True,
                ):
                    ax.loglog(
                        c["k_rad_km"],
                        c["power"],
                        color=color,
                        marker=marker,
                        label=name,
                    )
                mean_power = np.mean([c["power"] for c in member_curves], axis=0)
                ax.loglog(
                    curves[0]["k_rad_km"],
                    mean_power,
                    color="#c23a61",
                    marker="D",
                    label="Mean member spectrum",
                )
                ax.loglog(
                    native_curve["k_rad_km"],
                    native_curve["power"],
                    color="#168445",
                    marker="v",
                    markevery=max(1, len(native_curve["power"]) // 8),
                    label="Obs native grid",
                )
                ax.set(
                    title=f"{channel}: {region[0]}",
                    xlabel="Wavenumber (rad/km)",
                    ylabel=f"k P(k) ({unit})",
                )
                ax.grid(alpha=0.2)
                result["curves"][f"{seed}/{channel}/{region[0]}"] = dict(
                    observed=curves[0],
                    om4=curves[1],
                    ensemble_mean=curves[2],
                    individual_members=member_curves,
                    mean_member_power=mean_power.tolist(),
                    native=native_curve,
                )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=6)
        fig.suptitle(
            f"January 2015, day30, seed{seed}: members and ensemble mean; native observations are differently gridded"
        )
        fig.savefig(a.output / f"day30-broad-spectra-{seed}.png", dpi=130)
        plt.close(fig)
    (a.output / "day30-broad-spectra.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
