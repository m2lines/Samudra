#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize frozen-checkpoint annual components without creating a new selection score."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from plot_observation_rollout_maps import digest, plot  # type: ignore[import-not-found]


def summarize(source, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads(source.read_text())
    models = result["models"]
    origins = result["configuration"]["origins"]
    parents = [n for n in models if not n.endswith("-cooldown")]
    colors = dict(
        zip(parents, plt.get_cmap("tab10")(np.arange(len(parents))), strict=False)
    )
    leads, annual, ohc = [], [], []
    for name, record in models.items():
        for origin, scores in record["origins"].items():
            for forecast, key in [
                ("evolved", "metrics"),
                ("initialized persistence", "persistence"),
            ]:
                data = scores[key]
                common = dict(model=name, origin=origin, forecast=forecast)
                for lead, errors in data["leads"].items():
                    spectral = {
                        v + "_spectral_dex": statistics.mean(
                            s["error_dex"]
                            for k, s in data["spectra"].items()
                            if k.startswith(v + "/") and k.endswith("/day" + lead)
                        )
                        for v in ["sst", "adt"]
                    }
                    leads.append(
                        dict(**common, lead_days=int(lead), **errors, **spectral)
                    )
                annual.append(
                    dict(
                        **common,
                        annual_eke_rmse=data["annual_eke_rmse"],
                        annual_eke_spectral_dex=statistics.mean(
                            v["error_dex"]
                            for k, v in data["spectra"].items()
                            if k.startswith("annual-eke/")
                        ),
                    )
                )
                for i, (month, layers) in enumerate(
                    sorted(data["monthly_ohc"].items()), 1
                ):
                    ohc.append(
                        dict(
                            **common,
                            month=month,
                            forecast_month=i,
                            **{k + "_rmse": v for k, v in layers.items()},
                        )
                    )
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in [
        ("lead-metrics.csv", leads),
        ("annual-eke.csv", annual),
        ("monthly-ohc.csv", ohc),
    ]:
        with (output / filename).open("w") as stream:
            writer = csv.DictWriter(stream, list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    # Surface persistence is a common plotted control only if per-case metrics agree.
    for origin in origins:
        reference = models[parents[0]]["origins"][origin]["persistence"]
        for record in models.values():
            candidate = record["origins"][origin]["persistence"]
            if (
                candidate["leads"] != reference["leads"]
                or candidate["spectra"] != reference["spectra"]
            ):
                raise ValueError(
                    "Surface persistence differs; draw model-specific controls"
                )
    metrics = [
        ("sst_rmse", "SST RMSE (°C)"),
        ("adt_rmse", "ADT RMSE (m)"),
        ("velocity_rmse", "Geostrophic velocity RMSE (m/s)"),
        ("sst_spectral_dex", "SST spatial spectral error (dex)"),
        ("adt_spectral_dex", "ADT spatial spectral error (dex)"),
    ]
    for cohort in [None, *origins]:
        selected = [r for r in leads if cohort is None or r["origin"] == cohort]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        for ax, (metric, ylabel) in zip(axes.flat, metrics):
            for name in models:
                parent = name.removesuffix("-cooldown")
                points = [
                    r
                    for r in selected
                    if r["model"] == name and r["forecast"] == "evolved"
                ]
                x = sorted({r["lead_days"] for r in points})
                y = [
                    statistics.mean(r[metric] for r in points if r["lead_days"] == day)
                    for day in x
                ]
                ax.plot(
                    x,
                    y,
                    color=colors[parent],
                    ls="--" if name.endswith("-cooldown") else "-",
                    label=name,
                )
            points = [
                r
                for r in selected
                if r["model"] == parents[0]
                and r["forecast"] == "initialized persistence"
            ]
            ax.plot(
                x,
                [
                    statistics.mean(r[metric] for r in points if r["lead_days"] == day)
                    for day in x
                ],
                color="black",
                ls=":",
                label="Initialized persistence (surface metrics coincide)",
            )
            ax.set(xlabel="Forecast lead (days)", ylabel=ylabel)
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(
            *axes.flat[0].get_legend_handles_labels(),
            loc="center",
            fontsize=8,
            frameon=False,
        )
        fig.suptitle(
            "Continuous 365-day forecasts · "
            + (cohort or "mean of three per-origin scores")
            + "\nSolid: constant-rate parent; dashed: cooldown; lower is better"
        )
        stem = "lead-curves-" + (cohort or "mean")
        for ext in ["png", "pdf"]:
            fig.savefig(output / (stem + "." + ext), dpi=160)
        plt.close(fig)
    for layer in ["0_700", "700_2000"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for ax, parent in zip(axes.flat, parents):
            for name, color in [(parent, "#757575"), (parent + "-cooldown", "#0072B2")]:
                for forecast, style in [
                    ("evolved", "-"),
                    ("initialized persistence", "--"),
                ]:
                    points = [
                        r
                        for r in ohc
                        if r["model"] == name and r["forecast"] == forecast
                    ]
                    ax.plot(
                        range(1, 13),
                        [
                            statistics.mean(
                                r[layer + "_rmse"]
                                for r in points
                                if r["forecast_month"] == month
                            )
                            / 1e8
                            for month in range(1, 13)
                        ],
                        color=color,
                        ls=style,
                        label=(
                            "Cooldown"
                            if name.endswith("-cooldown")
                            else "Constant rate"
                        )
                        + " / "
                        + forecast,
                    )
            ax.set(
                title=parent,
                xlabel="Forecast calendar month",
                ylabel="OHC RMSE (10⁸ J/m²)",
            )
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(
            *axes.flat[0].get_legend_handles_labels(),
            loc="center",
            fontsize=8,
            frameon=False,
        )
        fig.suptitle(
            "Continuous-year OHC · "
            + layer.replace("_", "–")
            + " m · mean of three per-origin scores"
        )
        for ext in ["png", "pdf"]:
            fig.savefig(output / ("monthly-ohc-" + layer + "." + ext), dpi=160)
        plt.close(fig)
    for variable, prefix, suffix in [
        ("sst", "sst/", "/day365"),
        ("adt", "adt/", "/day365"),
        ("annual-eke", "annual-eke/", ""),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        first = models[parents[0]]["origins"][origins[0]]["metrics"]["spectra"]
        keys = sorted(k for k in first if k.startswith(prefix) and k.endswith(suffix))
        if len(keys) != 3:
            raise ValueError("Expected all three spectral regions")
        for ax, key in zip(axes, keys, strict=True):
            for name, record in models.items():
                curves = [
                    record["origins"][o]["metrics"]["spectra"][key] for o in origins
                ]
                wavenumbers = np.array(curves[0]["k_rad_km"])
                for c in curves:
                    np.testing.assert_array_equal(wavenumbers, c["k_rad_km"])
                y = np.mean([c["prediction_power"] for c in curves], axis=0)
                ax.loglog(
                    wavenumbers,
                    y,
                    color=colors[name.removesuffix("-cooldown")],
                    ls="--" if name.endswith("-cooldown") else "-",
                )
            reference = np.mean(
                [
                    models[parents[0]]["origins"][o]["metrics"]["spectra"][key][
                        "reference_power"
                    ]
                    for o in origins
                ],
                axis=0,
            )
            persistence = np.mean(
                [
                    models[parents[0]]["origins"][o]["persistence"]["spectra"][key][
                        "prediction_power"
                    ]
                    for o in origins
                ],
                axis=0,
            )
            ax.loglog(
                wavenumbers, reference, color="black", linewidth=2, label="Observations"
            )
            if variable == "annual-eke":
                # Holding a field fixed gives zero temporal-anomaly EKE;
                # floating-point residuals around 1e-58 are not physical power.
                ax.text(
                    0.03,
                    0.03,
                    "Persistence: zero temporal EKE (not on log axis)",
                    transform=ax.transAxes,
                    fontsize=7,
                )
            else:
                ax.loglog(
                    wavenumbers,
                    persistence,
                    color="black",
                    ls=":",
                    label="Initialized persistence",
                )
            ax.set(
                title=key.split("/")[1],
                xlabel="Wavenumber (rad/km)",
                ylabel="Power (evaluator units)",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(
            variable
            + " · mean power across three origins · "
            + (
                "within-year temporal anomalies"
                if variable == "annual-eke"
                else "day 365"
            )
            + "\nModel colors/styles match the lead-time curves"
        )
        for ext in ["png", "pdf"]:
            fig.savefig(output / (variable + "-spectra." + ext), dpi=160)
        plt.close(fig)
    for parent in parents:
        plot(
            source.parent / (parent + "-maps.npz"),
            output / "maps" / parent,
            global_observations=True,
        )
    (output / "provenance.json").write_text(
        json.dumps(
            dict(
                source_sha256=digest(source),
                script_sha256=digest(__file__),
                aggregation="Arithmetic mean of three per-origin scores; spectral plots instead show mean power",
                surface_persistence_equal=True,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.output)
