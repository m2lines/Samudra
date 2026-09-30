# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Day-30 maps, regional spectra and undetrended annual global means."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from global_physical_spectra import (  # type: ignore[import-not-found]
    REGIONS,
    geostrophic,
    spectrum,
)

sys.path.insert(0, str(Path(__file__).parent))
from plot_observation_missingness import (  # type: ignore[import-not-found]
    FIELDS,
    panel_plot,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arrays", type=Path, required=True)
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
arrays_root = args.arrays
native_root = args.native
figure_root = args.output
figure_root.mkdir(parents=True, exist_ok=True)
meta = json.loads((arrays_root / "COMPLETE.json").read_text())
grid = dict(np.load(arrays_root / "grid.npz"))
lat, lon = grid["lat"], grid["lon"]
wet = grid["mask"]
area = np.cos(np.deg2rad(lat))[:, None]
STAGES = [
    ("obs00050", "Early: 50 obs"),
    ("obs00500", "Developing: 500 obs"),
    ("obs02000", "Middle: 2,000 obs"),
    ("obs08000", "Final: 8,000 obs"),
]
monthly = {k: dict(np.load(arrays_root / (k + "-monthly.npz"))) for k, _ in STAGES}
OM4 = dict(np.load(arrays_root / "om4-day30.npz"))["surface"]
native_grids = {
    p: dict(np.load(native_root / (p + "-grid.npz"))) for p in ["oisst", "duacs"]
}
native = {
    p: np.stack(
        [
            np.load(native_root / (p + "-" + str(m) + ".npz"))["native"]
            for m in pd.period_range("2015-01", "2015-12", freq="M")
        ]
    )
    for p in ["oisst", "duacs"]
}
# Check that the native reference and existing evaluation are the same product/operator.
for c, p in enumerate(["oisst", "duacs"]):
    coarse = np.stack(
        [
            np.load(native_root / (p + "-" + str(m) + ".npz"))["coarse"]
            for m in pd.period_range("2015-01", "2015-12", freq="M")
        ]
    )
    target = monthly["obs08000"]["reference"][:, c]
    np.testing.assert_array_equal(np.isfinite(coarse), np.isfinite(target))
    np.testing.assert_allclose(coarse, target, atol=2e-6, rtol=1e-6, equal_nan=True)
print("native/coarse exact-bin equivalence verified", flush=True)
results = {
    "checkpoints": meta["checkpoints"],
    "spectra": {},
    "maps": {},
    "profiles": {},
}
for origin in ["2015-01-01", "2018-01-01", "2021-01-01"]:
    arrays = {
        k: dict(np.load(arrays_root / (k + "-" + origin + ".npz"))) for k, _ in STAGES
    }
    previous_month = str(pd.Period(origin, freq="M") - 1)
    interior = np.load(native_root / (previous_month + ".npz"))["values"]
    for phase in ["initial", "day30", "day365"]:
        for field in FIELDS:
            c, title, unit, limits, cmap = field
            panels = []
            for k, label in STAGES:
                a = arrays[k]
                z = (
                    a["initial"]
                    if phase == "initial"
                    else a["state"][0 if phase == "day30" else 1]
                )
                panels.append((label, z, grid))
            truth = np.full_like(arrays["obs08000"]["initial"], np.nan)
            reference_label: str | None = None
            if phase == "initial":
                raw = arrays["obs08000"]["initial_reference"]
                truth[0] = raw[0]
                truth[6] = raw[1]
                truth[4] = raw[2]
                truth[5] = raw[3]
                if c in [0, 6]:
                    reference_label = "Observations: last history bin"
                elif c in [4, 5]:
                    reference_label = "DUACS geostrophic proxy (not target)"
                elif c in [1, 2]:
                    truth[1] = interior[0, 9]
                    truth[2] = interior[1, 0]
                    reference_label = previous_month + " IAP monthly context"
            elif c in [0, 6]:
                raw = arrays["obs08000"]["reference"][5 if phase == "day30" else 72]
                truth[0] = raw[0]
                truth[6] = raw[1]
                reference_label = "Observations: matching five-day bin"
            elif c in [4, 5]:
                raw = arrays["obs08000"]["reference"][5 if phase == "day30" else 72]
                truth[4] = raw[2]
                truth[5] = raw[3]
                reference_label = "DUACS geostrophic proxy (not target)"
            if reference_label:
                panels.append((reference_label, truth, grid))
            name = f"{origin}-{phase}-channel{c}.png"
            results["maps"][name] = panel_plot(
                panels, f"{title}: {origin}, {phase}", field, figure_root / name
            )
# Broad, masked pseudo-spectra. The fixed coarse common mask is shared by every coarse curve.
colors = {
    "obs00050": "#c94438",
    "obs00500": "#9c65c5",
    "obs02000": "#35a38f",
    "obs08000": "#2167ad",
    "OM4 (1°)": "#e29624",
    "Observations (1°)": "#222222",
    "Observations (native)": "#777777",
}
for variable, channel, product, unit in [
    ("SST", 0, "oisst", "°C² km"),
    ("SSH", 1, "duacs", "m² km"),
    ("Geostrophic KE", 1, "duacs", "m² s⁻² km"),
]:
    results["spectra"][variable] = {}
    for region, xb, yb in REGIONS:
        iy = np.flatnonzero((lat >= yb[0]) & (lat <= yb[1]))
        ix = np.flatnonzero((lon >= xb[0]) & (lon <= xb[1]))
        s = np.ix_(iy, ix)
        reference = monthly["obs08000"]["reference"][:, channel]
        common = (
            wet[0 if channel == 0 else 6]
            & np.isfinite(reference).all(0)
            & np.isfinite(OM4[:, channel]).all(0)
        )
        if variable == "Geostrophic KE":
            # Identical coarse derivative support; finite land placeholders are not ocean.
            for source_ssh in [reference, OM4[:, channel]]:
                velocities = geostrophic(np.where(wet[6], source_ssh, np.nan), lat, lon)
                common &= np.logical_and.reduce(
                    [np.isfinite(v).all(0) for v in velocities]
                )
        curves = {
            k: (m["prediction"][:, channel], lat, lon, common)
            for k, m in monthly.items()
        }
        curves["OM4 (1°)"] = (OM4[:, channel], lat, lon, common)
        curves["Observations (1°)"] = (reference, lat, lon, common)
        ng = native_grids[product]
        nlat, nlon = ng["lat"], ng["lon"]
        # Nearest coarse-cell support in geographic coordinates; native product holes remain missing.
        near_y = np.abs(nlat[:, None] - lat[None]).argmin(1)
        near_x = np.abs(((nlon[:, None] - lon[None] + 180) % 360) - 180).argmin(1)
        nsupport = common[near_y[:, None], near_x[None]] & np.isfinite(
            native[product]
        ).all(0)
        curves["Observations (native)"] = (native[product], nlat, nlon, nsupport)
        records = {}
        for name, (z, ys, xs, support) in curves.items():
            y = np.flatnonzero((ys >= yb[0]) & (ys <= yb[1]))
            x = np.flatnonzero((xs >= xb[0]) & (xs <= xb[1]))
            block = np.ix_(y, x)
            fields = (
                [z]
                if variable != "Geostrophic KE"
                else list(
                    geostrophic(
                        np.where(wet[6], z, np.nan)
                        if name != "Observations (native)"
                        else z,
                        ys,
                        xs,
                    )
                )
            )
            available = support.copy()
            if variable == "Geostrophic KE":
                available &= (np.abs(ys) >= 5)[:, None]
            available &= np.logical_and.reduce([np.isfinite(f).all(0) for f in fields])
            stack = []
            base = None
            for t in range(12):
                component = [
                    spectrum(f[t][block], ys[y], xs[x], available[block])
                    for f in fields
                ]
                if any(v is None for v in component):
                    break
                base = component[0]
                stack.append(
                    np.mean([v["power"] for v in component], axis=0)
                    if len(component) == 2
                    else np.array(base["power"])
                )
            if len(stack) == 12:
                assert base is not None
                base["power"] = np.mean(stack, axis=0).tolist()
                base["origins"] = 12
                records[name] = base
            else:
                records[name] = {
                    "unavailable": "Insufficient finite common support; geostrophy excludes ±5°"
                }
        results["spectra"][variable][region] = records
    for subset, chosen in [
        (
            "final",
            ["obs08000", "OM4 (1°)", "Observations (1°)", "Observations (native)"],
        ),
        (
            "training",
            [k for k, _ in STAGES]
            + ["OM4 (1°)", "Observations (1°)", "Observations (native)"],
        ),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for ax, (region, xb, yb) in zip(axes.flat, REGIONS):
            for name in chosen:
                curve = results["spectra"][variable][region][name]
                if "power" not in curve:
                    continue
                label = dict(STAGES).get(name, name)
                ax.loglog(
                    curve["k_rad_km"],
                    curve["power"],
                    label=label,
                    color=colors[name],
                    ls="--" if name in ["OM4 (1°)", "Observations (native)"] else "-",
                    lw=1.7 if name in ["obs08000", "Observations (1°)"] else 1.1,
                )
            ax.set_title(f"{region}: {xb[0]}–{xb[1]}°E, {yb[0]}–{yb[1]}°N", fontsize=10)
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
        fig.suptitle(variable + ": mean of twelve day-30 spatial power spectra, 2015")
        fig.savefig(
            figure_root
            / (subset + "-" + variable.lower().replace(" ", "-") + "-spectra.png"),
            dpi=140,
        )
        plt.close(fig)
        print("spectra", variable, subset, flush=True)
# Descriptive distance on well-resolved wavelengths, separate from selection metrics.
results["spectral_distance"] = {}
for variable in ["SST", "SSH", "Geostrophic KE"]:
    distances: dict[str, dict[str, float]] = {}
    for stage, _ in STAGES:
        distances[stage] = {}
        for reference_name in ["Observations (1°)", "OM4 (1°)"]:
            region_values = []
            for region, _, _ in REGIONS[:4]:
                model_curve = results["spectra"][variable][region][stage]
                reference_curve = results["spectra"][variable][region][reference_name]
                spectral_k = np.array(model_curve["k_rad_km"])
                np.testing.assert_allclose(spectral_k, reference_curve["k_rad_km"])
                valid = spectral_k <= 0.5 * min(
                    model_curve["nyquist_rad_km"], reference_curve["nyquist_rad_km"]
                )
                region_values.append(
                    float(
                        np.mean(
                            np.abs(
                                np.log10(np.array(model_curve["power"])[valid])
                                - np.log10(np.array(reference_curve["power"])[valid])
                            )
                        )
                    )
                )
            distances[stage][reference_name] = float(np.mean(region_values))
    results["spectral_distance"][variable] = distances
results["spectral_distance_definition"] = (
    "Mean absolute log10 power difference: first average bins with k <= half the "
    "lower Nyquist within each of four western boundary current boxes, then give "
    "each box equal weight. Each curve is already the mean of twelve day-30 "
    "snapshot powers. Descriptive only; not used for checkpoint selection."
)
results["annual_mean_bias"] = {}
for origin, annual in meta["checkpoints"]["obs08000"]["annual"].items():
    results["annual_mean_bias"][origin] = {
        variable: {
            method: float(np.mean(np.array(values) - np.array(series["observation"])))
            for method, values in series.items()
            if method != "observation"
        }
        for variable, series in annual["means"].items()
    }
# Annual absolute global means: each series shares a fixed support throughout the year.
for origin in ["2015-01-01", "2018-01-01", "2021-01-01"]:
    a = meta["checkpoints"]["obs08000"]["annual"][origin]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, (var, title, scale) in zip(
        axes.flat,
        [
            ("sst", "Global mean SST (°C)", 1),
            ("adt", "Global mean SSH / ADT (m)", 1),
            ("ohc_0_700", "Mean OHC 0–700 m (GJ/m²)", 1e9),
            ("ohc_700_2000", "Mean OHC 700–2000 m (GJ/m²)", 1e9),
        ],
    ):
        dates = pd.to_datetime(a["dates"] if var in ["sst", "adt"] else a["months"])
        for method, color, style in [
            ("forecast", "#2167ad", "-"),
            ("observation", "#222222", "-"),
            ("persistence", "#888888", ":"),
            ("climatology", "#e29624", "--"),
        ]:
            ax.plot(
                dates,
                np.array(a["means"][var][method]) / scale,
                label=method,
                color=color,
                ls=style,
            )
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Final physical-only model: {origin}, annual means without detrending"
    )
    fig.savefig(figure_root / (origin + "-annual-global-means.png"), dpi=150)
    plt.close(fig)
# Exact day-30 surface error summaries and per-checkpoint exposure.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, metric, title in zip(
    axes,
    ["sst_rmse", "adt_rmse"],
    ["Day-30 SST RMSE (°C)", "Day-30 SSH RMSE (m)"],
):
    chart_x = [meta["checkpoints"][k]["lineage"]["global_step"] for k, _ in STAGES]
    for method, style in [
        ("forecast", "o-"),
        ("persistence", "o:"),
        ("climatology", "--"),
    ]:
        chart_y = [
            meta["checkpoints"][k]["day30"][method]["mean"][metric] for k, _ in STAGES
        ]
        ax.plot(chart_x, chart_y, style, label=method)
    ax.set_xlabel("Total optimizer updates")
    ax.set_ylabel(title)
    ax.grid(alpha=0.2)
    ax.legend()
fig.savefig(figure_root / "day30-training-errors.png", dpi=150)
plt.close(fig)
# Isolate the OHC quadrature/vertical-representation offset using the same analysis.
results["ohc_representation_check"] = {}
depth_edges = np.array(
    [0, 5, 15, 30, 50, 80, 130, 200, 300, 450, 650, 900, 1200, 1600, 2100]
)
for context_month, origin in [
    ("2014-12", "2015-01-01"),
    ("2017-12", "2018-01-01"),
    ("2020-12", "2021-01-01"),
]:
    ohc_context = dict(np.load(native_root / (context_month + ".npz")))
    annual_arrays = dict(np.load(arrays_root / ("obs08000-" + origin + ".npz")))
    representation_records: dict[str, dict[str, float | int]] = {}
    for band_index, (low, high) in enumerate([(0, 700), (700, 2000)]):
        thickness = np.maximum(
            0, np.minimum(depth_edges[1:], high) - np.maximum(depth_edges[:-1], low)
        )
        active = thickness > 0
        sampled_temperature = ohc_context["values"][0]
        represented_ohc = (
            np.sum(sampled_temperature[active] * thickness[active, None, None], axis=0)
            * 1035
            * 3850
        )
        common_ohc = (
            np.isfinite(sampled_temperature[active]).all(0)
            & np.isfinite(ohc_context["ohc"][band_index])
            & np.isfinite(annual_arrays["reference_ohc"][:, band_index]).all(0)
        )
        wet_weights = np.where(common_ohc, area, 0)
        difference = represented_ohc - ohc_context["ohc"][band_index]
        representation_records[f"{low}_{high}"] = {
            "represented_minus_native_J_m2": float(
                np.sum(np.where(common_ohc, difference, 0) * wet_weights)
                / wet_weights.sum()
            ),
            "cells": int(common_ohc.sum()),
        }
    results["ohc_representation_check"][context_month] = representation_records
results["ohc_representation_definition"] = (
    "Same preceding-December IAP analysis: integrate its remapped, model-depth "
    "temperature targets over the OM4 layer overlaps and subtract the remapped "
    "native-IAP-layer OHC integral. Cosine-area means on finite context columns "
    "with complete annual observation support. This is a context check, not a "
    "correction to annual forecasts or a decomposition of annual model bias."
)
(figure_root / "results.json").write_text(
    json.dumps(results, indent=2, allow_nan=False) + "\n"
)
print("REPORT_RENDER_COMPLETE")
