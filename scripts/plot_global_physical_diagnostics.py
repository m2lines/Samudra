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
from global_physical_heat_content import (  # type: ignore[import-not-found]
    cell_areas,
    heat_total,
)
from global_physical_spectra import (  # type: ignore[import-not-found]
    REGIONS,
    regional_spectra,
)
from global_physical_spectra_plotting import (  # type: ignore[import-not-found]
    colors,
    plot_regional_spectra,
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
parser.add_argument("--heat-context", type=Path)
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
physical_area = cell_areas(lat, lon)
heat_context = dict(np.load(args.heat_context)) if args.heat_context else {}
climate_ohc = (
    heat_context["climatology"]
    if heat_context
    else np.load(arrays_root / "climatology.npz")["ohc"]
)
STAGES = [
    ("obs00050", "Early: 50 obs"),
    ("obs00500", "Developing: 500 obs"),
    ("obs02000", "Middle: 2,000 obs"),
    ("obs08000", "Final: 8,000 obs"),
]
SCRATCH_STAGES = [
    (key, label)
    for key, label in [
        ("scratch08000", "Obs-only: 8,000 obs"),
        ("scratch16000", "Obs-only: 16,000 obs"),
    ]
    if key in meta["checkpoints"]
]
STAGES += SCRATCH_STAGES
ENDPOINTS = ["obs08000", *[key for key, _ in SCRATCH_STAGES]]
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
for values in monthly.values():
    np.testing.assert_array_equal(values["origins"], monthly["obs08000"]["origins"])
    np.testing.assert_allclose(
        values["reference"],
        monthly["obs08000"]["reference"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
print(
    "native/coarse exact-bin and checkpoint-reference equivalence verified", flush=True
)
results = {
    "checkpoints": meta["checkpoints"],
    "spectra": {},
    "maps": {},
    "profiles": {},
    "reference_coverage": {},
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
            if reference_label:
                missing = wet[c] & ~np.isfinite(truth[c])
                results["reference_coverage"][name] = {
                    "model_wet_cells": int(wet[c].sum()),
                    "observed_wet_cells": int((wet[c] & np.isfinite(truth[c])).sum()),
                    "missing_wet_cells": int(missing.sum()),
                    "missing_cosine_area_fraction": float(
                        np.sum(missing * area) / np.sum(wet[c] * area)
                    ),
                    "color_key": "gray = fixed model land/depth mask; white = missing reference within model wet mask",
                }
            results["maps"][name] = panel_plot(
                panels,
                f"{title}: {origin}, {phase} (gray: model land; white: missing)",
                field,
                figure_root / name,
                distinguish_missing=True,
            )
# Broad, masked pseudo-spectra. The fixed coarse common mask is shared by every coarse curve.

results["spectra"] = regional_spectra(
    {key: value["prediction"] for key, value in monthly.items()},
    monthly["obs08000"]["reference"],
    OM4,
    lat,
    lon,
    wet,
    native,
    native_grids,
)
plot_regional_spectra(
    results["spectra"],
    STAGES,
    ENDPOINTS,
    figure_root,
    "mean of twelve day-30 spatial power spectra, 2015",
)
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
# Absolute OHC totals use physical cell areas, with identical fixed support for all curves.
results["annual_heat_content"] = {}
results["annual_heat_content_by_model"] = {key: {} for key in ENDPOINTS}
results["annual_heat_content_definition"] = (
    "Sum column OHC (J/m2) times spherical cell area (m2), divided by 1e21 to give ZJ. "
    "Cell midpoint bounds and polar edges match observation remapping; Earth radius "
    "6371000 m. Each depth band's common observed/forecast/baseline support is fixed "
    "throughout its year. Binary model wet cells; no fractional-wet correction or "
    "extrapolation to unobserved cells. Temperature reference 0 deg C, rho=1035, cp=3850."
)
for origin in ["2015-01-01", "2018-01-01", "2021-01-01"]:
    a = meta["checkpoints"]["obs08000"]["annual"][origin]
    endpoint_arrays = {
        key: dict(np.load(arrays_root / (key + "-" + origin + ".npz")))
        for key in ENDPOINTS
    }
    initial_heat = {
        key: heat_context[key + "-" + origin]
        if heat_context
        else model_arrays["initial_ohc"]
        for key, model_arrays in endpoint_arrays.items()
    }
    annual_arrays = endpoint_arrays["obs08000"]
    month_indices = (
        pd.PeriodIndex(annual_arrays["months"], freq="M").month.to_numpy() - 1
    )
    heat: dict = {key: {} for key in ENDPOINTS}
    for c, var in enumerate(["ohc_0_700", "ohc_700_2000"]):
        climate = climate_ohc[month_indices, c]
        support = np.isfinite(annual_arrays["reference_ohc"][:, c]).all(
            0
        ) & np.isfinite(climate).all(0)
        for key, values in endpoint_arrays.items():
            np.testing.assert_array_equal(values["months"], annual_arrays["months"])
            np.testing.assert_allclose(
                values["reference_ohc"],
                annual_arrays["reference_ohc"],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            support &= np.isfinite(initial_heat[key][c])
        assert int(support.sum()) == a["support"][var]["cells"]
        for key, values in endpoint_arrays.items():
            series = {
                "forecast": heat_total(
                    values["predicted_ohc"][:, c], physical_area, support
                ),
                "observation": heat_total(
                    annual_arrays["reference_ohc"][:, c], physical_area, support
                ),
                "persistence": np.repeat(
                    heat_total(initial_heat[key][c], physical_area, support),
                    len(month_indices),
                ),
                "climatology": heat_total(climate, physical_area, support),
            }
            heat[key][var] = {
                "units": "ZJ",
                "series": {k: v.tolist() for k, v in series.items()},
                "mean_bias_ZJ": {
                    k: float(np.mean(v - series["observation"]))
                    for k, v in series.items()
                    if k != "observation"
                },
                "support_area_m2": float(physical_area[support].sum()),
                "surface_wet_area_fraction": float(
                    physical_area[support].sum() / physical_area[wet[0]].sum()
                ),
                "cells": int(support.sum()),
            }
    results["annual_heat_content"][origin] = heat["obs08000"]
    for key in ENDPOINTS:
        results["annual_heat_content_by_model"][key][origin] = heat[key]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, (var, title) in zip(
        axes.flat,
        [
            ("sst", "Global mean SST (°C)"),
            ("adt", "Global mean SSH / ADT (m)"),
            ("ohc_0_700", "OHC 0–700 m (ZJ; observed support)"),
            ("ohc_700_2000", "OHC 700–2000 m (ZJ; observed support)"),
        ],
    ):
        dates = pd.to_datetime(a["dates"] if var in ["sst", "adt"] else a["months"])
        series = (
            a["means"][var]
            if var in ["sst", "adt"]
            else heat["obs08000"][var]["series"]
        )
        if SCRATCH_STAGES:
            for key in ENDPOINTS:
                series_for_model = (
                    meta["checkpoints"][key]["annual"][origin]["means"][var]
                    if var in ["sst", "adt"]
                    else heat[key][var]["series"]
                )
                for baseline in ["observation", "climatology"]:
                    np.testing.assert_allclose(
                        series_for_model[baseline], series[baseline], rtol=0, atol=0
                    )
                ax.plot(
                    dates,
                    series_for_model["forecast"],
                    label=dict(STAGES)[key],
                    color=colors[key],
                )
                if var not in ["sst", "adt"]:
                    ax.plot(
                        dates,
                        series_for_model["persistence"],
                        label=dict(STAGES)[key] + " persistence",
                        color=colors[key],
                        ls=":",
                        alpha=0.65,
                    )
            baselines = [
                ("observation", "#222222", "-"),
                ("climatology", "#e29624", "--"),
            ]
            if var in ["sst", "adt"]:
                baselines.append(("persistence", "#888888", ":"))
        else:
            baselines = [
                ("forecast", "#2167ad", "-"),
                ("observation", "#222222", "-"),
                ("persistence", "#888888", ":"),
                ("climatology", "#e29624", "--"),
            ]
        for method, color, style in baselines:
            ax.plot(dates, series[method], label=method, color=color, ls=style)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=25)
    if SCRATCH_STAGES:
        legend_entries: dict = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            legend_entries.update(zip(labels, handles))
        fig.legend(
            legend_entries.values(),
            legend_entries.keys(),
            loc="outside lower center",
            ncol=3,
            fontsize=7,
        )
    else:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Physical-only endpoints: {origin}, annual series without detrending"
        if SCRATCH_STAGES
        else f"Final physical-only model: {origin}, annual series without detrending"
    )
    fig.savefig(figure_root / (origin + "-annual-global-means.png"), dpi=150)
    plt.close(fig)
# Day-30 errors: separate source trajectories, with both exposure and update-budget axes.
fig, axes = plt.subplots(
    2 if SCRATCH_STAGES else 1,
    2,
    figsize=(10, 7 if SCRATCH_STAGES else 4),
    layout="constrained",
    squeeze=False,
)
for row, count_field in enumerate(
    ["global_step", "observation"] if SCRATCH_STAGES else ["global_step"]
):
    for ax, metric, title in zip(
        axes[row],
        ["sst_rmse", "adt_rmse"],
        ["Day-30 SST RMSE (°C)", "Day-30 SSH RMSE (m)"],
    ):
        groups = [(STAGES[:4], "Mixed OM4 + obs", "#2167ad")]
        if SCRATCH_STAGES:
            groups.append((SCRATCH_STAGES, "Obs-only", "#c87526"))
        for stages, label, color in groups:
            chart_x = [
                meta["checkpoints"][k]["lineage"]["global_step"]
                if count_field == "global_step"
                else meta["checkpoints"][k]["lineage"]["task_counts"]["observation"]
                for k, _ in stages
            ]
            chart_y = [
                meta["checkpoints"][k]["day30"]["forecast"]["mean"][metric]
                for k, _ in stages
            ]
            ax.plot(chart_x, chart_y, "o-", label=label, color=color)
        all_x = [
            meta["checkpoints"][k]["lineage"]["global_step"]
            if count_field == "global_step"
            else meta["checkpoints"][k]["lineage"]["task_counts"]["observation"]
            for k, _ in STAGES
        ]
        for method, style in [("persistence", ":"), ("climatology", "--")]:
            baseline_y = meta["checkpoints"]["obs08000"]["day30"][method]["mean"][
                metric
            ]
            ax.plot(
                [min(all_x), max(all_x)],
                [baseline_y, baseline_y],
                style,
                label=method,
                color="#888888" if method == "persistence" else "#e29624",
            )
        ax.set_xlabel(
            "Total optimizer updates"
            if count_field == "global_step"
            else "Observation optimizer updates"
        )
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
            "represented_minus_native_ZJ": float(
                heat_total(difference, physical_area, common_ohc)
            ),
            "support_area_m2": float(physical_area[common_ohc].sum()),
            "cells": int(common_ohc.sum()),
        }
    results["ohc_representation_check"][context_month] = representation_records
results["ohc_representation_definition"] = (
    "Same preceding-December IAP analysis: integrate its remapped, model-depth "
    "temperature targets over the OM4 layer overlaps and subtract the remapped "
    "native-IAP-layer OHC integral. Area totals in ZJ and retained column means on finite context columns "
    "with complete annual observation support. This is a context check, not a "
    "correction to annual forecasts or a decomposition of annual model bias."
)
(figure_root / "results.json").write_text(
    json.dumps(results, indent=2, allow_nan=False) + "\n"
)
print("REPORT_RENDER_COMPLETE")
