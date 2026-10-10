#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify and summarize the fixed-budget six-arm continuation experiment."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter  # type: ignore[attr-defined]

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.metrics import spectra

ARMS = [
    "control",
    "multiscale",
    "replay",
    "multiscale-replay",
    "unroll12",
    "latent-jitter",
]
ORIGINS = ["2015-01-01", "2018-01-01", "2021-01-01"]
COMPONENTS = ["sst_rmse", "velocity_rmse", "ohc_0_700_rmse", "ohc_700_2000_rmse"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifests(inputs):
    return {
        k.split("/")[-2] if "/" in k else k: v
        for k, v in inputs.get("annual_manifests", inputs.get("data_manifests")).items()
    }


def endpoint(record, origin, day):
    month = origin[:4] + ("-01" if day == 30 else "-12")
    return {
        **record["leads"][str(day)],
        **{
            f"ohc_{layer}_rmse": record["monthly_ohc"][month][layer]
            for layer in ("0_700", "700_2000")
        },
    }


def pool(records):
    result = {
        k: float(np.sqrt(np.mean([r[k] ** 2 for r in records]))) for k in records[0]
    }
    assert all(np.isfinite(v) and v >= 0 for v in result.values())
    return result


def table(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def increments(fields, target, valid, area):
    """Four-direction grid increments; all traversed cells valid, periodic x.

    Returns weighted squared-error sums and weights, separately for members and
    their mean. This is target-gradient error, not an artifact-specific score.
    """
    sums = np.zeros(3)
    fields = np.where(valid, fields, 0)
    target = np.where(valid, target, 0)
    for lag in (1, 2, 4, 8):
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            mask = valid.copy()
            for step in range(1, lag + 1):
                mask &= np.roll(valid, (-dy * step, -dx * step), (-2, -1))
            if dy:
                mask[-lag:] = False
            weight = mask * area
            distance = lag * np.hypot(dy, dx)
            pred = (
                np.roll(fields, (-dy * lag, -dx * lag), (-2, -1)) - fields
            ) / distance
            truth = (
                np.roll(target, (-dy * lag, -dx * lag), (-2, -1)) - target
            ) / distance
            sums += [
                ((pred.mean(0) - truth) ** 2 * weight).sum(),
                (((pred - truth) ** 2).mean(0) * weight).sum(),
                weight.sum(),
            ]
    return sums


def maps(root, output, channel, label, limits):
    """Same January case and noise seed; native grid shown without interpolation."""
    sources = [np.load(root / a / "validation/members-2014-01.npz") for a in ARMS]
    first = sources[0]
    region = np.ix_(
        (first["lat"] >= 10) & (first["lat"] <= 50),
        (first["lon"] >= 150) & (first["lon"] <= 240),
    )
    target = first["day30_surface_reference"][channel][region]
    valid = first["mask"][channel][region].astype(bool) & np.isfinite(target)
    fields = [("Reference", target)] + [
        (a, s["day30"][0, channel][region]) for a, s in zip(ARMS, sources, strict=True)
    ]
    # Four image pixels per grid cell, with fixed margins and native aspect.
    height, width = target.shape
    panel_w, panel_h = width * 4, height * 4
    total_w, total_h = panel_w * 2 + 100, len(fields) * (panel_h + 32) + 135
    fig = plt.figure(figsize=(total_w / 100, total_h / 100), dpi=100)
    images = []
    for row, (name, field) in enumerate(fields):
        support = gaussian_filter(valid.astype(float), 2)
        smooth = gaussian_filter(np.where(valid, field, 0), 2) / np.maximum(
            support, 1e-12
        )
        for col, data in enumerate(
            (
                np.where(valid, field, np.nan),
                np.where(valid & (support > 0.999), field - smooth, np.nan),
            )
        ):
            ax = fig.add_axes(
                (
                    (35 + col * (panel_w + 30)) / total_w,
                    (total_h - 70 - row * (panel_h + 32) - panel_h) / total_h,
                    panel_w / total_w,
                    panel_h / total_h,
                )
            )
            im = ax.imshow(
                data,
                origin="lower",
                interpolation="nearest",
                cmap="turbo" if col == 0 else "RdBu_r",
                vmin=limits[col][0],
                vmax=limits[col][1],
            )
            ax.set_title(name + (" · draw 1" if col == 0 and row else ""), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                images.append(im)
    for col, im in enumerate(images):
        bar = fig.add_axes(
            (
                (35 + col * (panel_w + 30)) / total_w,
                35 / total_h,
                panel_w / total_w,
                10 / total_h,
            )
        )
        fig.colorbar(
            im,
            cax=bar,
            orientation="horizontal",
            label=label
            if col == 0
            else f"{label}: field minus 2-cell Gaussian smoothing",
        )
    fig.suptitle(
        f"Day-30 {label} · January 2014 · North Pacific\n10–50°N, 150–240°E; identical masks/scales; 4 image pixels per grid cell",
        fontsize=10,
    )
    save_png(fig, output / f"day30-{label.split()[0].lower()}-members.png", dpi=100)
    plt.close(fig)


def native_maps(root, output):
    """Whole-grid member/mean comparisons, one image pixel per source cell."""
    names = ["parent", *ARMS]
    sources = []
    for name in names:
        path = root / name / "native/fields-2015.npz"
        assert sha(path) == read(path.parent / "COMPLETE.json")["files"][path.name]
        with np.load(path) as archive:
            sources.append({key: archive[key] for key in archive.files})
    for data in sources[1:]:
        assert np.array_equal(
            data["reference"], sources[0]["reference"], equal_nan=True
        )
    for lead in (30, 365):
        index = list(sources[0]["leads_days"]).index(lead)
        for channel, label, limits in [
            (0, "SST", (-2, 32)),
            (1, "SSH", (-1, 2)),
            (2, "T550", (0, 15)),
            (3, "S550", (33, 36)),
        ]:
            target = sources[0]["reference"][index, channel]
            mask = sources[0]["mask"][channel].astype(bool)
            height, width = target.shape
            tw, th = 2 * width + 90, 8 * (height + 25) + 155
            fig = plt.figure(figsize=(tw / 100, th / 100), dpi=100)
            fields = [("OM4 reference", target, target)]
            for name, data in zip(names, sources, strict=True):
                fields.append(
                    (
                        name,
                        data["members"][:, index, channel].mean(0),
                        data["members"][0, index, channel],
                    )
                )
            for row, (name, mean, member) in enumerate(fields):
                for col, field in enumerate((mean, member)):
                    ax = fig.add_axes(
                        (
                            (30 + col * (width + 25)) / tw,
                            (th - 100 - row * (height + 25) - height) / th,
                            width / tw,
                            height / th,
                        )
                    )
                    im = ax.imshow(
                        np.where(mask, field, np.nan),
                        origin="lower",
                        interpolation="nearest",
                        cmap="viridis",
                        vmin=limits[0],
                        vmax=limits[1],
                    )
                    ax.set_title(
                        name
                        + (
                            (" · mean of 8" if col == 0 else " · draw 1") if row else ""
                        ),
                        fontsize=8,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
            bar = fig.add_axes((0.15, 45 / th, 0.7, 10 / th))
            fig.colorbar(
                im,
                cax=bar,
                orientation="horizontal",
                extend="both",
                label="m"
                if channel == 1
                else ("°C" if channel in (0, 2) else "salinity"),
            )
            fig.suptitle(
                f"OM4 initialized · 2015 · day {lead} · {label}\nIdentical fixed scales; values outside color limits clip. Native 1:1 grid pixels.",
                fontsize=9,
            )
            save_png(fig, output / f"native-day{lead}-{label.lower()}.png", dpi=100)
            plt.close(fig)


def scored_spectra(root, output):
    """Reconstruct scored mean spectra, then compute matching first-draw spectra."""
    rows = []
    curves = {}
    for arm in ARMS:
        directory = root / arm / "validation"
        done = read(directory / "COMPLETE.json")
        means, draws, references = [], [], []
        for origin in done["metrics"]["origins"]:
            with np.load(directory / f"members-{origin}.npz") as saved:
                means.append(saved["day30"][:, :2].mean(0))
                draws.append(saved["day30"][0, :2])
                references.append(saved["day30_surface_reference"])
                lat, lon, mask = saved["lat"], saved["lon"], saved["mask"][:2]
        support = np.isfinite(np.stack(references)) & mask.astype(bool)[None]
        for mode, values in (("mean", means), ("draw1", draws)):
            surface_values = np.where(support, np.stack(values), np.nan)
            for channel, fieldname in enumerate(("sst", "adt")):
                field = xr.DataArray(
                    surface_values[:, channel],
                    dims=("time", "lat", "lon"),
                    coords={
                        "time": np.arange(len(surface_values)),
                        "lat": lat,
                        "lon": lon,
                    },
                )
                for region, longitude, latitude in spectra.SPATIAL_REGIONS:
                    key = f"{fieldname}/{region}/day30"
                    original = done["metrics"]["spectra"][key]
                    k, power = spectra.region_spectrum(
                        field, longitude, latitude, name=region
                    )
                    cutoff = (
                        2
                        * np.pi
                        / (4 * spectra.METRES_PER_DEGREE / 1000 * np.max(np.diff(lat)))
                    )
                    keep = k <= cutoff
                    k, power = k[keep], power[keep]
                    np.testing.assert_allclose(k, original["k_rad_km"], rtol=1e-12)
                    if mode == "mean":
                        np.testing.assert_allclose(
                            power, original["prediction_power"], rtol=2e-4
                        )
                    reference = np.asarray(original["reference_power"])
                    curves[arm, mode, fieldname, region] = (k, power / reference)
                    for i in range(len(k)):
                        rows.append(
                            dict(
                                arm=arm,
                                mode=mode,
                                field=fieldname,
                                region=region,
                                k_rad_km=k[i],
                                reference_power=reference[i],
                                prediction_power=power[i],
                            )
                        )
    table(output / "spectra.csv", rows)
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for row, (region, _, _) in enumerate(spectra.SPATIAL_REGIONS):
        for col, field_label in enumerate(("sst", "adt")):
            ax = axes[row, col]
            for arm, color in zip(ARMS[:4], colors, strict=True):
                for mode, marker in (("mean", "s"), ("draw1", "^")):
                    k, ratio = curves[arm, mode, field_label, region]
                    ax.plot(
                        2 * np.pi / k,
                        ratio,
                        marker=marker,
                        markerfacecolor="white",
                        color=color,
                        label=f"{arm}: {mode}",
                        linewidth=1,
                    )
            ax.axhline(1, color="black", linewidth=1)
            ax.set_title(f"{field_label.upper()} · {region}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("Wavelength (km)")
            ax.set_ylabel("Power / reference power")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(
        "Day 30 · nine validation months · exact scored spectral bins\nSquares: mean of 8; triangles: first draw. Full values for all arms in spectra.csv"
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    save_png(fig, output / "day30-spectra.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--presentation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--updates", type=int, default=128)
    args = p.parse_args()
    if args.arms is not None:
        ARMS[:] = args.arms
    if args.updates <= 0 or args.updates % 2:
        raise ValueError("Expected a positive even update count")
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.presentation, "rt") as f:
        presentation = json.load(f)
    assert presentation["configuration"]["origins"] == ORIGINS
    climate = presentation["metrics"]["Training seasonal climatology"]
    expected = manifests(presentation["models"]["U-global"]["inputs"])
    native: list[dict[str, Any]] = []
    annual, validation, calibration, spatial, evidence = [], [], [], [], {}
    roots = {
        "parent": args.baseline / "final-annual-steps32",
        "deterministic": args.baseline / "global-endpoint-annual",
    }
    roots.update({a: args.root / a / "annual" for a in ARMS})
    for arm, directory in roots.items():
        inputs, done = read(directory / "input.json"), read(directory / "COMPLETE.json")
        assert manifests(inputs) == expected
        if arm in ARMS:
            assert inputs["counts"] == {
                "om4": 8000 + args.updates // 2,
                "observation": 8000 + args.updates // 2,
            }
            assert (
                inputs["step"] == 16000 + args.updates
                and inputs["members"] == 8
                and inputs["sampling_steps"] == 32
            )
            assert done["inputs"] == inputs
        records = {o: read(directory / f"{o}.json") for o in ORIGINS}
        evidence[arm] = {
            "inputs": inputs,
            "metric_hashes": {o: sha(directory / f"{o}.json") for o in ORIGINS},
        }
        for day in (30, 365):
            norm = pool([endpoint(climate[o], o, day) for o in ORIGINS])
            for origin in [*ORIGINS, "pooled"]:
                selected = ORIGINS if origin == "pooled" else [origin]
                values = pool([endpoint(records[o], o, day) for o in selected])
                annual.append(
                    dict(
                        arm=arm,
                        origin=origin,
                        day=day,
                        score=sum(values[k] / norm[k] for k in COMPONENTS) / 4,
                        **values,
                    )
                )
    for arm in ARMS:
        directory = args.root / arm
        done = read(directory / "validation/COMPLETE.json")
        assert done["checkpoint_sha256"] == evidence[arm]["inputs"]["checkpoint_sha256"]
        assert len(done["metrics"]["origins"]) == 9
        validation.append(
            dict(arm=arm, score=done["score"], **done["metrics"]["metrics"])
        )
        stats = read(directory / "validation/calibration.json")
        for group, section, nchan, indices in [
            ("SST", "surface", 2, [0]),
            ("SSH", "surface", 2, [1]),
            ("T", "interior", 28, list(range(14))),
            ("S", "interior", 28, list(range(14, 28))),
        ]:
            totals = {
                k: sum(
                    np.asarray(s[section][k]).reshape(-1, nchan)[:, indices].sum()
                    for s in stats
                )
                for k in (
                    "weight",
                    "mean_squared_error",
                    "ensemble_variance",
                    "fair_crps",
                    "coverage_90",
                )
            }
            calibration.append(
                dict(
                    arm=arm,
                    group=group,
                    spread_rmse_ratio=float(
                        np.sqrt(
                            totals["ensemble_variance"] / totals["mean_squared_error"]
                        )
                    ),
                    coverage90=totals["coverage_90"] / totals["weight"],
                    fair_crps=totals["fair_crps"] / totals["weight"],
                )
            )
        sums = np.zeros((2, 3))
        for name, checksum in done["files"].items():
            path = directory / "validation" / name
            assert sha(path) == checksum
            with np.load(path) as fields:
                area = np.cos(np.deg2rad(fields["lat"]))[:, None]
                for channel in range(2):
                    target = fields["day30_surface_reference"][channel]
                    valid = fields["mask"][channel].astype(bool) & np.isfinite(target)
                    sums[channel] += increments(
                        fields["day30"][:, channel], target, valid, area
                    )
        for channel, label in enumerate(("SST", "SSH")):
            spatial.append(
                dict(
                    arm=arm,
                    channel=label,
                    mean_increment_rmse=float(
                        np.sqrt(sums[channel, 0] / sums[channel, 2])
                    ),
                    member_increment_rmse=float(
                        np.sqrt(sums[channel, 1] / sums[channel, 2])
                    ),
                )
            )
    for arm in ["parent", *ARMS]:
        directory = args.root / arm / "native"
        done = read(directory / "COMPLETE.json")
        if arm in ARMS:
            assert (
                done["checkpoint_sha256"]
                == evidence[arm]["inputs"]["checkpoint_sha256"]
            )
        for year in (2015, 2018, 2021):
            record = read(directory / f"{year}.json")
            for day, values in record["metrics"].items():
                errors = np.asarray(values["normalized_mse"]).reshape(-1, 77).mean(0)
                native.append(
                    dict(
                        arm=arm,
                        year=year,
                        day=int(day),
                        latent_rms=record["latent_rms"][int(day) // 5],
                        sst_normalized_rmse=float(np.sqrt(errors[38])),
                        ssh_normalized_rmse=float(np.sqrt(errors[76])),
                        t_normalized_rmse=float(np.sqrt(errors[38:57].mean())),
                        s_normalized_rmse=float(np.sqrt(errors[57:76].mean())),
                    )
                )
    results = dict(
        annual=annual,
        validation=validation,
        calibration=calibration,
        spatial=spatial,
        native=native,
        provenance=evidence,
        presentation_sha256=sha(args.presentation),
    )
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, rows in (
        ("annual", annual),
        ("validation", validation),
        ("calibration", calibration),
        ("spatial", spatial),
        ("native", native),
    ):
        table(args.output / f"{name}.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    names = list(roots)
    for ax, day in zip(axes, (30, 365), strict=True):
        values = [
            next(
                r["score"]
                for r in annual
                if r["arm"] == a and r["day"] == day and r["origin"] == "pooled"
            )
            for a in names
        ]
        ax.barh(names, values, color=["0.6", "#009E73", *["#0072B2"] * len(ARMS)])
        ax.invert_yaxis()
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Climatology-normalized four-component RMSE ↓")
        if day == 365:
            ax.set_xscale("log")
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlim(right=max(values) * (2.0 if day == 365 else 1.35))
    save_png(fig, args.output / "annual-rmse.png", dpi=130)
    fig.savefig(args.output / "annual-rmse.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for arm in ["parent", *ARMS]:
        native_rows = [r for r in native if r["arm"] == arm]
        days = sorted({r["day"] for r in native_rows})
        for ax, key in zip(axes, ("latent_rms", "t_normalized_rmse"), strict=True):
            ax.plot(
                days,
                [np.mean([r[key] for r in native_rows if r["day"] == d]) for d in days],
                marker="o",
                label=arm,
            )
            ax.set_yscale("log")
            ax.set_xlabel("OM4 rollout lead (days)")
    axes[0].set_title("Latent RMS, averaged over three starts")
    axes[1].set_title("Temperature normalized RMSE, 19 depths")
    axes[0].legend(fontsize=7)
    save_png(fig, args.output / "native-stability.png", dpi=130)
    plt.close(fig)
    maps(args.root, args.output, 0, "SST °C", [(0, 30), (-0.7, 0.7)])
    maps(args.root, args.output, 1, "SSH m", [(-0.5, 1.2), (-0.08, 0.08)])
    native_maps(args.root, args.output)
    scored_spectra(args.root, args.output)
    print(json.dumps([r for r in annual if r["origin"] == "pooled"], indent=2))


if __name__ == "__main__":
    main()
