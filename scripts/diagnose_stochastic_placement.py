# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU diagnostics of amplitude and displacement from saved initializer members.

Input is the grid.npz / *-maps.npz / matching JSON output of
scripts/diagnose_stochastic_velocity_wave2.py. No model evaluation is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter  # type: ignore[attr-defined]

REGIONS = {
    "global": (-90, 90, 0, 360),
    "north-pacific": (20, 40, 150, 190),
    "gulf-stream": (25, 45, 280, 320),
    "equatorial-pacific": (-10, 10, 140, 260),
}
THRESHOLDS = (0.03, 0.1, 0.3, 1.0)
RADII = (0, 1, 2, 4, 8)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_mean(values, weights):
    denominator = weights.sum()
    return float(np.sum(np.where(weights > 0, values, 0) * weights) / denominator)


def weighted_quantile(values, weights, probability):
    valid = weights > 0
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][min(np.searchsorted(cdf, probability), len(order) - 1)])


def neighborhood_fraction(event, mask, lat, radius):
    """Cosine-weighted wet-cell event fraction; longitude wraps, latitude does not.

    Windows can cross land geometrically; this is not ocean-path displacement.
    Invalid cells contribute neither events nor area. Radius is in grid cells.
    """
    if radius < 0 or int(radius) != radius:
        raise ValueError("Radius must be a nonnegative integer")
    weight = np.asarray(mask, dtype=float) * np.cos(np.deg2rad(lat))[:, None]
    numerator = uniform_filter(
        np.where(mask, event, 0) * weight,
        size=2 * radius + 1,
        mode=("constant", "wrap"),
    )
    denominator = uniform_filter(weight, size=2 * radius + 1, mode=("constant", "wrap"))
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-14
    )


def fractions_score(prediction, reference, weights):
    """FSS and additive components; undefined if neither field has an event."""
    mse = weighted_mean((prediction - reference) ** 2, weights)
    norm = weighted_mean(prediction**2 + reference**2, weights)
    return dict(
        fraction_mse=mse,
        fraction_norm=norm,
        fss=None if norm <= 1e-20 else 1 - mse / norm,
    )


def regional_spectrum(u, v, mask):
    """Hann-window vector power in cycles/grid-cell, common masked support.

    Frequencies are computational-grid frequencies, not physical isotropic
    wavenumbers. Demean over common wet cells before tapering. Sum of spectral
    power equals taper-weighted mean squared vector anomaly by Parseval.
    """
    h, w = mask.shape
    taper = np.hanning(h)[:, None] * np.hanning(w)[None, :] * mask
    if taper.sum() <= 0:
        raise ValueError("Spectrum has no tapered wet support")
    power = np.zeros((h, w), dtype=float)
    for field in (u, v):
        anomaly = np.where(mask, field - field[mask].mean(), 0)
        power += abs(np.fft.fft2(anomaly * taper)) ** 2 / (h * w * np.sum(taper**2))
    k = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    edges = np.linspace(0, np.sqrt(0.5), 17)
    values, _ = np.histogram(k.ravel(), bins=edges, weights=power.ravel())
    counts, _ = np.histogram(k.ravel(), bins=edges)
    return edges, values, counts


def analyze_case(members, gold, masks, names, lat, lon, metadata):
    """Return amplitude, neighborhood and spectra rows for all paired u/v levels."""
    if (
        members.ndim != 4
        or gold.shape != members.shape[1:]
        or masks.shape != gold.shape
    ):
        raise ValueError("Expected members M,C,H,W and matching gold/mask C,H,W")
    if not np.isfinite(members[:, masks]).all() or not np.isfinite(gold[masks]).all():
        raise ValueError("Non-finite values on wet support")
    amplitude, neighborhoods, spectra = [], [], []
    names = list(names)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((len(lat), len(lon)))
    for name in names:
        if not name.startswith("uo_") or name.replace("uo_", "vo_") not in names:
            continue
        ui, vi = names.index(name), names.index(name.replace("uo_", "vo_"))
        depth = int(name.split("_")[1])
        common = masks[ui] & masks[vi]
        truth_speed = np.hypot(gold[ui], gold[vi])
        fields = [
            (f"member-{i + 1}", members[i, ui], members[i, vi])
            for i in range(len(members))
        ]
        fields += [("ensemble-mean", members[:, ui].mean(0), members[:, vi].mean(0))]
        fields += [
            ("reference", gold[ui], gold[vi]),
            ("zero", np.zeros_like(gold[ui]), np.zeros_like(gold[vi])),
        ]
        truth_fractions = {
            (threshold, radius): neighborhood_fraction(
                truth_speed >= threshold, common, lat, radius
            )
            for threshold in THRESHOLDS
            for radius in RADII
        }
        for sample, u, v in fields:
            speed = np.hypot(u, v)
            fractions = {
                (threshold, radius): neighborhood_fraction(
                    speed >= threshold, common, lat, radius
                )
                for threshold in THRESHOLDS
                for radius in RADII
            }
            for region, (south, north, west, east) in REGIONS.items():
                yy = (lat >= south) & (lat <= north)
                xx = (lon % 360 >= west) & (lon % 360 < east)
                support = common & yy[:, None] & xx[None, :]
                if not support.any():
                    continue
                weights = area * support
                base = dict(
                    **metadata,
                    sample=sample,
                    depth=depth,
                    region=region,
                    wet_cells=int(support.sum()),
                    area_weight=float(weights.sum()),
                )
                amplitude.append(
                    dict(
                        **base,
                        vector_power=weighted_mean(speed**2, weights),
                        vector_mse=weighted_mean(
                            (u - gold[ui]) ** 2 + (v - gold[vi]) ** 2, weights
                        ),
                        speed_mean=weighted_mean(speed, weights),
                        **{
                            f"speed_q{int(100 * q)}": weighted_quantile(
                                speed, weights, q
                            )
                            for q in (0.5, 0.9, 0.99)
                        },
                        **{
                            f"exceedance_{threshold:g}": weighted_mean(
                                speed >= threshold, weights
                            )
                            for threshold in THRESHOLDS
                        },
                    )
                )
                for threshold in THRESHOLDS:
                    for radius in RADII:
                        neighborhoods.append(
                            dict(
                                **base,
                                threshold=threshold,
                                radius=radius,
                                **fractions_score(
                                    fractions[threshold, radius],
                                    truth_fractions[threshold, radius],
                                    weights,
                                ),
                            )
                        )
                if region != "global" and yy.sum() >= 4 and xx.sum() >= 4:
                    crop = np.ix_(yy, xx)
                    edges, power, counts = regional_spectrum(
                        u[crop], v[crop], common[crop]
                    )
                    for lo, hi, value, count in zip(
                        edges[:-1], edges[1:], power, counts, strict=True
                    ):
                        spectra.append(
                            dict(
                                **base,
                                k_low=float(lo),
                                k_high=float(hi),
                                modes=int(count),
                                power=float(value),
                                wet_fraction=float(common[crop].mean()),
                            )
                        )
    return amplitude, neighborhoods, spectra


def plots(amplitude, neighborhoods, spectra, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    members = amplitude[amplitude["sample"].str.startswith("member-")]
    gold = amplitude[amplitude["sample"] == "reference"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for index, (variant, frame) in enumerate(members.groupby("variant")):
        marker = markers[index % len(markers)]
        surface = frame[(frame.depth == 0) & (frame.region == "global")]
        reference = gold[
            (gold.variant == variant) & (gold.depth == 0) & (gold.region == "global")
        ]
        axes[0, 0].scatter(
            index,
            surface.vector_power.mean() / reference.vector_power.mean(),
            marker=marker,
            label=variant,
        )
        axes[0, 1].plot(
            THRESHOLDS,
            [surface[f"exceedance_{t:g}"].mean() or np.nan for t in THRESHOLDS],
            marker=marker,
            label=variant,
        )
        subset = neighborhoods[
            (neighborhoods.variant == variant)
            & (neighborhoods.depth == 0)
            & (neighborhoods.region == "global")
            & (neighborhoods.threshold == 0.1)
            & neighborhoods["sample"].str.startswith("member-")
        ]
        grouped = subset.groupby("radius")[["fraction_mse", "fraction_norm"]].mean()
        axes[1, 0].plot(
            grouped.index,
            1 - grouped.fraction_mse / grouped.fraction_norm,
            marker=marker,
        )
        subset = spectra[
            (spectra.variant == variant)
            & (spectra.depth == 0)
            & (spectra.region == "north-pacific")
            & spectra["sample"].str.startswith("member-")
        ]
        curve = subset.groupby("k_low").power.mean()
        axes[1, 1].plot(curve.index, curve, marker=marker, markersize=3)
    reference = gold[(gold.depth == 0) & (gold.region == "global")]
    axes[0, 1].plot(
        THRESHOLDS,
        [reference[f"exceedance_{t:g}"].mean() for t in THRESHOLDS],
        color="black",
        marker="+",
        label="OM4 reference",
    )
    reference = (
        spectra[
            (spectra.depth == 0)
            & (spectra.region == "north-pacific")
            & (spectra["sample"] == "reference")
        ]
        .groupby("k_low")
        .power.mean()
    )
    axes[1, 1].plot(
        reference.index, reference, color="black", marker="+", label="OM4 reference"
    )
    axes[0, 0].axhline(1, color="black", linewidth=0.8)
    axes[0, 0].set(
        title="Global surface member vector power / OM4",
        ylabel="Power ratio",
        xticks=range(members.variant.nunique()),
    )
    axes[0, 0].set_xticklabels(
        sorted(members.variant.unique()), rotation=15, fontsize=8
    )
    axes[0, 1].set(
        title="Surface speed exceedance (all wet cells)",
        xlabel="Speed threshold (m/s)",
        ylabel="Area fraction",
        xscale="log",
        yscale="log",
    )
    axes[1, 0].set(
        title="Surface speed ≥ 0.1 m/s: member FSS",
        xlabel="Neighborhood radius (grid cells)",
        ylabel="FSS (higher is better)",
        ylim=(0, 1),
    )
    axes[1, 1].set(
        title="North Pacific surface member vector spectrum",
        xlabel="Cycles/grid cell",
        ylabel="Band-integrated power (m²/s²)",
        yscale="log",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7)
    fig.savefig(output / "placement-amplitude.png", dpi=160)
    fig.savefig(output / "placement-amplitude.pdf")
    plt.close(fig)


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input / "grid.npz") as grid:
        lat, lon = grid["lat"], grid["lon"]
        selection = grid["selection"]
        masks, names = grid["mask"][selection], grid["names"][selection]
    rows: list[list[dict[str, Any]]] = [[], [], []]
    sources = []
    gold_by_month: dict[str, np.ndarray] = {}
    cohorts: dict[tuple[str, int, str], list[str]] = {}
    for path in sorted(args.input.glob("*-maps.npz")):
        receipt_path = path.with_name(path.name.removesuffix("-maps.npz") + ".json")
        receipt = json.loads(receipt_path.read_text())
        if args.variants and receipt["variant"] not in args.variants:
            continue
        metadata = {key: receipt[key] for key in ("variant", "step", "month")}
        # The legacy diagnostic omitted a weight label and always selected EMA.
        metadata["weights"] = receipt.get("weights", "ema")
        key = (metadata["variant"], metadata["step"], metadata["weights"])
        cohorts.setdefault(key, []).append(metadata["month"])
        with np.load(path) as maps:
            if metadata["month"] in gold_by_month:
                np.testing.assert_array_equal(
                    maps["gold"], gold_by_month[metadata["month"]]
                )
            else:
                gold_by_month[metadata["month"]] = maps["gold"].copy()
            result = analyze_case(
                maps["members"], maps["gold"], masks, names, lat, lon, metadata
            )
        for target, values in zip(rows, result, strict=True):
            target.extend(values)
        sources.append(
            dict(
                path=str(path),
                sha256=fingerprint(path),
                receipt_sha256=fingerprint(receipt_path),
                checkpoint_sha256=receipt["checkpoint_sha256"],
            )
        )
    if not sources:
        raise ValueError("No selected maps")
    expected = sorted(next(iter(cohorts.values())))
    if any(
        sorted(months) != expected or len(set(months)) != len(months)
        for months in cohorts.values()
    ):
        raise ValueError(f"Unequal or duplicate date cohorts: {cohorts}")
    frames = [pd.DataFrame(values) for values in rows]
    for name, frame in zip(
        ("amplitude", "neighborhood", "spectra"), frames, strict=True
    ):
        frame.to_csv(args.output / f"{name}.csv", index=False)
    plots(frames[0], frames[1], frames[2], args.output)
    notes = {
        "complete": True,
        "sources": sources,
        "script_sha256": fingerprint(Path(__file__)),
        "grid_sha256": fingerprint(args.input / "grid.npz"),
        "regions": REGIONS,
        "thresholds_m_s": THRESHOLDS,
        "radii_grid_cells": RADII,
        "members_are_scored_separately": True,
        "dates": expected,
        "notes": [
            "All amplitudes use all paired wet u/v cells in a fixed region, never reference-speed-selected cells.",
            "Vector power is u²+v², twice kinetic energy per unit mass; ensemble-mean velocity is averaged before speed.",
            "Cosine latitude weights; grid-cell neighborhoods have latitude-dependent physical distances.",
            "Neighborhood FSS relaxes placement; it is not a proper probabilistic score, a transport estimate or proof of realistic members.",
            "Neighborhoods exclude dry-cell area but can connect across land geometrically; outer neighborhood windows may extend beyond regional boundaries.",
            "Spectra are demeaned masked Hann-window FFTs in cycles/grid cell, not DEX or spherical physical-wavenumber spectra; compare identical masks only.",
            "Reference and zero are provided. No velocity climatology or persistence field is available in this input export.",
            "Legacy receipts without a weights field are EMA as specified by their producer.",
        ],
    }
    (args.output / "COMPLETE.json").write_text(
        json.dumps(notes, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
