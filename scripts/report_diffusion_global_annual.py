#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched annual maps, error curves and temporal member diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_calibration import ensemble_field_statistics
from samudra.experiments.diffusion_latent_maps import grid
from samudra.experiments.observation_pilot import digest


def increments(values, reference, mask, area):
    """Area/time/member pooled physical increments on pairwise valid support."""
    mean = values.mean(0)
    valid = np.broadcast_to(mask, mean.shape).copy()
    if reference is not None:
        valid &= np.isfinite(reference)
    weights = (valid[1:] & valid[:-1]) * area
    total = weights.sum()
    if total <= 0:
        raise ValueError("Empty annual increment support")

    def moment(array):
        array = np.where(weights > 0, array, 0)
        return float(np.sum(array * weights, dtype=np.float64) / total)

    deviations = values - mean[None]
    covariance = np.mean([moment(member[1:] * member[:-1]) for member in deviations])
    variance1 = np.mean([moment(member[:-1] ** 2) for member in deviations])
    variance2 = np.mean([moment(member[1:] ** 2) for member in deviations])
    result = dict(
        member_increment_rms=float(
            np.sqrt(
                np.mean([moment(np.diff(member, axis=0) ** 2) for member in values])
            )
        ),
        mean_increment_rms=float(np.sqrt(moment(np.diff(mean, axis=0) ** 2))),
        adjacent_member_deviation_correlation=(
            float(covariance / np.sqrt(variance1 * variance2))
            if variance1 * variance2 > 0
            else None
        ),
    )
    if reference is not None:
        result["reference_increment_rms"] = float(
            np.sqrt(moment(np.diff(reference, axis=0) ** 2))
        )
    return result


def main():
    from global_physical_heat_content import (  # type: ignore[import-not-found]
        cell_areas,
        heat_total,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.root / "MEMBERS_COMPLETE.json").read_text())
    for name, expected in receipt["files"].items():
        if Path(name).name != name or digest(args.root / name) != expected:
            raise ValueError("Annual export hash mismatch")
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = ["#0072b2", "#d55e00", "#009e73"]
    for date_index, origin in enumerate(("2015-01-01", "2018-01-01", "2021-01-01")):
        with np.load(args.root / f"{origin}.npz") as saved:
            point = dict(saved)
        with np.load(args.baseline / f"{origin}.npz") as saved:
            baseline = dict(saved)
        np.testing.assert_array_equal(point["reference"], baseline["reference"])
        np.testing.assert_array_equal(point["reference_ohc"], baseline["reference_ohc"])
        diagnostics = json.loads((args.root / f"{origin}.json").read_text())
        control = json.loads((args.baseline / f"{origin}.json").read_text())
        results[origin] = dict(
            diffusion=diagnostics,
            deterministic=control,
            temporal={},
            surface_ensemble={},
        )
        with np.load(args.root / f"members-{origin}.npz") as saved:
            members = saved["fields"]
            mask, lat = saved["mask"].astype(bool), saved["lat"]
            names = saved["channel_names"]
            physical_area = cell_areas(lat, saved["lon"])
        area = np.cos(np.deg2rad(lat))[:, None]
        mean_fig, mean_axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        results[origin]["absolute_series"] = {}
        for variable, ax in zip(
            ("sst", "ssh", "ohc_0_700", "ohc_700_2000"), mean_axes.flat, strict=True
        ):
            surface = variable in ("sst", "ssh")
            channel = 0 if variable in ("sst", "ohc_0_700") else 1
            key = "surface" if surface else "predicted_ohc"
            arrays = dict(
                reference=point["reference" if surface else "reference_ohc"][
                    :, channel
                ],
                diffusion=point[key][:, channel],
                deterministic=baseline[key][:, channel],
            )
            support = np.logical_and.reduce(
                [np.isfinite(v).all(0) for v in arrays.values()]
            )
            if surface:
                support &= mask[channel]
                weight = support * area
                curves = {
                    k: (np.where(support, v, 0) * weight).sum((-2, -1)) / weight.sum()
                    for k, v in arrays.items()
                }
            else:
                curves = {
                    k: heat_total(v, physical_area, support) for k, v in arrays.items()
                }
            result = dict(
                curves={k: v.tolist() for k, v in curves.items()},
                mean_bias={
                    k: float(np.mean(v - curves["reference"]))
                    for k, v in curves.items()
                    if k != "reference"
                },
                cells=int(support.sum()),
                area_m2=float(physical_area[support].sum()),
            )
            results[origin]["absolute_series"][variable] = result
            leads_for_curve = np.arange(1, len(curves["reference"]) + 1) * (
                5 if surface else 1
            )
            for method, marker, color in (
                ("reference", "x", "black"),
                ("deterministic", "s", "#0072b2"),
                ("diffusion", "o", "#d55e00"),
            ):
                ax.plot(
                    leads_for_curve,
                    curves[method],
                    label=method,
                    marker=marker,
                    markersize=3,
                    markevery=6 if surface else 1,
                    color=color,
                )
            unit = ("°C" if variable == "sst" else "m") if surface else "ZJ"
            ax.set(
                title=f"{origin}: {variable}",
                xlabel="Lead (days)" if surface else "Calendar month",
                ylabel=f"{'Area mean' if surface else 'Observed-support total'} ({unit})",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        save_png(
            mean_fig, args.output / f"annual-{origin}-absolute-series.png", dpi=150
        )
        plt.close(mean_fig)
        for channel, name in enumerate(names):
            reference = point["reference"][:, channel] if channel < 2 else None
            for last in (6, 18, 73):
                results[origin]["temporal"][f"{name}/through_day{last * 5}"] = (
                    increments(
                        members[:, :last, channel],
                        None if reference is None else reference[:last],
                        mask[channel],
                        area,
                    )
                )
            if channel >= 2:
                continue
            assert reference is not None
            for day in (30, 365):
                index = day // 5 - 1
                fields = members[:, index, channel]
                statistics = ensemble_field_statistics(
                    torch.from_numpy(fields[:, None]),
                    torch.from_numpy(reference[index, None]),
                    torch.from_numpy(mask[channel, None]),
                    torch.from_numpy(area),
                    1.0,
                )
                weight = float(statistics["weight"].sum())
                scores = {
                    key: float(statistics[key].sum()) / weight
                    for key in (
                        "mean_squared_error",
                        "ensemble_variance",
                        "fair_crps",
                        "empirical_crps",
                    )
                }
                ranks = statistics["rank_weights"].reshape(9, -1).sum(1) / weight
                scores.update(
                    rmse=scores["mean_squared_error"] ** 0.5,
                    spread=scores["ensemble_variance"] ** 0.5,
                    outside_ensemble_fraction=float(ranks[0] + ranks[-1]),
                )
                results[origin]["surface_ensemble"][f"{name}/day{day}"] = scores
                grid(
                    [
                        reference[index],
                        baseline["surface"][index, channel],
                        fields.mean(0),
                        *fields[:3],
                    ],
                    [
                        "Observed five-day reference",
                        "Deterministic baseline",
                        "Eight-member diffusion mean",
                        "Diffusion member 1",
                        "Diffusion member 2",
                        "Diffusion member 3",
                    ],
                    mask[channel],
                    lat,
                    args.output / f"annual-{origin}-day{day}-{name}.png",
                    f"{origin}; day{day}; one initialization; prescribed ERA5",
                    str(name),
                    show_loss_boundary=False,
                    distinguish_missing=True,
                )
        leads = np.array(sorted(map(int, diagnostics["leads"])))
        for ax, field in zip(axes, ("sst", "adt"), strict=True):
            for source, marker, label in (
                (diagnostics, "o", "Diffusion"),
                (control, "x", "Deterministic"),
            ):
                values = [source["leads"][str(day)][f"{field}_rmse"] for day in leads]
                ax.semilogy(
                    leads,
                    values,
                    marker=marker,
                    color=colors[date_index],
                    label=f"{label} {origin[:4]}",
                )
            ax.set(
                xlabel="Forecast lead (days)",
                ylabel=f"{field.upper()} RMSE ({'°C' if field == 'sst' else 'm'}, log scale)",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    save_png(fig, args.output / "annual-error-curves.png", dpi=150)
    plt.close(fig)
    (args.output / "annual-comparison.json").write_text(
        json.dumps(
            dict(
                cases=results,
                protocol=receipt["inputs"],
                verified="Full annual surface/velocity and monthly OHC target arrays identical to deterministic reference",
                temporal_scope="Independent physical readouts along a deterministic latent path. Increment moments pool time and wet area; intervals ending90/365days can be dominated by drift. Instantaneous interior references unavailable.",
                absolute_series_scope="Undetrended SST/SSH means and absolute OHC totals on fixed common finite support within each year. OHC uses midpoint spherical areas, R6371km, binary wet cells, no extrapolation, temperature reference0C. No persistence/climatology curve is added here.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
