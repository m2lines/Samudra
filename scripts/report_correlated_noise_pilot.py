# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired physical metrics, member spectra and maps for the short noise pilot."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_calibration import ensemble_field_statistics
from samudra.experiments.diffusion_latent_maps import grid
from samudra.experiments.observation_pilot import digest


def roughness(field, mask, weights):
    parts = []
    for axis in (-1, -2):
        pairs = mask & np.roll(mask, -1, axis=axis)
        if axis == -2:
            pairs[-1] = False
        w = weights * pairs
        parts.append(
            np.sum(np.where(pairs, field - np.roll(field, -1, axis=axis), 0) ** 2 * w)
            / w.sum()
        )
    return float(np.sqrt(np.mean(parts)))


def patch(mask, lat, lon):
    # Choose one shared 32x32 entirely wet Pacific patch, close to 0N,230E.
    candidates = []
    for y in range(len(lat) - 31):
        if abs(lat[y : y + 32]).max() > 60:
            continue
        for x in np.flatnonzero((lon >= 180) & (lon <= 240)):
            if x + 32 > len(lon) or not mask[y : y + 32, x : x + 32].all():
                continue
            distance = lat[y : y + 32].mean() ** 2 + (lon[x : x + 32].mean() - 230) ** 2
            candidates.append((distance, y, int(x)))
    if not candidates:
        raise ValueError("No common fully wet Pacific spectral patch")
    _, y, x = min(candidates)
    return y, x


def spectrum(values, y, x):
    a = values[..., y : y + 32, x : x + 32].astype(np.float64)
    a = a - a.mean((-2, -1), keepdims=True)
    window = np.hanning(32)[:, None] * np.hanning(32)[None, :]
    power = abs(np.fft.fft2(a * window)) ** 2 / (window**2).sum()
    f = np.fft.fftfreq(32) * 32
    radius = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    return np.stack(
        [
            power[..., (radius >= i - 0.5) & (radius < i + 0.5)].mean(-1)
            for i in range(1, 17)
        ],
        -1,
    )


def paired_map(fields, titles, mask, path, caption, unit):
    values = np.concatenate([f[mask] for f in fields])
    low, high = np.quantile(values, [0.01, 0.99])
    fig = plt.figure(figsize=(23, 10), dpi=100)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d1d1d1")
    for i, (field, title) in enumerate(zip(fields, titles, strict=True)):
        ax = fig.add_axes(
            (
                (30 + (i % 3) * 760) / 2300,
                (565 - (i // 3) * 400) / 1000,
                720 / 2300,
                360 / 1000,
            )
        )
        im = ax.imshow(
            np.ma.masked_where(~mask, field),
            origin="lower",
            interpolation="nearest",
            vmin=low,
            vmax=high,
            cmap=cmap,
            aspect="equal",
        )
        np.testing.assert_allclose(
            [ax.get_window_extent().width, ax.get_window_extent().height], [720, 360]
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    cax = fig.add_axes((0.3, 0.09, 0.4, 0.012))
    fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        extend="both",
        label=unit + "; shared 1–99% limits, clipped tails",
    )
    fig.text(0.5, 0.025, caption, ha="center")
    save_png(fig, path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, dict[str, Any]] = {}
    contracts = {}
    sources = {}
    for arm in ("white", "correlated"):
        root = a.root / arm
        complete = json.loads((root / "PILOT_COMPLETE.json").read_text())
        protocol = json.loads((root / "protocol.json").read_text())
        if (
            complete["signature"] != protocol
            or complete["state"]["step"] != 2000
            or not complete["state"]["complete"]
        ):
            raise ValueError("Incomplete or mismatched pilot")
        contracts[arm] = complete
        exports = root / "evaluation/final"
        manifest = json.loads((exports / "COMPLETE.json").read_text())
        if len(manifest["files"]) != 3:
            raise ValueError("Incomplete evaluation")
        datasets[arm] = {}
        for name, sha in manifest["files"].items():
            path = exports / name
            if digest(path) != sha:
                raise ValueError("Member checksum differs")
            sources[str(path)] = sha
            with np.load(path) as z:
                datasets[arm][name[:4]] = dict(z)
    left = contracts["white"]["signature"]
    right = contracts["correlated"]["signature"]
    if left["noise_correlation"] != 0 or right["noise_correlation"] != 0.5:
        raise ValueError("Wrong noise arms")
    if {k: v for k, v in left.items() if k != "noise_correlation"} != {
        k: v for k, v in right.items() if k != "noise_correlation"
    }:
        raise ValueError("Unmatched training contracts")
    rows = []
    spectra: dict[str, Any] = {}
    patch_info = {}
    for year in ("2015", "2018", "2021"):
        white = datasets["white"][year]
        color = datasets["correlated"][year]
        for key in (
            "reference",
            "mask",
            "lat",
            "lon",
            "channels",
            "origin",
            "target_midpoint",
            "lead_days",
        ):
            np.testing.assert_array_equal(white[key], color[key])
        if (
            white["members"].shape != (8, 4, 180, 360)
            or color["members"].shape != white["members"].shape
        ):
            raise ValueError("Unexpected field layout")
        lat, lon = white["lat"], white["lon"]
        area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))
        for c, channel in enumerate(white["channels"]):
            name = str(channel)
            mask = white["mask"][c].astype(bool)
            scored = mask & (abs(lat[:, None]) <= 60)
            target = white["reference"][c]
            y, x = patch(mask, lat, lon)
            patch_info[name] = dict(
                y=y,
                x=x,
                lat=[float(lat[y]), float(lat[y + 31])],
                lon=[float(lon[x]), float(lon[x + 31])],
            )
            spectra.setdefault(name, {"reference": [], "white": [], "correlated": []})[
                "reference"
            ].append(spectrum(target, y, x))
            for arm, z in [("white", white), ("correlated", color)]:
                members = z["members"][:, c]
                mean = members.mean(0)
                stats = ensemble_field_statistics(
                    torch.from_numpy(members[:, None]),
                    torch.from_numpy(target[None]),
                    torch.from_numpy(scored[None]),
                    torch.from_numpy(area),
                    1,
                )
                weight = float(stats["weight"][0])
                mse = float(stats["mean_squared_error"][0]) / weight
                var = float(stats["ensemble_variance"][0]) / weight
                w = area * scored
                coverage = float(
                    np.sum(
                        ((target >= members.min(0)) & (target <= members.max(0))) * w
                    )
                    / w.sum()
                )
                rows.append(
                    dict(
                        arm=arm,
                        year=year,
                        channel=name,
                        mean_rmse=np.sqrt(mse),
                        fair_crps=float(stats["fair_crps"][0]) / weight,
                        spread=np.sqrt(var),
                        spread_rmse=np.sqrt(var / mse),
                        member_range_coverage=coverage,
                        reference_neighbor_rms=roughness(target, scored, area),
                        mean_neighbor_rms=roughness(mean, scored, area),
                        member_neighbor_rms=np.sqrt(
                            np.mean([roughness(m, scored, area) ** 2 for m in members])
                        ),
                    )
                )
                spectra[name][arm].append(
                    dict(mean=spectrum(mean, y, x), members=spectrum(members, y, x))
                )
                grid(
                    [target, mean, *members[:4]],
                    [
                        "OM4 reference",
                        "Mean (8)",
                        *[f"Member {i}" for i in range(1, 5)],
                    ],
                    mask,
                    lat,
                    a.output / f"{arm}-{year}-{name}.png",
                    f"{arm}; seed1729; 2000 OM4-only updates; {year}; day30; {name}. Independent readouts.",
                    name,
                    show_loss_boundary=False,
                    distinguish_missing=True,
                )
            unit = {
                "thetao_0": "SST (°C)",
                "zos": "SSH (m)",
                "thetao_9": "T at 550 m (°C)",
                "so_9": "S at 550 m",
            }[name]
            paired_map(
                [
                    target,
                    white["members"][:, c].mean(0),
                    color["members"][:, c].mean(0),
                    target,
                    white["members"][0, c],
                    color["members"][0, c],
                ],
                [
                    "OM4 reference",
                    "White: mean (8)",
                    "Correlated: mean (8)",
                    "OM4 reference (repeated)",
                    "White: member 1",
                    "Correlated: member 1",
                ],
                mask,
                a.output / f"paired-{year}-{name}.png",
                f"{year}; day30; seed1729; matched scratch 2000-update OM4 models. Noise covariance alone changes.",
                unit,
            )
    with (a.output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    serial = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    k = np.arange(1, 17) / 32
    for ax, (name, values) in zip(axes.flat, spectra.items(), strict=True):
        truth = np.mean(values["reference"], axis=0)
        ax.loglog(k, truth, "ks-", label="OM4 reference", markersize=4)
        serial[name] = dict(
            k_cycles_per_cell=k.tolist(),
            reference=truth.tolist(),
            patch=patch_info[name],
        )
        for arm, color, marker in [
            ("white", "tab:blue", "^"),
            ("correlated", "tab:orange", "D"),
        ]:
            means = np.mean([r["mean"] for r in values[arm]], axis=0)
            members = np.mean([r["members"] for r in values[arm]], axis=0)
            ax.loglog(
                k,
                means,
                color=color,
                marker="o",
                markersize=3,
                label=arm + " ensemble mean",
            )
            ax.loglog(
                k,
                members.mean(0),
                color=color,
                linestyle="--",
                marker=marker,
                markersize=4,
                label=arm + " mean member power",
            )
            ax.fill_between(k, members.min(0), members.max(0), color=color, alpha=0.12)
            serial[name][arm] = dict(
                ensemble_mean=means.tolist(), individual_member_power=members.tolist()
            )
        ax.set_title(name)
        ax.set_xlabel("Spatial frequency (cycles/grid cell)")
        ax.set_ylabel("Window-normalized power")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    save_png(fig, a.output / "member-spectra.png", dpi=130)
    plt.close(fig)
    (a.output / "spectra.json").write_text(json.dumps(serial, indent=2) + "\n")
    (a.output / "provenance.json").write_text(
        json.dumps(
            dict(
                contracts=contracts,
                sources=sources,
                spectral_method="Same fully wet 32x32 Pacific patch in both arms; remove patch mean, apply 2D Hann, radially average power; average powers across three dates. Cycles/grid cell, not isotropic km. Includes grid-scale frequencies; window leakage remains.",
                scope="Three dates, one seed, fixed 2000 updates; common finite wet support within ±60 for metrics; global maps",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
