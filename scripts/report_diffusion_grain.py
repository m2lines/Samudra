#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize frozen-latent refinement and paired noise response."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from report_correlated_noise_pilot import (  # type: ignore[import-not-found]
    paired_map,
    patch,
    roughness,
    spectrum,
)

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_calibration import ensemble_field_statistics
from samudra.experiments.observation_pilot import digest

UNITS = {
    "thetao_0": "SST (°C)",
    "zos": "SSH (m)",
    "thetao_9": "T at 550 m (°C)",
    "so_9": "S at 550 m",
}


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectra(spectra, output):
    for day in [30, 365]:
        fig, axes = plt.subplots(3, 4, figsize=(18, 12), layout="constrained")
        for row, (mode, task) in enumerate(
            [("pretrained", "om4"), ("adapted", "om4"), ("adapted", "obs")]
        ):
            for col, ch in enumerate(["thetao_0", "zos", "thetao_9", "so_9"]):
                ax = axes[row, col]
                subset = [
                    r
                    for r in spectra
                    if r["mode"] == mode
                    and r["case"].startswith(task)
                    and r["day"] == day
                    and r["channel"] == ch
                ]
                if not subset:
                    ax.set_visible(False)
                    continue
                k = np.arange(1, 17) / 32
                ref = np.array(
                    [r["reference_power"] for r in subset if r["steps"] == 16],
                    dtype=float,
                ).mean(0)
                if np.isfinite(ref).all():
                    ax.loglog(
                        k,
                        ref,
                        color="black",
                        marker="x",
                        markersize=3,
                        label="reference",
                    )
                for steps, color, marker in [
                    (16, "#0072b2", "o"),
                    (32, "#e69f00", "s"),
                    (64, "#009e73", "^"),
                    (128, "#cc79a7", "D"),
                ]:
                    values = [r for r in subset if r["steps"] == steps]
                    for key, style, label in [
                        ("member_power", "-", "member"),
                        ("mean_power", ":", "mean"),
                    ]:
                        power = np.array([r[key] for r in values]).mean(0)
                        ax.loglog(
                            k,
                            power,
                            color=color,
                            linestyle=style,
                            marker=marker if key == "member_power" else "x",
                            markersize=3,
                            label=f"{steps} {label}",
                        )
                ax.set(
                    title=f"{mode} / {task} / {ch}",
                    xlabel="cycles/grid cell",
                    ylabel="window-normalized power",
                )
                ax.legend(fontsize=5)
                ax.grid(alpha=0.2)
        save_png(fig, output / f"sampler-spectra-day{day}.png", dpi=110)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    noise_receipt = json.loads((a.root / "initial-noise.json").read_text())
    if digest(a.root / "initial-noise.npz") != noise_receipt["sha256"]:
        raise ValueError("Initial noise receipt mismatch")
    initial_noise = np.load(a.root / "initial-noise.npz")["noise"]
    spectra = []
    metrics = []
    response = []
    sources = {}
    for mode in ["pretrained", "adapted"]:
        root = a.root / mode
        if not (root / "COMPLETE.json").exists():
            raise ValueError("Incomplete diagnostic run")
        for case in sorted(root.glob("*")):
            if not case.is_dir() or not (case / "COMPLETE.json").exists():
                continue
            receipt = json.loads((case / "COMPLETE.json").read_text())
            for name, expected in receipt["files"].items():
                # Latent caches remain remote; all analyzed arrays are verified here.
                if name.endswith(".pt"):
                    continue
                if digest(case / name) != expected:
                    raise ValueError(f"Bad transfer: {case / name}")
                sources[str((case / name).relative_to(a.root))] = expected
            for file in sorted(case.glob("day*.npz")):
                z = dict(np.load(file))
                lead = int(file.stem.split("-")[0][3:])
                steps = int(file.stem.split("steps")[1])
                for c, channel in enumerate(z["channels"]):
                    members = z["members"][:, c]
                    target = z["reference"][c]
                    mask = z["mask"][c].astype(bool)
                    finite = mask & np.isfinite(target) & np.isfinite(members).all(0)
                    y, x = patch(mask, z["lat"], z["lon"])
                    member_power = spectrum(members, y, x).mean(0)
                    mean_power = spectrum(members.mean(0), y, x)
                    spectra.append(
                        dict(
                            mode=mode,
                            case=case.name,
                            day=lead,
                            steps=steps,
                            channel=str(channel),
                            member_power=member_power.tolist(),
                            mean_power=mean_power.tolist(),
                            reference_power=[
                                float(v) if np.isfinite(v) else None
                                for v in spectrum(target, y, x)
                            ],
                        )
                    )
                    for domain in ["global", "60"]:
                        scored = finite.copy()
                        if domain == "60":
                            scored &= abs(z["lat"][:, None]) <= 60
                        if not scored.any():
                            continue
                        area = np.cos(np.deg2rad(z["lat"]))[:, None] * np.ones((1, 360))
                        stats = ensemble_field_statistics(
                            torch.from_numpy(members[:, None]),
                            torch.from_numpy(target[None]),
                            torch.from_numpy(scored[None]),
                            torch.from_numpy(area),
                            1,
                        )
                        weight = float(stats["weight"][0])
                        rmse = np.sqrt(float(stats["mean_squared_error"][0]) / weight)
                        spread = np.sqrt(float(stats["ensemble_variance"][0]) / weight)
                        ec = members - members.mean(0)
                        nc = initial_noise[:, c] - initial_noise[:, c].mean(0)
                        cross = ep = npower = 0.0
                        for axis in [-1, -2]:
                            support = scored & np.roll(scored, -1, axis=axis)
                            if axis == -2:
                                support[-1] = False
                            weight_pair = area * support
                            de = ec - np.roll(ec, -1, axis=axis)
                            dn = nc - np.roll(nc, -1, axis=axis)
                            cross += (de * dn * weight_pair).sum()
                            ep += (de**2 * weight_pair).sum()
                            npower += (dn**2 * weight_pair).sum()
                        metrics.append(
                            dict(
                                mode=mode,
                                case=case.name,
                                day=lead,
                                steps=steps,
                                channel=str(channel),
                                domain=domain,
                                rmse=rmse,
                                initial_noise_gradient_correlation=cross
                                / np.sqrt(ep * npower),
                                initial_noise_gradient_gain=cross / npower,
                                crps=float(stats["fair_crps"][0]) / weight,
                                spread=spread,
                                spread_rmse=spread / rmse,
                                member_roughness=np.sqrt(
                                    np.mean(
                                        [
                                            roughness(m, scored, area) ** 2
                                            for m in members
                                        ]
                                    )
                                ),
                                reference_roughness=roughness(target, scored, area),
                                member_high_power=member_power[-4:].sum(),
                                mean_high_power=mean_power[-4:].sum(),
                            )
                        )
            for day in [30, 365]:
                if "2018" not in case.name:
                    continue
                z = dict(np.load(case / f"day{day}-steps16.npz"))
                b = dict(np.load(case / f"day{day}-steps128.npz"))
                for c, ch in enumerate(z["channels"]):
                    if not np.isfinite(z["reference"][c]).any():
                        continue
                    fields = [
                        z["reference"][c],
                        z["members"][:, c].mean(0),
                        b["members"][:, c].mean(0),
                        z["reference"][c],
                        z["members"][0, c],
                        b["members"][0, c],
                    ]
                    paired_map(
                        fields,
                        [
                            "Reference",
                            "16 steps: mean",
                            "128 steps: mean",
                            "Reference",
                            "16 steps: member 1",
                            "128 steps: member 1",
                        ],
                        z["mask"][c].astype(bool) & np.isfinite(z["reference"][c]),
                        a.output / f"{mode}-{case.name}-day{day}-{ch}.png",
                        f"{mode}; {case.name}; day {day}; same cached latent and initial Gaussian draw",
                        UNITS[str(ch)],
                    )
            noise_file = case / "denoising.npz"
            if not noise_file.exists():
                continue
            z = dict(np.load(noise_file))
            if "2018" in case.name:
                target_fields = np.load(case / "day30-steps16.npz")["reference"]
                for c, ch in enumerate(z["channels"]):
                    target = target_fields[c]
                    fields = []
                    titles = []
                    for sigma in [0.03, 0.1]:
                        i = list(z["sigmas"]).index(sigma)
                        fields.extend(
                            [
                                target,
                                target + sigma * z["noise"][i, 0, c] * z["scale"][c],
                                target + z["errors"][i, 0, c] * z["scale"][c],
                            ]
                        )
                        titles.extend(
                            [
                                "Clean OM4",
                                f"Corrupted, sigma={sigma}",
                                f"Denoised, sigma={sigma}",
                            ]
                        )
                    paired_map(
                        fields,
                        titles,
                        z["mask"][c].astype(bool),
                        a.output / f"denoise-{mode}-{ch}.png",
                        f"{mode}; OM4 2018 day30; one denoiser call, matched conditioning",
                        UNITS[str(ch)],
                    )

            for c, ch in enumerate(z["channels"]):
                mask = z["mask"][c].astype(bool)
                w = np.cos(np.deg2rad(z["lat"]))[:, None] * mask
                for i, sigma in enumerate(z["sigmas"]):
                    e = z["errors"][i, :, c].astype(np.float64)
                    n = z["noise"][i, :, c].astype(np.float64)
                    ec = e - e.mean(0)
                    nc = n - n.mean(0)
                    gain = (ec * nc * w).sum() / (nc**2 * w).sum() / sigma
                    cross = ep = npower = 0.0
                    for axis in [-1, -2]:
                        support = mask & np.roll(mask, -1, axis=axis)
                        if axis == -2:
                            support[-1] = False
                        weight = w * support
                        de = ec - np.roll(ec, -1, axis=axis)
                        dn = nc - np.roll(nc, -1, axis=axis)
                        cross += (de * dn * weight).sum()
                        ep += (de**2 * weight).sum()
                        npower += (dn**2 * weight).sum()
                    response.append(
                        dict(
                            mode=mode,
                            case=case.name,
                            channel=str(ch),
                            sigma=sigma,
                            normalized_rmse=np.sqrt((e**2 * w).sum() / (4 * w.sum())),
                            physical_rmse=np.sqrt((e**2 * w).sum() / (4 * w.sum()))
                            * z["scale"][c],
                            noise_gain=gain,
                            gradient_noise_gain=cross / npower / sigma,
                            gradient_noise_correlation=cross / np.sqrt(ep * npower),
                        )
                    )
    write_csv(a.output / "sampler.csv", metrics)
    write_csv(a.output / "denoising.csv", response)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, ch in zip(axes.flat, ["thetao_0", "zos", "thetao_9", "so_9"], strict=True):
        for mode, marker in [("pretrained", "o"), ("adapted", "s")]:
            rows = [r for r in response if r["mode"] == mode and r["channel"] == ch]
            sigmas = sorted({r["sigma"] for r in rows})
            gain = [
                np.mean([r["gradient_noise_gain"] for r in rows if r["sigma"] == s])
                for s in sigmas
            ]
            ax.semilogx(sigmas, gain, marker=marker, label=mode)
        ax.axhline(1, color="gray", linestyle=":", label="unchanged injected noise")
        ax.set(
            title=ch,
            xlabel="Noise sigma (normalized units)",
            ylabel="Noise-to-error gain, spatial differences",
        )
        ax.legend()
        ax.grid(alpha=0.2)
    save_png(fig, a.output / "denoiser-noise-gain.png", dpi=130)
    plt.close(fig)
    plot_spectra(spectra, a.output)
    (a.output / "spectra.json").write_text(
        json.dumps(spectra, indent=2, allow_nan=False) + "\n"
    )
    (a.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


if __name__ == "__main__":
    main()
