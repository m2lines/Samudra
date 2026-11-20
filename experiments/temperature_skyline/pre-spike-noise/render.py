# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render matched native maps and fully wet, windowed short-wave spectra."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from samudra.metrics.spectra import isotropic_spectrum, remove_plane

DEPTHS = {0: 2.5, 7: 250, 13: 1850}
MODELS = ("gold", "samudra", "temperature25k", "fields21k", "fields25k")
LABELS = (
    "OM4",
    "Samudra",
    "T only · EMA 25k",
    "All fields · EMA 21k",
    "All fields · EMA 25k",
)
COLORS = ("black", "#cc79a7", "#0072b2", "#e69f00", "#d55e00")


def wet_patch(mask):
    """Largest 64/32-square fully wet tile, nearest the region center."""
    h, w = mask.shape
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for side in (64, 32):
        if side > min(h, w):
            continue
        counts = (
            integral[side:, side:]
            - integral[:-side, side:]
            - integral[side:, :-side]
            + integral[:-side, :-side]
        )
        y, x = np.where(counts == side * side)
        if len(y):
            order = np.lexsort(
                (x, y, (y + side / 2 - h / 2) ** 2 + (x + side / 2 - w / 2) ** 2)
            )
            return int(y[order[0]]), int(x[order[0]]), side
    return None


def short_wave_power(field, include_corners=False):
    """Variance in radial k=0.25..0.5 cycles/cell, with planar detrend/Hann."""
    h, w = field.shape
    window = np.outer(
        0.5 - 0.5 * np.cos(2 * np.pi * np.arange(h) / h),
        0.5 - 0.5 * np.cos(2 * np.pi * np.arange(w) / w),
    )
    power = np.abs(
        np.fft.fft2(remove_plane(field.astype(float)) * window, norm="forward")
    ) ** 2 / np.mean(window**2)
    ky, kx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    k = np.sqrt(kx * kx + ky * ky)
    selected = (k >= 0.25) & ((k <= 0.5) | include_corners)
    return float(power[selected].sum())


def load(root):
    patches = []
    archive = np.load(root) if root.is_file() else None
    for region in range(3):
        for depth in DEPTHS:
            files = {}
            for name in MODELS[2:]:
                prefix = f"{name}-r{region}-z{depth:02d}"
                files[name] = (
                    {
                        k.split("/", 1)[1]: archive[k]
                        for k in archive.files
                        if k.startswith(prefix + "/")
                    }
                    if archive is not None
                    else np.load(root / f"{prefix}.npz")
                )
            a = files["fields21k"]
            for b in files.values():
                for key in ("gold", "samudra", "wet", "lat", "lon", "area"):
                    np.testing.assert_array_equal(a[key], b[key])
            data = {name: a[name] for name in MODELS[:2]}
            data.update({name: files[name]["prediction"] for name in MODELS[2:]})
            patches.append(
                dict(
                    region=str(a["region"]),
                    depth=depth,
                    wet=a["wet"],
                    lat=a["lat"],
                    lon=a["lon"],
                    area=a["area"],
                    data=data,
                    tile=wet_patch(a["wet"]),
                )
            )
    return patches


def analyze(patches, output):
    metrics: list[dict] = []
    spectral: list[dict] = []
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
    errfig, erraxes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
    for i, p in enumerate(patches):
        mask, data, tile = p["wet"], p["data"], p["tile"]
        weight = p["area"][mask] / p["area"][mask].sum()
        patch_data = None
        if tile:
            y, x, side = tile
            patch_data = {
                name: data[name][y : y + side, x : x + side] for name in MODELS
            }
            assert all(np.isfinite(v).all() for v in patch_data.values())
        for name in MODELS:
            error = data[name][mask] - data["gold"][mask]
            row = dict(
                region=p["region"],
                depth_m=DEPTHS[p["depth"]],
                model=name,
                wet_count=int(mask.sum()),
                rmse_c=float(np.sqrt(np.sum(weight * error**2))),
                bias_c=float(np.sum(weight * error)),
                spectrum_side=None,
                hf_field_variance_c2=None,
                hf_error_variance_c2=None,
                hf_field_ratio_om4=None,
                hf_error_ratio_samudra=None,
                hf_field_ratio_om4_including_corners=None,
                hf_error_ratio_samudra_including_corners=None,
            )
            if patch_data:
                row["spectrum_side"] = tile[2]
                row["hf_field_variance_c2"] = short_wave_power(patch_data[name])
                row["hf_error_variance_c2"] = short_wave_power(
                    patch_data[name] - patch_data["gold"]
                )
                row["hf_field_ratio_om4"] = row[
                    "hf_field_variance_c2"
                ] / short_wave_power(patch_data["gold"])
                row["hf_error_ratio_samudra"] = row[
                    "hf_error_variance_c2"
                ] / short_wave_power(patch_data["samudra"] - patch_data["gold"])
                row["hf_field_ratio_om4_including_corners"] = short_wave_power(
                    patch_data[name], True
                ) / short_wave_power(patch_data["gold"], True)
                row["hf_error_ratio_samudra_including_corners"] = short_wave_power(
                    patch_data[name] - patch_data["gold"], True
                ) / short_wave_power(patch_data["samudra"] - patch_data["gold"], True)
            metrics.append(row)
        for ax, errors in ((axes.flat[i], False), (erraxes.flat[i], True)):
            if patch_data:
                for name, label, color in zip(MODELS, LABELS, COLORS, strict=True):
                    if errors and name == "gold":
                        continue
                    values = (
                        patch_data[name] - patch_data["gold"]
                        if errors
                        else patch_data[name]
                    )
                    k, power = isotropic_spectrum(values, num_bins=32)
                    ax.loglog(k, power, label=label, color=color, lw=1.5)
                    if not errors:
                        spectral.extend(
                            dict(
                                region=p["region"],
                                depth_m=DEPTHS[p["depth"]],
                                model=name,
                                k_cycles_per_cell=float(kk),
                                power=float(pp),
                            )
                            for kk, pp in zip(k, power, strict=True)
                        )
                ax.axvspan(0.25, 0.5, color=".8", alpha=0.25)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No fully wet 32×32 tile",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(
                f"{p['region']} · {DEPTHS[p['depth']]:g} m · {tile[2] if tile else 'no'}-cell tile"
            )
            ax.set_xlabel("Wavenumber (cycles/native grid cell)")
            ax.set_ylabel("k × power spectral density (°C²)")
            ax.grid(alpha=0.2)
    for f, ax, filename, title in (
        (fig, axes, "field-spectra.png", "Temperature field spectra"),
        (errfig, erraxes, "error-spectra.png", "Temperature error spectra against OM4"),
    ):
        handles, labels = ax.flat[0].get_legend_handles_labels()
        f.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        f.suptitle(
            title
            + " · same fully wet tile for every model\nPlanar detrend + periodic Hann; shaded band: wavelengths 2–4 grid cells",
            fontsize=12,
        )
        f.savefig(output / filename, dpi=130)
        plt.close(f)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output / "spectra.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(spectral[0]))
        writer.writeheader()
        writer.writerows(spectral)
    (output / "tiles.json").write_text(
        json.dumps(
            [
                dict(region=p["region"], depth_m=DEPTHS[p["depth"]], tile=p["tile"])
                for p in patches
            ],
            indent=2,
        )
        + "\n"
    )
    return metrics


def maps(patches, metrics, output, errors=False):
    names = MODELS[1:] if errors else MODELS
    labels = LABELS[1:] if errors else LABELS
    # Identical sample size: 2 pixels along each axis, including the narrower regions.
    px = 2
    width = 180 + len(names) * 270 + 80
    height = 95 + len(patches) * 205
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    for i, p in enumerate(patches):
        h, w = p["wet"].shape
        top = height - 70 - i * 205
        gold = p["data"]["gold"]
        if errors:
            limit = max(
                float(
                    np.quantile(np.abs(p["data"][n][p["wet"]] - gold[p["wet"]]), 0.99)
                )
                for n in names
            )
            vmin, vmax, cmap = -limit, limit, "RdBu_r"
        else:
            vmin, vmax = float(np.min(gold[p["wet"]])), float(np.max(gold[p["wet"]]))
            cmap = "viridis"
        fig.text(
            15 / width,
            (top - 60) / height,
            f"{p['region']}\n{DEPTHS[p['depth']]:g} m\n¼° native grid",
            fontsize=10,
            va="center",
        )
        for j, (name, label) in enumerate(zip(names, labels, strict=True)):
            ax = fig.add_axes(
                (
                    (180 + j * 270) / width,
                    (top - h * px) / height,
                    w * px / width,
                    h * px / height,
                )
            )
            value = p["data"][name] - gold if errors else p["data"][name]
            im = ax.imshow(
                np.ma.masked_where(~p["wet"], value),
                origin="lower",
                interpolation="none",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ax.set_axis_off()
            if p["tile"]:
                y, x, s = p["tile"]
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        s,
                        s,
                        fill=False,
                        edgecolor="black",
                        linewidth=0.55,
                        linestyle="--",
                    )
                )
            if i == 0:
                ax.text(
                    0.5, 1.13, label, ha="center", transform=ax.transAxes, fontsize=10
                )
            if name != "gold":
                r = next(
                    r
                    for r in metrics
                    if r["region"] == p["region"]
                    and r["depth_m"] == DEPTHS[p["depth"]]
                    and r["model"] == name
                )
                ax.text(
                    0.5,
                    -0.08,
                    f"RMSE {r['rmse_c']:.3f} °C",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=9,
                )
        cax = fig.add_axes(
            ((width - 60) / width, (top - h * px) / height, 10 / width, h * px / height)
        )
        fig.colorbar(im, cax=cax, extend="both").ax.tick_params(labelsize=8)
    fig.text(
        0.5,
        0.995,
        ("Errors against OM4" if errors else "Temperature fields")
        + " · July 21, 2016 (+5 days) · black dashed box: spectral tile",
        ha="center",
        va="top",
        fontsize=12,
    )
    fig.savefig(
        output / ("error-maps.png" if errors else "temperature-maps.png"), dpi=100
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    patches = load(args.source)
    metrics = analyze(patches, args.output)
    maps(patches, metrics, args.output)
    maps(patches, metrics, args.output, errors=True)


if __name__ == "__main__":
    main()
