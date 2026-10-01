# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-date physical maps for the latent comparison, at two pixels per grid cell."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_latent_diagnostics import ORIGINS


def grid(
    fields,
    titles,
    mask,
    lat,
    path,
    caption,
    unit,
    *,
    show_loss_boundary=True,
    distinguish_missing=False,
):
    """Six panels, shared physical color limits, no image interpolation."""
    if len(fields) != 6 or any(field.shape != (180, 360) for field in fields):
        raise ValueError("Expected six native 180x360 fields")
    wet = np.concatenate([field[mask & np.isfinite(field)] for field in fields])
    low, high = np.quantile(wet, [0.01, 0.99])
    fig = plt.figure(figsize=(16, 14), dpi=100)
    for i, (field, title) in enumerate(zip(fields, titles, strict=True)):
        ax = fig.add_axes((0.03 + i % 2 * 0.5, 0.70 - i // 2 * 0.285, 0.45, 360 / 1400))
        im = ax.imshow(
            np.ma.masked_where(~mask | ~np.isfinite(field), field),
            origin="lower",
            vmin=low,
            vmax=high,
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
        )
        if distinguish_missing:
            rgba = im.to_rgba(np.nan_to_num(field), bytes=True)
            rgba[~np.isfinite(field)] = [255, 255, 255, 255]
            rgba[~mask] = [209, 209, 209, 255]
            im.set_data(rgba)
        if show_loss_boundary:
            for latitude in (-60, 60):
                row = np.interp(latitude, lat, np.arange(180))
                ax.axhline(float(row), color="white", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # A native cell occupies exactly 2x2 image pixels in the saved canvas.
        bounds = ax.get_window_extent()
        np.testing.assert_allclose([bounds.width, bounds.height], [720, 360])
    cax = fig.add_axes((0.25, 0.085, 0.5, 0.012))
    fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        extend="both",
        label=f"{unit}; shared 1–99% limits, clipped tails",
    )
    fig.text(
        0.5,
        0.016,
        caption
        + (
            "\nWhite lines: ±60° observation loss/scoring boundary."
            if show_loss_boundary
            else "\nGray: model land; white: missing values on model ocean."
        ),
        ha="center",
        fontsize=10,
    )
    save_png(fig, path, dpi=100)
    plt.close(fig)


def monthly(baseline, physical, latent, output, seed, origin):
    with (
        np.load(baseline / "members" / f"{origin}.npz") as a,
        np.load(physical / "members" / f"{origin}.npz") as b,
        np.load(latent / "members" / f"{origin}.npz") as d,
    ):
        for other in (b, d):
            for key in ("channel_names", "interior_channel_indices", "lat", "mask"):
                np.testing.assert_array_equal(a[key], other[key])
        index = a["channel_names"].tolist().index("so_9")
        target = a["interior_channel_indices"].tolist().index(index)
        fields = [
            a["observed_monthly_interior"][0, target],
            a["monthly_states"][0, 0, index],
        ]
        for z in (b, d):
            states = z["monthly_states"][:, 0, index]
            fields.extend([states.mean(0), states[0]])
        titles = [
            "IAP monthly target",
            "Deterministic baseline",
            "Physical diffusion: mean of 8",
            "Physical diffusion: member 1 monthly mean",
            "Latent + diffusion readout: mean of 8",
            "Latent: readout sequence 1 monthly mean",
        ]
        grid(
            fields,
            titles,
            a["mask"][index].astype(bool),
            a["lat"],
            output / f"monthly-so9-{seed}-{origin}.png",
            f"{origin}, 550 m salinity, seed {seed}. Latent samples are independent at each lead.",
            "Practical salinity",
        )


def initialization(pretraining, adapted, output, seed, origin):
    with (
        np.load(pretraining / "members" / f"{origin}.npz") as pre,
        np.load(adapted / "members" / f"{origin}.npz") as post,
    ):
        for key in ("channel_names", "lat", "mask"):
            np.testing.assert_array_equal(pre[key], post[key])
        for channel, unit in (
            ("so_9", "Practical salinity"),
            ("thetao_9", "Temperature (°C)"),
            ("uo_9", "Zonal velocity (m/s)"),
        ):
            index = pre["channel_names"].tolist().index(channel)
            fields, titles = [], []
            values = [z["initial"][:, 0, -1, index] for z in (pre, post)]
            for kind in (None, 0, 1):
                for stage, members in zip(
                    ("OM4 pretrained", "Observation adapted"), values, strict=True
                ):
                    fields.append(members.mean(0) if kind is None else members[kind])
                    label = "mean of 8" if kind is None else f"member {kind + 1}"
                    titles.append(f"{stage}: {label}")
            grid(
                fields,
                titles,
                pre["mask"][index].astype(bool),
                pre["lat"],
                output / f"initial-{channel}-{seed}-{origin}.png",
                f"{origin}, {channel}, seed {seed}. Instantaneous initial readouts from the same observed inputs.",
                unit,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--physical-reports", type=Path, required=True)
    parser.add_argument("--latent-runs", type=Path, required=True)
    parser.add_argument("--pretraining-runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in (1729, 1730):
        run = args.latent_runs / f"D-{seed}"
        for origin in ORIGINS:
            monthly(
                args.baseline,
                args.physical_reports / f"B-{seed}",
                run / "report-v1",
                args.output,
                seed,
                origin,
            )
            initialization(
                (args.pretraining_runs or args.latent_runs)
                / f"D-{seed}"
                / "pretraining-maps-v1",
                run / "report-v1",
                args.output,
                seed,
                origin,
            )


if __name__ == "__main__":
    main()
