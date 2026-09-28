# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate spectrum of the ensemble mean from spectra of individual readouts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_latent_diagnostics import ORIGINS
from samudra.experiments.observation_metrics import spatial_error
from samudra.metrics.spectra import SPATIAL_REGIONS, log10_rmse_between_curves


def compare(predictions, reference, lat, lon, region):
    """Member,time,y,x inputs; average powers, never fields, for member summary."""
    mean = spatial_error(predictions.mean(0), reference, lat, lon, region)
    if mean is None:
        return None
    members = [spatial_error(p, reference, lat, lon, region) for p in predictions]
    if any(m is None for m in members):
        raise ValueError("Member spectral support differs from the ensemble mean")
    for member in members:
        np.testing.assert_array_equal(member["k_rad_km"], mean["k_rad_km"])
    powers = np.asarray([m["prediction_power"] for m in members])
    k = np.asarray(mean["k_rad_km"])
    averaged = powers.mean(0)
    return dict(
        ensemble_mean=mean,
        individual_members=members,
        mean_member_power=averaged.tolist(),
        member_power_min=powers.min(0).tolist(),
        member_power_max=powers.max(0).tolist(),
        mean_member_error_dex=float(np.mean([m["error_dex"] for m in members])),
        mean_member_power_error_dex=float(
            log10_rmse_between_curves(
                k, np.asarray(mean["reference_power"]), k, averaged
            )
        ),
    )


def summarize(root):
    predictions: dict[str, list[np.ndarray]] = {}
    references: dict[str, list[np.ndarray]] = {}
    lat = lon = None
    for origin in ORIGINS:
        with np.load(root / "members" / f"{origin}.npz") as z:
            if lat is not None and lon is not None:
                np.testing.assert_array_equal(z["lat"], lat)
                np.testing.assert_array_equal(z["lon"], lon)
            lat, lon = z["lat"], z["lon"]
            names = z["channel_names"].tolist()
            domain = z["mask"].astype(bool) & (np.abs(lat)[None, :, None] <= 60)
            states, observed = z["states_at_leads"], z["observed_surface"]
            np.testing.assert_array_equal(z["leads_days"], [5, 15, 30])
            for j, (day, target_step) in enumerate(((5, 0), (15, 2), (30, 5))):
                for c, channel in enumerate(("thetao_0", "zos")):
                    index = names.index(channel)
                    key = f"{channel}/day{day}"
                    predictions.setdefault(key, []).append(
                        np.where(domain[index], states[:, 0, j, index], np.nan)
                    )
                    references.setdefault(key, []).append(
                        np.where(domain[index], observed[0, target_step, c], np.nan)
                    )
            for channel in ("thetao_9", "so_9"):
                index = names.index(channel)
                target_index = z["interior_channel_indices"].tolist().index(index)
                key = channel + "/monthly"
                predictions.setdefault(key, []).append(
                    np.where(domain[index], z["monthly_states"][:, 0, index], np.nan)
                )
                references.setdefault(key, []).append(
                    np.where(
                        domain[index],
                        z["observed_monthly_interior"][0, target_index],
                        np.nan,
                    )
                )
    result = {}
    unavailable = []
    for key in predictions:
        for region in SPATIAL_REGIONS:
            name = key + "/" + region[0]
            value = compare(
                np.stack(predictions[key], axis=1),
                np.stack(references[key]),
                lat,
                lon,
                region,
            )
            if value is None:
                unavailable.append(name)
            else:
                result[name] = value
    return dict(
        scope="Three fixed map origins only; descriptive diagnostics, not the 96-origin selection score",
        origins=list(ORIGINS),
        caution="Power of the ensemble mean, average member power, and average member spectral error are different quantities. Monthly latent members average independent readouts. Spectral agreement alone does not establish forecast accuracy or calibrated uncertainty.",
        curves=result,
        unavailable=unavailable,
    )


def plot(result, path):
    fields = ("thetao_0/day15", "zos/day15", "thetao_9/monthly", "so_9/monthly")
    fig, axes = plt.subplots(4, len(SPATIAL_REGIONS), figsize=(13, 13), squeeze=False)
    for row, field in enumerate(fields):
        for column, region in enumerate(SPATIAL_REGIONS):
            ax = axes[row, column]
            curve = result["curves"].get(field + "/" + region[0])
            if curve is None:
                ax.text(
                    0.5,
                    0.5,
                    "Unavailable on common support",
                    ha="center",
                    transform=ax.transAxes,
                )
                continue
            mean = curve["ensemble_mean"]
            k = mean["k_rad_km"]
            for index, member in enumerate(curve["individual_members"]):
                ax.loglog(
                    k,
                    member["prediction_power"],
                    color="#aaaaaa",
                    alpha=0.6,
                    label="Individual members" if index == 0 else None,
                )
            ax.loglog(
                k,
                mean["reference_power"],
                color="black",
                marker="^",
                label="Observations",
            )
            ax.loglog(
                k,
                mean["prediction_power"],
                color="#286cb0",
                marker="o",
                label="Spectrum of ensemble mean",
            )
            ax.loglog(
                k,
                curve["mean_member_power"],
                color="#c84638",
                marker="s",
                label="Mean member spectrum",
            )
            ax.set_title(field + " — " + region[0], fontsize=9)
            ax.set_xlabel("Wavenumber (rad/km)")
            ax.set_ylabel("Power (physical field units² × m²)")
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(
        "Member versus ensemble-mean spatial spectra: three fixed example dates\nMonthly latent fields average independent readouts; not a full-cohort selection score"
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    save_png(fig, path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.maps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
