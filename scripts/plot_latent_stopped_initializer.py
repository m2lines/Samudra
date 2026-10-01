# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Interior initialization maps with prior-month observational context and climatology."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from samudra.experiments.diffusion_latent_maps import grid
from samudra.experiments.observation_pilot import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("short-runs", "stopped-runs", "references", "climatology", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    climate = np.load(a.climatology)["ts"]
    results: dict = dict(
        scope="Last initialized five-day state compared with preceding December monthly IAP analysis; context agreement, not instantaneous truth. Temperature surface excluded because supplied. Common wet finite support; no polar exclusion.",
        sources={},
        origins={},
    )
    results["sources"][str(a.climatology)] = digest(a.climatology)
    for year in (2015, 2018, 2021):
        origin = f"{year}-01"
        rp = a.references / f"{year - 1}-12.npz"
        reference = np.load(rp)["values"]
        depths = np.load(rp)["depth"]
        results["sources"][str(rp)] = digest(rp)
        for seed in (1729, 1730):
            paths = [
                a.short_runs
                / f"D-{seed}"
                / "pretraining-maps-v1/members"
                / f"{origin}.npz",
                a.short_runs / f"D-{seed}" / "report-v1/members" / f"{origin}.npz",
                a.stopped_runs / f"D-{seed}" / "report-v1/members" / f"{origin}.npz",
            ]
            arrays = [dict(np.load(path)) for path in paths]
            for path in paths:
                results["sources"][str(path)] = digest(path)
            mask, lat = arrays[0]["mask"].astype(bool), arrays[0]["lat"]
            for z in arrays[1:]:
                for key in ("channel_names", "lat", "mask"):
                    np.testing.assert_array_equal(arrays[0][key], z[key])
            predictions = [z["initial"][:, 0, -1] for z in arrays]
            record = {}
            for channel, group, depth, unit in [
                ("thetao_9", 0, 9, "°C"),
                ("so_0", 1, 0, "Practical salinity"),
                ("so_9", 1, 9, "Practical salinity"),
            ]:
                ci = arrays[0]["channel_names"].tolist().index(channel)
                fields = (
                    [reference[group, depth], climate[group, depth]]
                    + [v[:, ci].mean(0) for v in predictions]
                    + [predictions[-1][0, ci]]
                )
                titles = [
                    "IAP preceding-December context",
                    "Training December climatology",
                    "OM4 pretraining: mean of 8",
                    "Short adaptation: mean of 8",
                    "Stopped adaptation: mean of 8",
                    "Stopped adaptation: member 1",
                ]
                grid(
                    fields,
                    titles,
                    mask[ci],
                    lat,
                    a.output / f"initializer-{channel}-{seed}-{origin}.png",
                    f"{origin}, {channel}, seed {seed}. Monthly context is not instantaneous initialization truth.",
                    unit,
                    show_loss_boundary=False,
                    distinguish_missing=True,
                )
                support = mask[ci] & np.logical_and.reduce(
                    [np.isfinite(f) for f in fields]
                )
                weights = np.cos(np.deg2rad(lat))[:, None] * support
                record[channel] = {
                    name: float(
                        np.sqrt(
                            np.sum(np.where(support, f - fields[0], 0) ** 2 * weights)
                            / weights.sum()
                        )
                    )
                    for name, f in zip(titles[1:], fields[1:], strict=True)
                }
            profiles = {}
            fig, axes = plt.subplots(1, 2, figsize=(9, 6), layout="constrained")
            for group, prefix in enumerate(("thetao", "so")):
                curves: dict[str, list[float]] = {name: [] for name in titles[1:]}
                used_depths = []
                for depth in range(len(depths)):
                    if group == 0 and depth == 0:
                        continue  # Surface temperature is supplied, not inferred.
                    ci = arrays[0]["channel_names"].tolist().index(f"{prefix}_{depth}")
                    fields = (
                        [reference[group, depth], climate[group, depth]]
                        + [v[:, ci].mean(0) for v in predictions]
                        + [predictions[-1][0, ci]]
                    )
                    support = mask[ci] & np.logical_and.reduce(
                        [np.isfinite(f) for f in fields]
                    )
                    weights = np.cos(np.deg2rad(lat))[:, None] * support
                    for name, field in zip(titles[1:], fields[1:], strict=True):
                        curves[name].append(
                            float(
                                np.sqrt(
                                    np.sum(
                                        np.where(support, field - fields[0], 0) ** 2
                                        * weights
                                    )
                                    / weights.sum()
                                )
                            )
                        )
                    used_depths.append(float(depths[depth]))
                for (name, values), marker in zip(
                    curves.items(), ("s", "^", "D", "o", "v"), strict=True
                ):
                    axes[group].plot(values, used_depths, marker=marker, label=name)
                axes[group].invert_yaxis()
                axes[group].set(
                    title=prefix,
                    xlabel="Context RMSE (°C)"
                    if group == 0
                    else "Context RMSE (salinity)",
                    ylabel="Depth (m)",
                )
                axes[group].grid(alpha=0.2)
                profiles[prefix] = dict(depth_m=used_depths, rmse=curves)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
            fig.suptitle(
                f"{origin}, seed{seed}: initial interior versus preceding December IAP\nMonthly context, not instantaneous truth; supplied SST excluded"
            )
            fig.savefig(a.output / f"initializer-profiles-{seed}-{origin}.png", dpi=130)
            plt.close(fig)
            results["origins"][f"{seed}-{origin}"] = dict(
                maps=record, profiles=profiles
            )
    (a.output / "initializer-context.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
