# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Interior initialization maps with prior-month observational context and climatology."""

import argparse
import json
from pathlib import Path

import numpy as np

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
            results["origins"][f"{seed}-{origin}"] = record
    (a.output / "initializer-context.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
