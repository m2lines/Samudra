#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed-origin reconstruction maps against the previously saved truth."""

import argparse
import io
import json
from pathlib import Path

import matplotlib
import numpy as np

from samudra.experiments.surface_adaptation_analysis import read_bytes

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "A": "A: 30M U-Net, short",
    "B": "B: 30M U-Net, expanded",
    "C": "C: 121M U-Net, short",
    "D": "D: 121M U-Net, expanded",
    "E": "E: attention, short",
    "F": "F: attention, expanded",
}
LICENSE = "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"


def snapshot(directory, index):
    directory = Path(directory)
    manifest = json.loads(read_bytes(directory / "manifest.json"))
    found = []
    for rank in range(manifest["world_size"]):
        with np.load(
            io.BytesIO(read_bytes(directory / f"snapshots-rank{rank}.npz"))
        ) as data:
            if f"{index}_date" in data:
                found.append({key: data[key] for key in data.files})
    if len(found) != 1:
        raise ValueError("Requested snapshot must occur in exactly one rank file")
    return found[0]


def plot(raw, reference, output, index=294, title="Held-out initialization"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(io.BytesIO(read_bytes(Path(reference)))) as data:
        previous = {key: data[key] for key in data.files}
    snapshots = {arm: snapshot(Path(raw) / arm, index) for arm in LABELS}
    common = snapshots["A"]
    for value in snapshots.values():
        for field in ["latitude", "longitude", "channels", "depths_m", f"{index}_date"]:
            np.testing.assert_array_equal(value[field], common[field])
        for field in ["truth", "climatology"]:
            np.testing.assert_array_equal(
                value[f"{index}_{field}"], common[f"{index}_{field}"]
            )
    for coordinate in ["latitude", "longitude", "depths_m"]:
        np.testing.assert_array_equal(common[coordinate], previous[coordinate])
    date = str(common[f"{index}_date"].item())
    audit = {
        "origin_index": index,
        "origin_time": date,
        "reference": str(reference),
        "channels": {},
    }
    for c, channel in enumerate(common["channels"].tolist()):
        old_truth = previous[f"{index}_truth_{channel}"]
        old_climate = previous[f"{index}_climatology_{channel}"]
        wet = np.isfinite(old_truth)
        truth, climate = common[f"{index}_truth"][c], common[f"{index}_climatology"][c]
        np.testing.assert_allclose(truth[wet], old_truth[wet], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(climate[wet], old_climate[wet], rtol=2e-6, atol=2e-6)
        assert np.all(truth[~wet] == 0) and np.all(climate[~wet] == 0)
        anomaly = old_truth - old_climate
        limit = float(np.nanquantile(np.abs(anomaly), 0.98))
        depth = common["depths_m"][int(channel.split("_")[1])]
        fields = [
            ("Previous joint initializer (wave 2 E)", previous[f"{index}_E_{channel}"])
        ]
        fields += [
            (
                LABELS[arm],
                np.where(wet, snapshots[arm][f"{index}_prediction"][c], np.nan),
            )
            for arm in LABELS
        ]
        for error in [False, True]:
            fig, axes = plt.subplots(
                4, 2, figsize=(10, 11.5), layout="constrained", sharex=True, sharey=True
            )
            panels = [("Fixed true anomaly", anomaly)] + [
                (label, field - (old_truth if error else old_climate))
                for label, field in fields
            ]
            for ax, (label, field) in zip(axes.flat, panels, strict=True):
                im = ax.pcolormesh(
                    common["longitude"],
                    common["latitude"],
                    field,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    shading="auto",
                    rasterized=True,
                )
                ax.set(title=label, xlabel="Longitude", ylabel="Latitude")
                ax.set_facecolor("#dedede")
            mode = (
                "prediction minus truth"
                if error
                else "anomaly from fixed monthly climatology"
            )
            fig.colorbar(
                im,
                ax=axes,
                location="bottom",
                shrink=0.8,
                label="Normalized anomaly / error",
            )
            fig.suptitle(
                f"{title}: {channel}, {depth:g} m, {date}\n{mode}; common limits at true anomaly's 98th absolute percentile",
                fontsize=11,
            )
            path = output / f"maps-{channel}-{'error' if error else 'anomaly'}.jpg"
            for quality in [85, 80, 75, 70, 65]:
                fig.savefig(
                    path,
                    dpi=105,
                    pil_kwargs={"quality": quality, "optimize": True},
                    bbox_inches="tight",
                )
                if path.stat().st_size <= 240 * 1024:
                    break
            else:
                raise ValueError(f"Map exceeds the repository asset limit: {path}")
            plt.close(fig)
            Path(str(path) + ".license").write_text(LICENSE)
        audit["channels"][channel] = {
            "wet_cells": int(wet.sum()),
            "normalized_color_limit": limit,
            "depth_m": float(depth),
        }
    path = output / "map-reference-audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n")
    Path(str(path) + ".license").write_text(LICENSE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument(
        "--reference",
        default="docs/experiments/surface-wave2-results/initialization/raw/snapshots.npz",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--origin-index",
        type=int,
        default=294,
        help="Prespecified middle of the 99 origins; 0 is also saved",
    )
    p.add_argument("--title", default="Held-out initialization")
    args = p.parse_args()
    plot(args.raw, args.reference, args.output, args.origin_index, args.title)


if __name__ == "__main__":
    main()
