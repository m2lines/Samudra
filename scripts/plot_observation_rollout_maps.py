#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Extract fixed day-30 examples from saved exports and plot common-support maps."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = ["2022-01", "2022-07"]
METHODS = [
    ("Full fine-tuning", "primary-evaluation", "selected-predictions.npz"),
    ("Observation-only scratch", "scratch-evaluation", "selected-predictions.npz"),
    ("Adapter only", "adapter-evaluation", "selected-predictions.npz"),
    ("Source D / zero adapter", "primary-evaluation", "source-with-zero-forcing.npz"),
    (
        "Full-model initial-state persistence",
        "primary-evaluation",
        "selected-inferred-persistence.npz",
    ),
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(root, data, bundle):
    provenance = {
        "scope": "Saved day-30 five-day mean forecasts; no new inference or training",
        "cases": CASES,
        "case_choice": "January and July of the final held-out year, fixed before inspecting maps",
        "lead_index": 5,
        "lead_bin_end_days": 30,
        "grid_sha256": digest(data / "grid.npz"),
        "script_sha256": digest(Path(__file__)),
        "sources": [],
        "case_times": {},
    }
    with np.load(data / "grid.npz") as grid:
        arrays = {key: grid[key] for key in ("lat", "lon")}
        arrays["mask"] = grid["mask"][0]
    predictions = []
    reference = None
    for label, directory, filename in METHODS + [
        (
            "Training seasonal climatology",
            "primary-evaluation",
            "seasonal-climatology.npz",
        )
    ]:
        folder = root / directory
        complete = json.loads((folder / "COMPLETE.json").read_text())
        audit = json.loads((folder / "anomaly-year-report.json").read_text())
        if complete["origins"] != 96:
            raise ValueError("Incomplete evaluation")
        path = folder / filename
        checksum = digest(path)
        if checksum != audit["files_sha256"][filename]:
            raise ValueError("Export differs from completed evaluation audit")
        with np.load(path) as export:
            origins = export["origins"].tolist()
            indices = [origins.index(case) for case in CASES]
            prediction = export["prediction"][indices, 5, :2]
            truth = export["reference"][indices, 5, :2]
        if reference is None:
            reference = truth
        elif not np.array_equal(reference, truth, equal_nan=True):
            raise ValueError("Mismatched observations")
        if label == "Training seasonal climatology":
            arrays["climatology"] = prediction
        else:
            predictions.append(prediction)
        provenance["sources"].append(
            {
                "method": label,
                "path": str(path),
                "sha256": checksum,
                "selected_checkpoint_sha256": complete["selected_sha256"],
                "source_checkpoint_sha256": complete["source_sha256"],
            }
        )
    for case in CASES:
        with np.load(data / "test" / (case + ".npz")) as sample:
            provenance["case_times"][case] = {
                "last_history_midpoint": str(sample["midpoints"][18]),
                "day30_bin_midpoint": str(sample["midpoints"][24]),
            }
    arrays["reference"] = reference
    arrays["prediction"] = np.array(predictions)
    arrays["provenance"] = np.array(json.dumps(provenance))
    with bundle.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "sha256": digest(bundle),
                "bytes": bundle.stat().st_size,
            }
        )
    )


def plot(bundle, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(bundle) as archive:
        arrays = {key: archive[key] for key in archive.files}
    provenance = json.loads(str(arrays["provenance"]))
    lon = (arrays["lon"] + 180) % 360 - 180
    order = np.argsort(lon)
    lat = arrays["lat"]
    domain = arrays["mask"] & (np.abs(lat[:, None]) <= 60)
    # Identical reference-valid support for every candidate; filled cells are hidden.
    valid = np.isfinite(arrays["reference"]) & domain
    all_fields = np.concatenate([arrays["reference"][None], arrays["prediction"]])
    labels = ["Observations"] + [item[0] for item in METHODS]
    output.mkdir(parents=True, exist_ok=True)
    render = []
    for case_index, case in enumerate(CASES):
        for anomaly in (False, True):
            if not anomaly and case != "2022-07":
                continue
            fields = all_fields[:, case_index].copy()
            if anomaly:
                fields -= arrays["climatology"][case_index]
            limits = [(-3, 3), (-0.25, 0.25)] if anomaly else [(-2, 32), (-1.5, 1.5)]
            for column, (variable, unit) in enumerate(
                (("SST", "°C"), ("ADT / SSH", "m"))
            ):
                fig, axes = plt.subplots(
                    len(labels), 1, figsize=(6, 11), layout="constrained"
                )
                vmin, vmax = limits[column]
                for row, label in enumerate(labels):
                    ax = axes[row]
                    values = np.where(
                        valid[case_index, column], fields[row, column], np.nan
                    )
                    ax.set_facecolor("0.82")
                    im = ax.pcolormesh(
                        lon[order],
                        lat,
                        values[:, order],
                        shading="nearest",
                        cmap="RdBu_r" if anomaly or column == 1 else "turbo",
                        vmin=vmin,
                        vmax=vmax,
                        rasterized=True,
                    )
                    ax.set_xlim(-180, 180)
                    ax.set_ylim(-60, 60)
                    ax.set_xticks([-180, -90, 0, 90, 180])
                    ax.set_yticks([-60, 0, 60])
                    ax.tick_params(labelsize=7)
                    ax.set_title(label, fontsize=9)
                    ax.set_ylabel("Latitude (°)")
                axes[-1].set_xlabel("Longitude (°)")
                fig.colorbar(
                    im,
                    ax=axes,
                    orientation="horizontal",
                    shrink=0.85,
                    pad=0.015,
                    label=f"{variable}{' anomaly' if anomaly else ''} ({unit})",
                    extend="both",
                )
                kind = "anomalies" if anomaly else "fields"
                midpoint = provenance["case_times"][case]["day30_bin_midpoint"][:10]
                fig.suptitle(
                    f"{variable} · {case} origin · day-30 five-day mean\nMidpoint {midpoint} · "
                    + (
                        "training-climatology anomalies"
                        if anomaly
                        else "absolute fields"
                    ),
                    fontsize=10,
                )
                stem = f"day30-{case}-{kind}-{'sst' if column == 0 else 'adt'}"
                fig.savefig(
                    output / (stem + ".jpg"),
                    dpi=100,
                    pil_kwargs={"quality": 75, "optimize": True},
                )
                fig.savefig(output / (stem + ".pdf"), dpi=72)
                plt.close(fig)
                render.append({"file": stem, "limits": [vmin, vmax]})
    for source in provenance["sources"]:
        source["weights_used_sha256"] = (
            None
            if source["method"] == "Training seasonal climatology"
            else source["source_checkpoint_sha256"]
            if source["method"] == "Source D / zero adapter"
            else source["selected_checkpoint_sha256"]
        )
    provenance.update(
        {
            "bundle_sha256": digest(bundle),
            "plot_script_sha256": digest(Path(__file__)),
            "rendered": render,
            "mask": "60S-60N model wet domain intersected with finite target observations, identical across candidates",
            "interpretation": "Illustrative cases, not aggregate skill; gray is land or unavailable observations. Colorbar extensions mark saturation. Endpoint is the last common exported surface lead, not a multi-year continuous rollout.",
        }
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.extract:
        extract(args.root, args.data, args.bundle)
    else:
        plot(args.bundle, args.output)


if __name__ == "__main__":
    main()
