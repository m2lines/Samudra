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


def extract(root, data, bundle, methods=None):
    methods = METHODS if methods is None else methods
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
    for label, directory, filename in methods + [
        (
            "Training seasonal climatology",
            methods[0][1],
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
    from PIL import Image

    with np.load(bundle) as archive:
        arrays = {key: archive[key] for key in archive.files}
    provenance = json.loads(str(arrays["provenance"]))
    lon = (arrays["lon"] + 180) % 360 - 180
    order = np.argsort(lon)
    lat = arrays["lat"]
    latitude_rows = np.flatnonzero(np.abs(lat) <= 60)
    pixels_per_cell, dpi = 2, 128
    panel_width = len(lon) * pixels_per_cell
    panel_height = len(latitude_rows) * pixels_per_cell
    left, right, top, gap, bottom = 70, 30, 80, 55, 130
    width = left + panel_width + right
    domain = arrays["mask"] & (np.abs(lat[:, None]) <= 60)
    # Identical reference-valid support for every candidate; filled cells are hidden.
    valid = np.isfinite(arrays["reference"]) & domain
    all_fields = np.concatenate([arrays["reference"][None], arrays["prediction"]])
    labels = ["Observations"] + [
        item["method"]
        for item in provenance["sources"]
        if item["method"] != "Training seasonal climatology"
    ]
    if len(labels) != len(all_fields):
        raise ValueError("Map labels do not match the exported methods")
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
                height = top + len(labels) * (panel_height + gap) - gap + bottom
                fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                axes = [
                    fig.add_axes(
                        (
                            left / width,
                            (height - top - (row + 1) * panel_height - row * gap)
                            / height,
                            panel_width / width,
                            panel_height / height,
                        )
                    )
                    for row in range(len(labels))
                ]
                expected_pixels = []
                vmin, vmax = limits[column]
                for row, label in enumerate(labels):
                    ax = axes[row]
                    values = np.where(
                        valid[case_index, column], fields[row, column], np.nan
                    )
                    ax.set_facecolor("0.82")
                    im = ax.imshow(
                        values[latitude_rows][:, order],
                        origin="lower",
                        interpolation="none",
                        aspect="equal",
                        cmap="RdBu_r" if anomaly or column == 1 else "turbo",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    rgba = im.to_rgba(values[latitude_rows][:, order], bytes=True)
                    expected_pixels.append(
                        np.where(rgba[..., 3:] > 0, rgba[..., :3], 209)[::-1]
                    )
                    # Index-space axes keep every grid location exactly square.
                    # Tick labels report real coordinates, not an equidistant projection.
                    xticks = [0, 90, 180, 270, len(lon) - 1]
                    yticks = [0, len(latitude_rows) // 2, len(latitude_rows) - 1]
                    ax.set_xticks(xticks, [f"{lon[order][i]:.1f}" for i in xticks])
                    ax.set_yticks(
                        yticks, [f"{lat[latitude_rows][i]:.1f}" for i in yticks]
                    )
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.tick_params(labelsize=7, length=0)
                    ax.set_title(label, fontsize=9)
                    ax.set_ylabel("Latitude (°)")
                axes[-1].set_xlabel("Longitude (°)")
                color_ax = fig.add_axes(
                    (left / width, 55 / height, panel_width / width, 18 / height)
                )
                fig.colorbar(
                    im,
                    cax=color_ax,
                    orientation="horizontal",
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
                    y=1 - 12 / height,
                )
                stem = f"day30-{case}-{kind}-{'sst' if column == 0 else 'adt'}"
                fig.canvas.draw()
                for ax in axes:
                    np.testing.assert_allclose(
                        ax.get_window_extent().size,
                        [panel_width, panel_height],
                        atol=1e-8,
                    )
                fig.savefig(output / (stem + ".png"), dpi=dpi)
                with Image.open(output / (stem + ".png")) as saved:
                    np.testing.assert_array_equal(saved.size, [width, height])
                    pixels = np.asarray(saved)
                for row, expected in enumerate(expected_pixels):
                    y = top + row * (panel_height + gap)
                    actual = pixels[y : y + panel_height, left : left + panel_width, :3]
                    target = np.repeat(
                        np.repeat(expected, pixels_per_cell, axis=0),
                        pixels_per_cell,
                        axis=1,
                    )
                    np.testing.assert_array_equal(actual, target)
                fig.savefig(output / (stem + ".pdf"), dpi=dpi)
                plt.close(fig)
                render.append(
                    {
                        "file": stem,
                        "limits": [vmin, vmax],
                        "figure_pixels": [width, height],
                        "panel_pixels": [panel_width, panel_height],
                        "panel_grid_shape_yx": [len(latitude_rows), len(lon)],
                        "panel_left_px": left,
                        "panel_top_px": [
                            top + row * (panel_height + gap)
                            for row in range(len(labels))
                        ],
                    }
                )
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
            "pixel_verification": "Every saved panel RGB pixel equals the source colormap cell repeated exactly 2x2, including the common missing-data mask",
            "rasterization": "Lossless PNG: exactly 2 x 2 pixels per retained grid cell at native image size; no interpolation. PDF embeds native grid with interpolation disabled.",
            "coordinates": "Square cells in grid-index space, sorted longitude and retained 60S-60N latitude rows; ticks label actual cell-center coordinates. No geographic resampling.",
            "mask": "60S-60N model wet domain intersected with finite target observations, identical across candidates",
            "interpretation": "Illustrative cases, not aggregate skill; gray is land or unavailable observations. Colorbar extensions mark saturation. Endpoint is the last common exported surface lead, not a multi-year continuous rollout.",
        }
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--methods",
        type=Path,
        help="JSON list of [label, evaluation directory, export filename]",
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.extract:
        methods = json.loads(args.methods.read_text()) if args.methods else None
        extract(args.root, args.data, args.bundle, methods)
    else:
        plot(args.bundle, args.output)


if __name__ == "__main__":
    main()
