# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render full-state comparison arrays without GPU inference."""

import argparse
import csv
import gzip
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
HELPER_PATH = HERE.parent / "pre-spike-noise/render.py"
spec = importlib.util.spec_from_file_location("spectral_helpers", HELPER_PATH)
assert spec and spec.loader
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
FIELDS = [
    ("uo", "Eastward velocity", "m/s", 0),
    ("vo", "Northward velocity", "m/s", 19),
    ("thetao", "Temperature", "°C", 38),
    ("so", "Salinity", "PSU", 57),
    ("zos", "Sea surface height", "m", 76),
]
REGIONS = [
    ("north-pacific", "North Pacific", slice(468, 532), slice(708, 772)),
    ("gulf-stream", "Gulf Stream", slice(468, 532), slice(1208, 1272)),
    ("agulhas", "Agulhas", slice(168, 232), slice(168, 232)),
]
MODELS = [
    ("classifier", "Small classifier", "#d55e00"),
    ("samudra", "Samudra", "#0072b2"),
    ("persistence", "Persistence", "#999900"),
]


def area_rmse(prediction, target, wet, area):
    if not wet.any():
        return float("nan")
    return float(
        np.sqrt(
            np.average(
                (prediction[wet].astype(float) - target[wet]) ** 2, weights=area[wet]
            )
        )
    )


def maps(data, channels, region, title, unit, output, stride):
    rows, cols = region
    width = 2110 if stride == 4 else 1340
    height = 135 + 260 * len(channels)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    headings = [
        "OM4",
        "Small all-fields classifier",
        "Samudra",
        "Classifier − OM4",
        "Samudra − OM4",
    ]
    for index, channel in enumerate(channels):
        wet = data["wet"][channel, rows, cols]
        area = data["area"][rows, cols]
        gold = data["gold"][channel, rows, cols]
        classifier = data["classifier"][channel, rows, cols]
        samudra = data["samudra"][channel, rows, cols]
        values = [gold, classifier, samudra, classifier - gold, samudra - gold]
        if not wet.any():
            continue
        field_min = min(float(value[wet].min()) for value in values[:3])
        field_max = max(float(value[wet].max()) for value in values[:3])
        error_limit = max(
            1e-12, *(float(np.abs(value[wet]).max()) for value in values[3:])
        )
        if stride == 4:
            field_min, field_max = np.quantile(gold[wet], [0.005, 0.995])
            error_limit = max(
                1e-12,
                float(
                    np.quantile(
                        np.concatenate([np.abs(value[wet]) for value in values[3:]]),
                        0.995,
                    )
                ),
            )
        depth = 0 if channel == 76 else data["depths"][channel % 19]
        fig.text(
            0.005, (height - 170 - 260 * index) / height, f"{depth:g} m", fontsize=11
        )
        for column, value in enumerate(values):
            panel_width, panel_height = (360, 180) if stride == 4 else (192, 192)
            left = 85 + 400 * column if stride == 4 else 85 + 240 * column
            bottom = height - 295 - 260 * index
            ax = fig.add_axes(
                (
                    left / width,
                    bottom / height,
                    panel_width / width,
                    panel_height / height,
                )
            )
            image = ax.imshow(
                np.ma.masked_where(~wet[::stride, ::stride], value[::stride, ::stride]),
                origin="lower",
                interpolation="nearest",
                resample=False,
                aspect="auto",
                cmap="viridis" if column < 3 else "RdBu_r",
                vmin=field_min if column < 3 else -error_limit,
                vmax=field_max if column < 3 else error_limit,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if index == 0:
                ax.set_title(headings[column], fontsize=11)
            if column in (1, 3) and stride == 1:
                ax.add_patch(
                    Rectangle(
                        (23, 23), 17, 17, fill=False, color="black", ls="--", lw=0.7
                    )
                )
            if column >= 3:
                score = area_rmse(values[column - 2], gold, wet, area)
                ax.set_xlabel(f"RMSE {score:.4g} {unit}", fontsize=10)
            if column in (2, 4):
                cax = fig.add_axes(
                    (
                        (left + panel_width + 8) / width,
                        bottom / height,
                        9 / width,
                        panel_height / height,
                    )
                )
                fig.colorbar(image, cax=cax).ax.tick_params(labelsize=8)
    sampling = (
        "1° display · field scale OM4 0.5–99.5% · error scale pooled |error| 99.5% (extremes saturated)"
        if stride == 4
        else "64×64 native patch · dashed box 17×17 input context"
    )
    fig.suptitle(
        f"{title} · +5 days · target 2016-07-21 · {unit}\n{sampling} · hard cell boundaries · row-shared field/error scales",
        fontsize=12,
        y=0.985,
    )
    fig.savefig(output, dpi=100)
    plt.close(fig)


def spectrum(values):
    side = len(values)
    window = np.outer(np.hanning(side + 1)[:-1], np.hanning(side + 1)[:-1])
    power = abs(
        np.fft.fft2(helpers.remove_plane(values.astype(float)) * window, norm="forward")
    ) ** 2 / np.mean(window**2)
    fy, fx = np.meshgrid(np.fft.fftfreq(side), np.fft.fftfreq(side), indexing="ij")
    frequency = np.hypot(fy, fx)
    edges = np.linspace(0, 0.5, side // 2 + 1)
    shells = [
        (frequency > low) & (frequency <= high)
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    ]
    selected = [mask for mask in shells if mask.any()]
    return [
        (float(frequency[mask].mean()), float(power[mask].sum())) for mask in selected
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "report")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.arrays) as archive:
        data = {key: archive[key] for key in archive.files}
    assert data["gold"].shape == (77, 720, 1440)
    if data["wet"].shape == (19, 720, 1440):
        data["wet"] = np.concatenate([data["wet"]] * 4 + [data["wet"][:1]])
    for model, _, _ in MODELS:
        assert np.isfinite(data[model][data["wet"]]).all()
    regional_metrics = []
    spectral_rows = []
    for field, label, unit, offset in FIELDS:
        depths = [0] if field == "zos" else [0, 7, 13]
        all_channels = [76] if field == "zos" else list(range(offset, offset + 19))
        for start in range(0, len(all_channels), 4):
            maps(
                data,
                all_channels[start : start + 4],
                (slice(None), slice(None)),
                f"Global {label}",
                unit,
                args.output / f"global-{field}-{start:02d}.png",
                4,
            )
        for region_id, region_label, rows, cols in REGIONS:
            maps(
                data,
                [offset + depth for depth in depths],
                (rows, cols),
                f"{region_label} · {label}",
                unit,
                args.output / f"{region_id}-{field}.png",
                1,
            )
            for depth in depths:
                channel = offset + depth
                wet = data["wet"][channel, rows, cols]
                gold = data["gold"][channel, rows, cols]
                for model, _, _ in MODELS:
                    regional_metrics.append(
                        dict(
                            region=region_id,
                            field=field,
                            channel=str(data["channels"][channel]),
                            depth_m=float(data["depths"][depth])
                            if field != "zos"
                            else 0,
                            model=model,
                            area_rmse=area_rmse(
                                data[model][channel, rows, cols],
                                gold,
                                wet,
                                data["area"][rows, cols],
                            ),
                            wet_cells=int(wet.sum()),
                        )
                    )
        for kind in ("field", "error"):
            fig, axes = plt.subplots(
                3,
                len(depths),
                figsize=(5 * len(depths), 11),
                squeeze=False,
                layout="constrained",
            )
            for row, (region_id, region_label, rows, cols) in enumerate(REGIONS):
                for column, depth in enumerate(depths):
                    channel = offset + depth
                    ax = axes[row, column]
                    tile = helpers.wet_patch(data["wet"][channel, rows, cols])
                    ax.set_title(
                        f"{region_label} · {data['depths'][depth]:g} m"
                        if field != "zos"
                        else region_label
                    )
                    if tile is None:
                        ax.text(0.1, 0.5, "No fully wet 32×32 tile")
                        ax.set_axis_off()
                        continue
                    y, x, side = tile
                    gold = data["gold"][channel, rows, cols][y : y + side, x : x + side]
                    models = (
                        [("gold", "OM4", "black")] + MODELS
                        if kind == "field"
                        else MODELS
                    )
                    for model, model_label, color in models:
                        values = data[model][channel, rows, cols][
                            y : y + side, x : x + side
                        ]
                        if kind == "error":
                            values = values - gold
                        curve = spectrum(values)
                        ax.loglog(
                            *zip(*curve, strict=True),
                            color=color,
                            label=model_label,
                            ls=":" if model == "persistence" else "-",
                        )
                        for frequency, power in curve:
                            spectral_rows.append(
                                dict(
                                    region=region_id,
                                    field=field,
                                    depth=int(depth),
                                    model=model,
                                    kind=kind,
                                    cycles_per_cell=frequency,
                                    power=power,
                                    tile_y=y,
                                    tile_x=x,
                                    tile_side=side,
                                )
                            )
                    ax.axvspan(0.25, 0.5, color="gray", alpha=0.1)
                    ax.grid(alpha=0.2)
                    ax.set_xlabel("Cycles / native grid cell")
                    ax.set_ylabel(f"Power per shell ({unit})²")
                    if row == 0 and column == 0:
                        ax.legend(fontsize=8)
            fig.suptitle(
                f"{label} {kind} spectra · planar detrend + Hann · shaded wavelengths 2–4 cells",
                fontsize=13,
            )
            fig.savefig(args.output / f"{field}-{kind}-spectra.png", dpi=110)
            plt.close(fig)
    for filename, records in [
        ("regional-metrics.csv", regional_metrics),
        ("regional-spectra.csv.gz", spectral_rows),
    ]:
        opener = gzip.open if filename.endswith(".gz") else open
        with opener(args.output / filename, "wt") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, (field, label, unit, offset) in zip(axes.flat, FIELDS[:4], strict=True):
        for model, model_label, color in MODELS:
            scores = [
                area_rmse(
                    data[model][channel],
                    data["gold"][channel],
                    data["wet"][channel],
                    data["area"],
                )
                for channel in range(offset, offset + 19)
            ]
            ax.plot(scores, data["depths"], label=model_label, color=color, marker=".")
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(f"Area-weighted RMSE ({unit})")
        ax.set_ylabel("Depth (m)")
        ax.set_title(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Full native quarter-degree global errors · target 2016-07-21 · +5 days"
    )
    fig.savefig(args.output / "global-depth-rmse.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
