#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot task-specific latent drift and cached direct-processor pilot outputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from samudra.experiments.diffusion_ab_figures import save_png

ARMS = ("residual", "residual-warm", "direct-warm")
NAMES = ("parent", *ARMS)
ORIGINS = ("2015-01-01", "2018-01-01", "2021-01-01")
COLORS = ("#777777", "#0072B2", "#009E73", "#D55E00")
MARKERS = ("o", "s", "^", "D")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latent_curves(roots, output):
    rows: list[dict] = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for column, origin in enumerate(ORIGINS):
        for name, color, marker in zip(NAMES, COLORS, MARKERS, strict=True):
            for row, task in enumerate(("observation", "OM4")):
                directory = roots[name]
                if task == "observation":
                    record = read(directory / "annual" / f"latent-{origin}.json")
                    days, values = record["leads_days"], record["latent_rms"]
                else:
                    record = read(directory / "native" / f"{origin[:4]}.json")
                    values = record["latent_rms"]
                    days = list(range(0, 5 * len(values), 5))
                if not np.isfinite(values).all() or len(values) != len(days):
                    raise ValueError("Invalid latent trajectory")
                rows.extend(
                    dict(arm=name, task=task, origin=origin, day=day, latent_rms=value)
                    for day, value in zip(days, values, strict=True)
                )
                ax = axes[row, column]
                ax.plot(
                    days,
                    values,
                    label=name,
                    color=color,
                    marker=marker,
                    markevery=12,
                    markersize=4,
                    markerfacecolor="white",
                    linewidth=1,
                )
                ax.set(
                    title=f"{task} surface history · {origin[:4]}",
                    xlabel="Forecast lead (days)",
                    ylabel="Latent RMS",
                    yscale="log",
                )
                ax.grid(alpha=0.2)
    # One shared scale prevents the bounded native curve from visually resembling blow-up.
    limits = [ax.get_ylim() for ax in axes.flat]
    for ax in axes.flat:
        ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Persistent latent evolution: observation/ERA5 versus OM4 surface/flux inputs\n"
        "Separate task adapters; neither initialization observes the full interior"
    )
    save_png(fig, output / "task-latent-rms.png", dpi=130)
    plt.close(fig)
    table(output / "task-latent-rms.csv", rows)


def training_curves(logs, output):
    rows: list[dict] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for name, color, marker in zip(
        (*ARMS, "direct"), (*COLORS[1:], "#CC79A7"), (*MARKERS[1:], "X"), strict=True
    ):
        events = [
            json.loads(line)
            for line in (logs / name / "events.jsonl").read_text().splitlines()
        ]
        events = [e for e in events if e.get("event") == "update"]
        rows.extend(
            dict(arm=name, update=e["update"], task=e["task"], loss=e["loss"])
            for e in events
        )
        for ax, task in zip(axes, ("om4", "observation"), strict=True):
            selected = [e for e in events if e["task"] == task]
            ax.plot(
                [e["update"] for e in selected],
                [e["loss"] for e in selected],
                color=color,
                marker=marker,
                markevery=8,
                markersize=4,
                markerfacecolor="white",
                linewidth=1,
                label=name if name != "direct" else "direct: failed on update 3",
            )
            ax.set(
                xlabel="Added mixed updates",
                ylabel="Training objective",
                yscale="log",
                title="OM4 denoising + replay/bound"
                if task == "om4"
                else "Observation CRPS + auxiliary losses",
            )
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[0].annotate(
        "Direct update 1: 9.64e15\nStopped at nonfinite gradient norm on update 3",
        xy=(1, 9.642945531355138e15),
        xytext=(12, 1e11),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.3"},
    )
    zoom = axes[0].inset_axes((0.51, 0.25, 0.46, 0.34))
    for name, color, marker in zip(ARMS, COLORS[1:], MARKERS[1:], strict=True):
        selected = [r for r in rows if r["arm"] == name and r["task"] == "om4"]
        zoom.plot(
            [r["update"] for r in selected],
            [r["loss"] for r in selected],
            color=color,
            marker=marker,
            markevery=8,
            markersize=3,
            markerfacecolor="white",
            linewidth=0.8,
        )
    zoom.set(title="Continuing arms: expanded scale", yscale="log")
    zoom.title.set_fontsize(8)
    zoom.tick_params(labelsize=7)
    zoom.grid(alpha=0.2)
    fig.suptitle(
        "Paired continuation batches; distinct objectives, not held-out metrics\n"
        "256 processor-only warm-up updates excluded from this axis"
    )
    save_png(fig, output / "training-loss.png", dpi=130)
    plt.close(fig)
    table(output / "training-loss.csv", rows)


def annual_maps(roots, output, origin):
    arrays: list[dict] = []
    provenance = []
    shared_contract = None
    for name in NAMES:
        directory = roots[name] / "annual"
        complete = read(directory / "MEMBERS_COMPLETE.json")
        if complete["inputs"] != read(directory / "input.json"):
            raise ValueError("Annual input receipt mismatch")
        inputs = complete["inputs"]
        count = 8128 if name == "parent" else 8192
        if (
            inputs["counts"] != {"om4": count, "observation": count}
            or inputs["step"] != 2 * count
        ):
            raise ValueError("Unexpected pilot checkpoint update counts")
        if (inputs["members"], inputs["sampling_steps"]) != (8, 32):
            raise ValueError("Unexpected pilot sampler settings")
        contract = {key: inputs[key] for key in ("annual_manifests", "seed", "domain")}
        if shared_contract is not None and contract != shared_contract:
            raise ValueError("Annual data or evaluation protocol mismatch")
        shared_contract = contract
        for filename in (f"{origin}.npz", f"members-{origin}.npz"):
            checksum = digest(directory / filename)
            if checksum != complete["files"][filename]:
                raise ValueError(f"Checksum mismatch: {directory / filename}")
            provenance.append(
                dict(arm=name, file=str(directory / filename), sha256=checksum)
            )
        with np.load(directory / f"members-{origin}.npz") as saved:
            indices = [saved["leads_days"].tolist().index(day) for day in (30, 365)]
            fields = saved["fields"][:, indices, :2].copy()
            if fields.shape[:3] != (8, 2, 2) or not np.isfinite(fields).all():
                raise ValueError("Invalid annual members")
            data = {key: saved[key] for key in ("mask", "lat", "lon")}
        with np.load(directory / f"{origin}.npz") as saved:
            data["reference"] = saved["reference"][[5, 72], :2]
        data["mean"], data["draw"] = fields.mean(0), fields[0]
        if arrays:
            for key in ("reference", "mask", "lat", "lon"):
                np.testing.assert_array_equal(data[key], arrays[0][key])
        arrays.append(data)
    clip_rows = []
    for time, day in enumerate((30, 365)):
        for channel, label, unit, limits in (
            (0, "SST", "°C", (-2, 32)),
            (1, "SSH", "m", (-1, 2)),
        ):
            mask = arrays[0]["mask"][channel].astype(bool)
            target = arrays[0]["reference"][time, channel]
            height, width = target.shape
            tw, th = 2 * width + 90, 5 * (height + 28) + 190
            fig = plt.figure(figsize=(tw / 100, th / 100), dpi=100)
            fields = [("Observed reference", target, target)] + [
                (name, data["mean"][time, channel], data["draw"][time, channel])
                for name, data in zip(NAMES, arrays, strict=True)
            ]
            for row, (name, mean, draw) in enumerate(fields):
                for col, field in enumerate((mean, draw)):
                    valid = mask & np.isfinite(field)
                    clipped = float(
                        ((field[valid] < limits[0]) | (field[valid] > limits[1])).mean()
                    )
                    clip_rows.append(
                        dict(
                            arm=name,
                            origin=origin,
                            day=day,
                            channel=label,
                            mode="mean8" if col == 0 else "draw1",
                            vmin=limits[0],
                            vmax=limits[1],
                            clipped_fraction=clipped,
                            minimum=float(field[valid].min()),
                            maximum=float(field[valid].max()),
                        )
                    )
                    ax = fig.add_axes(
                        (
                            (30 + col * (width + 25)) / tw,
                            (th - 105 - row * (height + 28) - height) / th,
                            width / tw,
                            height / th,
                        )
                    )
                    im = ax.imshow(
                        np.where(valid, field, np.nan),
                        origin="lower",
                        interpolation="nearest",
                        cmap="viridis",
                        vmin=limits[0],
                        vmax=limits[1],
                    )
                    kind = "mean of 8" if col == 0 else "draw 1"
                    suffix = f" · {kind} · {clipped:.0%} clipped" if row else ""
                    ax.set_title(name + suffix, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
            bar = fig.add_axes((0.15, 48 / th, 0.7, 10 / th))
            fig.colorbar(
                im,
                cax=bar,
                orientation="horizontal",
                extend="both",
                label=f"{label} ({unit}), fixed range {limits[0]} to {limits[1]}",
            )
            fig.suptitle(
                f"Observation initialized · {origin} · day-{day} five-day bin · {label}\n"
                "Mean and individual conditional readout; draws are independent across time\n"
                "Native 1:1 grid pixels; identical scales; out-of-range values clip",
                fontsize=9,
                y=1 - 15 / th,
            )
            save_png(fig, output / f"obs-day{day}-{label.lower()}.png", dpi=100)
            plt.close(fig)
    table(output / "obs-map-clipping.csv", clip_rows)
    (output / "supplement-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", choices=ORIGINS, default=ORIGINS[0])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roots = {"parent": args.parent, **{name: args.root / name for name in ARMS}}
    latent_curves(roots, args.output)
    training_curves(args.logs, args.output)
    annual_maps(roots, args.output, args.origin)


if __name__ == "__main__":
    main()
