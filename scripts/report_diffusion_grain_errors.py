#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native-pixel physical/error maps from scratch diffusion evaluation archives.

Repeat --run LABEL=EVALUATION_ROOT to compare checkpoints with shared scales.
Native OM4 includes initialization (day 0); annual obs uses five-day target bins.
Writes separate panels to preserve the repository's 250 KiB asset limit. No spatial
smoothing or interpolation is applied. Range JSON records clipping and raw errors.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LICENSE = (
    "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
)
CHANNELS = {
    "thetao_0": ("SST", "°C"),
    "zos": ("SSH", "m"),
    "thetao_9": ("Temperature, level 9", "°C"),
    "so_9": ("Salinity, level 9", "practical salinity"),
}


def load_fields(root, source, origin, requested):
    """Return physical members/reference; never use latent states as truth."""
    if source == "native":
        with np.load(root / "native" / f"fields-{origin[:4]}.npz") as data:
            leads = data["leads_days"].tolist()
            indices = [i for i, lead in enumerate(leads) if lead in requested]
            return {
                "leads": [leads[i] for i in indices],
                "reference": data["reference"][indices],
                "members": data["members"][:, indices],
                "mask": data["mask"].astype(bool),
                "channels": data["channels"].tolist(),
            }
    with (
        np.load(root / "annual" / f"{origin}.npz") as reference,
        np.load(root / "annual" / f"members-{origin}.npz") as samples,
    ):
        leads = samples["leads_days"].tolist()
        indices = [i for i, lead in enumerate(leads) if lead in requested]
        # Only SSH/SST have matching daily observed verification targets here.
        # Interior reconstruction requires the monthly validation target archive.
        return {
            "leads": [leads[i] for i in indices],
            "reference": reference["reference"][indices, :2],
            "members": samples["fields"][:, indices, :2],
            "mask": samples["mask"][:2].astype(bool),
            "channels": ["thetao_0", "zos"],
        }


def load_baseline(directory, model, origin, base):
    """Verify viewer file hashes and selected ocean target bins before selecting a baseline."""
    catalog = json.loads((directory / "catalog.json").read_text())

    def verified(filename):
        path = directory / filename
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != catalog["files"][filename]["sha256"]:
            raise ValueError(f"Viewer asset hash mismatch: {path}")
        return np.load(path)

    record = catalog["records"][origin]
    if model in record["models"]:
        filename = record["models"][model]["surface"]
    elif model in catalog["annual_extension"]["models"]:
        filename = f"annual-{model}-{origin}-surface.npy"
    else:
        raise ValueError(f"Unknown viewer baseline model: {model}")
    prediction = verified(filename)
    reference = verified(record["surface_reference"])
    if prediction.shape != (73, 2, 180, 360) or reference.shape != prediction.shape:
        raise ValueError("Expected 73 five-day surface verification bins")
    indices = [lead // 5 - 1 for lead in base["leads"]]
    if any(lead < 5 or lead % 5 for lead in base["leads"]):
        raise ValueError("Baseline requires five-day annual verification leads")
    valid = np.broadcast_to(base["mask"], base["reference"].shape)
    np.testing.assert_allclose(
        reference[indices][valid], base["reference"][valid], equal_nan=True
    )
    np.testing.assert_array_equal(
        verified(catalog["surface_mask"]).astype(bool), base["mask"]
    )
    return dict(
        leads=base["leads"],
        channels=base["channels"],
        mask=base["mask"],
        reference=base["reference"],
        members=prediction[None, indices],
    )


def panel(field, mask, limits, title, subtitle, unit, path, diverging=False):
    """Render each grid cell as exactly one image pixel, with explicit saturation."""
    valid = mask & np.isfinite(field)
    values = field[valid]
    if not values.size:
        raise ValueError(f"No finite ocean values: {title}")
    clipped = float(np.mean((values < limits[0]) | (values > limits[1])))
    height, width = field.shape
    canvas_width = max(width + 40, 460)
    canvas_height = height + 170
    fig = plt.figure(figsize=(canvas_width / 100, canvas_height / 100), dpi=100)
    ax = fig.add_axes(
        (
            (canvas_width - width) / 2 / canvas_width,
            96 / canvas_height,
            width / canvas_width,
            height / canvas_height,
        )
    )
    cmap = plt.get_cmap("RdBu_r" if diverging else "viridis").copy()
    cmap.set_bad("0.82")
    image = ax.imshow(
        np.ma.masked_where(~valid, field),
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
    )
    ax.set_axis_off()
    np.testing.assert_allclose(
        [ax.get_window_extent().width, ax.get_window_extent().height], [width, height]
    )
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=8)
    cax = fig.add_axes((0.16, 65 / canvas_height, 0.68, 9 / canvas_height))
    fig.colorbar(image, cax=cax, orientation="horizontal", extend="both")
    cax.tick_params(labelsize=7)
    fig.text(
        0.5,
        0.035,
        f"{unit}; {clipped:.1%} clipped; 1 pixel/cell, gray land/missing\n"
        f"Finite ocean range: {values.min():.4g} to {values.max():.4g}",
        ha="center",
        fontsize=7,
    )
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    with Image.open(path) as rendered:
        rendered.convert("RGB").save(path, optimize=True, compress_level=9)
    if path.stat().st_size > 250 * 1024:
        raise ValueError(f"Panel exceeds repository size limit: {path}")
    path.with_name(path.name + ".license").write_text(LICENSE)
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "clipped_fraction": clipped,
        "finite_ocean_cells": int(values.size),
        "nonfinite_ocean_cells": int(np.sum(mask & ~np.isfinite(field))),
        "limits": [float(x) for x in limits],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="LABEL=EVALUATION_ROOT"
    )
    parser.add_argument(
        "--viewer-baseline",
        action="append",
        default=[],
        help="LABEL=VIEWER_DIRECTORY:MODEL (annual only)",
    )
    parser.add_argument("--source", choices=("native", "annual"), required=True)
    parser.add_argument("--origin", default="2018-01-01")
    parser.add_argument("--leads", type=int, nargs="+", default=[0, 30, 365])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {}
    paths = {}
    for value in args.run:
        label, path = value.split("=", 1)
        if label in runs:
            raise ValueError(f"Duplicate run label: {label}")
        runs[label] = load_fields(Path(path), args.source, args.origin, args.leads)
        paths[label] = str(Path(path).resolve())
    base = next(iter(runs.values()))
    for value in args.viewer_baseline:
        if args.source != "annual":
            raise ValueError("Viewer baselines support annual observations only")
        label, location = value.split("=", 1)
        directory, model = location.rsplit(":", 1)
        if label in runs:
            raise ValueError(f"Duplicate run label: {label}")
        runs[label] = load_baseline(Path(directory), model, args.origin, base)
        paths[label] = location
    for label, data in runs.items():
        if data["leads"] != base["leads"] or data["channels"] != base["channels"]:
            raise ValueError(f"Mismatched leads/channels: {label}")
        np.testing.assert_array_equal(data["mask"], base["mask"])
        np.testing.assert_allclose(data["reference"], base["reference"], equal_nan=True)
    summary = {
        "source": args.source,
        "origin": args.origin,
        "runs": paths,
        "missing_requested_leads": sorted(set(args.leads) - set(base["leads"])),
        "semantics": "Native: physical OM4 targets. Annual: five-day observed surface bins. "
        "Draw 1 is independently sampled at each lead, not a coherent trajectory. "
        "Spread is pointwise sample standard deviation (ddof=1), not error.",
        "scales": "Physical limits use pooled reference 1st/99th percentiles across selected leads. "
        "Error and spread limits use pooled 99th percentiles across runs and selected leads. "
        "Errors are prediction minus reference. RMSE is unweighted over finite ocean cells, "
        "a map diagnostic, not the blended rollout score.",
        "panels": {},
    }
    for c, channel in enumerate(base["channels"]):
        name, unit = CHANNELS[channel]
        mask = base["mask"][c]
        refs = base["reference"][:, c]
        valid = np.broadcast_to(mask, refs.shape) & np.isfinite(refs)
        physical_limits = np.quantile(refs[valid], [0.01, 0.99])
        error_values, spread_values = [], []
        for data in runs.values():
            draws = data["members"][:, :, c].astype(np.float64)
            for prediction in (draws.mean(0), draws[0]):
                error = prediction - refs
                error_values.append(np.abs(error[valid & np.isfinite(error)]))
            spread = draws.std(0, ddof=1) if len(draws) > 1 else np.zeros_like(draws[0])
            spread_values.append(spread[valid & np.isfinite(spread)])
        error_max = max(float(np.quantile(np.concatenate(error_values), 0.99)), 1e-12)
        spread_max = max(float(np.quantile(np.concatenate(spread_values), 0.99)), 1e-12)
        for label, data in runs.items():
            safe = re.sub(r"[^A-Za-z0-9_-]+", "-", label)
            for i, lead in enumerate(data["leads"]):
                draws = data["members"][:, i, c].astype(np.float64)
                mean, member = draws.mean(0), draws[0]
                spread = draws.std(0, ddof=1) if len(draws) > 1 else np.zeros_like(mean)
                fields = {
                    "reference": (refs[i], physical_limits, False),
                    "mean": (mean, physical_limits, False),
                    "draw1": (member, physical_limits, False),
                    "mean-error": (mean - refs[i], (-error_max, error_max), True),
                    "draw1-error": (member - refs[i], (-error_max, error_max), True),
                    "spread": (spread, (0, spread_max), False),
                }
                for kind, (field, limits, diverging) in fields.items():
                    if len(draws) == 1 and kind in ("draw1", "draw1-error", "spread"):
                        continue
                    path = args.output / f"{safe}-{channel}-day{lead}-{kind}.png"
                    details = panel(
                        field,
                        mask,
                        limits,
                        f"{label}: {name}, day {lead}",
                        f"{kind}; {args.source} {args.origin}; {len(draws)} member(s)",
                        unit,
                        path,
                        diverging,
                    )
                    if kind.endswith("error"):
                        values = field[mask & np.isfinite(field)]
                        details["unweighted_rmse"] = float(np.sqrt(np.mean(values**2)))
                        details["unweighted_bias"] = float(np.mean(values))
                    summary["panels"][path.name] = details
    destination = args.output / "map-ranges.json"
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    destination.with_name(destination.name + ".license").write_text(LICENSE)


if __name__ == "__main__":
    main()
