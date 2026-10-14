#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Physical early-checkpoint maps with reference-fixed limits and explicit clipping."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from samudra.experiments.diffusion_ab_figures import save_png

LICENSE = (
    "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
)


def maps(truth, draws, mask, output, title, unit, temporal):
    """Two leads, three columns; each 180x360 field uses exactly 360x180 pixels."""
    fields = [[t, d.mean(0), d[0]] for t, d in zip(truth, draws, strict=True)]
    reference = np.concatenate([t[mask & np.isfinite(t)] for t in truth])
    low, high = np.quantile(reference, [0.01, 0.99])
    width, height = 2400, 1120
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=50)
    fig.text(0.5, 0.966, title, ha="center", fontsize=20)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.82")
    details = []
    for row, lead in enumerate((30, 365)):
        for col, label in enumerate(("Reference", "Mean of 8", "Draw 1")):
            field = fields[row][col]
            valid = mask & np.isfinite(field)
            values = field[valid]
            clipped = float(np.mean((values < low) | (values > high)))
            minimum, maximum = float(values.min()), float(values.max())
            details.append(
                dict(
                    lead=lead,
                    panel=label,
                    minimum=minimum,
                    maximum=maximum,
                    clipped_fraction=clipped,
                    unit=unit,
                )
            )
            ax = fig.add_axes(
                (
                    (55 + col * 800) / width,
                    (650 - row * 465) / height,
                    720 / width,
                    360 / height,
                )
            )
            im = ax.imshow(
                np.ma.masked_where(~valid, field),
                origin="lower",
                cmap=cmap,
                vmin=low,
                vmax=high,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(
                f"Day {lead}: {label}\nRange {minimum:.4g} to {maximum:.4g} {unit}; "
                f"{100 * clipped:.1f}% clipped",
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            bounds = ax.get_window_extent()
            np.testing.assert_allclose([bounds.width, bounds.height], [360, 180])
    cax = fig.add_axes((0.25, 0.125, 0.5, 0.016))
    fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        extend="both",
        label=f"{unit}; shared reference-only 1–99% range at both leads",
    )
    fig.text(
        0.5,
        0.025,
        f"{temporal}\nNative grid: exactly 1 × 1 image pixel per cell. "
        "Gray: land or missing data. Clipped percentages use finite ocean cells.\n"
        "Draw 1 uses independent diffusion samples at each lead; it is not a coherent sampled trajectory.",
        ha="center",
        fontsize=12,
    )
    save_png(fig, output, dpi=50)
    plt.close(fig)
    # Split rows without resizing or quantization, retaining exact RGB cell colors.
    with Image.open(output) as rendered:
        for lead, top, bottom in ((30, 28, 240), (365, 258, 474)):
            sections = [
                rendered.crop((0, 0, 1200, 26)),
                rendered.crop((0, top, 1200, bottom)),
                rendered.crop((0, 478, 1200, 560)),
            ]
            panel = Image.new(
                "RGB", (1200, sum(section.height for section in sections)), "white"
            )
            offset = 0
            for section in sections:
                panel.paste(section, (0, offset))
                offset += section.height
            destination = output.with_name(f"{output.stem}-day{lead}.png")
            panel.save(destination, optimize=True, compress_level=9)
            if destination.stat().st_size > 250 * 1024:
                raise ValueError(f"Figure exceeds repository size limit: {destination}")
            destination.with_name(destination.name + ".license").write_text(LICENSE)
    output.unlink()
    output.with_name(output.name + ".license").unlink(missing_ok=True)
    return dict(color_limits=[float(low), float(high)], panels=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    with (
        np.load(args.root / "annual" / "2018-01-01.npz") as annual,
        np.load(args.root / "annual" / "members-2018-01-01.npz") as ensemble,
    ):
        indices = [ensemble["leads_days"].tolist().index(day) for day in (30, 365)]
        members = ensemble["fields"][:, indices]
        reference = annual["reference"][indices]
        for channel, name, unit in ((0, "sst", "°C"), (1, "ssh", "m")):
            path = args.output / f"obs-2018-{name}.png"
            summary[path.name] = maps(
                reference[:, channel],
                members[:, :, channel].swapaxes(0, 1),
                ensemble["mask"][channel],
                path,
                f"Observation initialization, 2018-01-01: {name.upper()} — direct scratch, 2k obs",
                unit,
                "Surface comparison uses five-day verification bins labeled day 30 and day 365.",
            )
    with np.load(args.root / "native" / "fields-2018.npz") as native:
        indices = [native["leads_days"].tolist().index(day) for day in (30, 365)]
        for name, label, unit in (
            ("thetao_0", "sst", "°C"),
            ("zos", "ssh", "m"),
            ("so_9", "so9", "practical salinity"),
        ):
            channel = native["channels"].tolist().index(name)
            path = args.output / f"om4-2018-{label}.png"
            summary[path.name] = maps(
                native["reference"][indices, channel],
                native["members"][:, indices, channel].swapaxes(0, 1),
                native["mask"][channel],
                path,
                f"OM4 surface initialization, 2018: {name} — direct scratch, 2k obs",
                unit,
                "Matched OM4 targets on native five-day output cadence; interiors are inferred from surface history.",
            )
    (args.output / "map-ranges.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "map-ranges.json.license").write_text(LICENSE)


if __name__ == "__main__":
    main()
