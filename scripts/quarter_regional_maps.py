# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact-pixel regional maps from the verified spectral input bundles."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from quarter_regional_spectra import (  # type: ignore[import-not-found]
    REGIONS,
    common_support,
    geostrophic,
    sha256,
    validate_dates,
    validate_source,
)


def load_sources(config):
    sources = {}
    aliases = {
        "surface": "fields",
        "mask": "wet",
        "lead_days": "leads_days",
        "bin_start": "target_bin_starts",
        "bin_end": "target_bin_ends",
        "bin_midpoint": "target_bin_midpoints",
    }
    for record in config["sources"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Input hash mismatch")
        with np.load(record["path"], allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        for key, alias in aliases.items():
            if key not in data:
                data[key] = (
                    data["leads"]
                    if key == "lead_days" and alias not in data
                    else data[alias]
                )
        validate_source(data)
        sources[record["label"]] = data
    validate_dates(sources)
    return sources


def raster(values, valid, low, high, palette, factor, logarithmic=False):
    """Every native cell becomes a constant factor-by-factor pixel block."""
    if logarithmic:
        scaled = (np.log10(np.maximum(values, low)) - np.log10(low)) / (
            np.log10(high) - np.log10(low)
        )
    else:
        scaled = (values - low) / (high - low)
    rgba = matplotlib.colormaps[palette](np.clip(scaled, 0, 1), bytes=True)
    rgba[~valid] = [215, 215, 215, 255]
    rgb = rgba[::-1, :, :3]
    image = Image.fromarray(rgb)
    return image.resize(
        (image.width * factor, image.height * factor), Image.Resampling.NEAREST
    )


def run(config, output):
    sources = load_sources(config)
    grids = [(s["lat"], s["lon"]) for s in sources.values()]
    reference = int(np.argmin([len(y) * len(x) for y, x in grids]))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14)
    small = ImageFont.truetype(font_path, 11)
    titlefont = ImageFont.truetype(font_path, 19)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for variable, channel, palette, unit in [
        ("sst", 0, "turbo", "°C"),
        ("adt", 1, "RdBu_r", "m"),
        ("geostrophic-ke", 1, "magma", "m²/s²; log color scale"),
    ]:
        fields, valid = [], []
        for source in sources.values():
            field = np.where(
                source["mask"][channel], source["surface"][:, :, channel], np.nan
            )
            if variable == "geostrophic-ke":
                u, v = geostrophic(field, source["lat"], source["lon"])
                field = 0.5 * (u**2 + v**2)
            fields.append(field)
            valid.append(np.isfinite(field).all(axis=(0, 1)))
        shared, supports = common_support(valid, grids, reference)
        for region, xb, yb in REGIONS:
            blocks, masks, factors = [], [], []
            for field, support, (lat, lon) in zip(fields, supports, grids, strict=True):
                yi = np.flatnonzero((lat >= yb[0]) & (lat <= yb[1]))
                xi = np.flatnonzero((lon >= xb[0]) & (lon <= xb[1]))
                blocks.append(field[:, :, yi][:, :, :, xi])
                masks.append(support[np.ix_(yi, xi)])
                factors.append(8 if len(lat) == 180 else 2)
            values = [
                b[..., m].ravel() for b, m in zip(blocks, masks, strict=True) if m.any()
            ]
            if not values:
                continue
            all_values = np.concatenate(values)
            low, high = np.percentile(all_values, [1, 99])
            if variable == "adt":
                high = max(abs(low), abs(high))
                low = -high
            if variable == "geostrophic-ke":
                low = max(float(low), 1e-6)
                high = max(float(high), low * 10)
            if high <= low:
                raise ValueError("Degenerate color scale")
            width = max(b.shape[-1] * f for b, f in zip(blocks, factors, strict=True))
            height = max(b.shape[-2] * f for b, f in zip(blocks, factors, strict=True))
            colwidth = max(width, 220)
            left, top, gap = 150, 120, 12
            canvas = Image.new(
                "RGB",
                (left + 5 * (colwidth + gap), top + 6 * (height + 30) + 95),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 8),
                f"{region} — {variable.upper()} — {xb[0]}–{xb[1]}°E, {yb[0]}–{yb[1]}°N",
                font=titlefont,
                fill="black",
            )
            draw.text(
                (12, 36),
                "Five-day means; native fields on the spectrum's shared wet support. North up; east right.",
                font=font,
                fill="black",
            )
            draw.text(
                (12, 57),
                "Shared colors across all models, years and leads; 1st–99th percentile limits (tails clipped). Gray = excluded.",
                font=small,
                fill="black",
            )
            labels = [
                "U-global selected\n¼°",
                "Observation-only selected\n¼°",
                "Legacy U-global\n1°, seed 1729",
                "OM4\n¼°",
                "Observations\n¼°",
            ]
            for col, label in enumerate(labels):
                draw.multiline_text(
                    (left + col * (colwidth + gap), 80), label, font=font, fill="black"
                )
            panel_receipts = []
            for oi, year in enumerate((2015, 2018, 2021)):
                for li, lead in enumerate((30, 365)):
                    y = top + (oi * 2 + li) * (height + 30)
                    draw.multiline_text(
                        (10, y + 5),
                        f"{year}-01-01 start\nDay {lead}",
                        font=font,
                        fill="black",
                    )
                    for col, (block, mask, factor) in enumerate(
                        zip(blocks, masks, factors, strict=True)
                    ):
                        tile = raster(
                            block[oi, li],
                            mask,
                            low,
                            high,
                            palette,
                            factor,
                            variable == "geostrophic-ke",
                        )
                        # Audit pixel blocks: no filtering or interpolated values.
                        arr = np.asarray(tile)
                        reduced = arr[::factor, ::factor]
                        assert np.array_equal(
                            arr,
                            np.repeat(
                                np.repeat(reduced, factor, axis=0), factor, axis=1
                            ),
                        )
                        x = left + col * (colwidth + gap)
                        canvas.paste(tile, (x, y))
                        panel_receipts.append(
                            {
                                "source": list(sources)[col],
                                "origin": year,
                                "lead": lead,
                                "native_shape": list(block.shape[-2:]),
                                "pixels_per_cell": factor,
                                "rectangle": [x, y, tile.width, tile.height],
                            }
                        )
            barwidth = min(600, canvas.width - 180)
            bar_y = canvas.height - 65
            ramp = (
                np.linspace(low, high, barwidth)
                if variable != "geostrophic-ke"
                else np.geomspace(low, high, barwidth)
            )
            bar = raster(
                np.broadcast_to(ramp, (16, barwidth)),
                np.ones((16, barwidth), bool),
                low,
                high,
                palette,
                1,
                variable == "geostrophic-ke",
            )
            canvas.paste(bar, (left, bar_y))
            for fraction in (0, 0.25, 0.5, 0.75, 1):
                value = (
                    low + fraction * (high - low)
                    if variable != "geostrophic-ke"
                    else low * (high / low) ** fraction
                )
                draw.text(
                    (left + int(fraction * barwidth), bar_y + 20),
                    f"{value:.3g}",
                    font=small,
                    fill="black",
                )
            draw.text((left + barwidth + 50, bar_y), unit, font=font, fill="black")
            filename = (
                region.lower().replace(" ", "-").replace("ñ", "n").replace(".", "")
                + "-"
                + variable
                + ".png"
            )
            canvas.save(output / filename, optimize=True)
            records.append(
                {
                    "file": filename,
                    "sha256": sha256(output / filename),
                    "region": region,
                    "variable": variable,
                    "color_limits": [float(low), float(high)],
                    "panels": panel_receipts,
                    "support_sha256": hashlib.sha256(shared.tobytes()).hexdigest(),
                }
            )
            print(filename, flush=True)
    receipt = {
        "inputs": config,
        "maps": records,
        "pixel_checks_passed": True,
        "note": "Quarter native cells2x2, legacy native cells8x8 for matching geographic footprint; no field interpolation. KE is scalar 0.5(u_g²+v_g²), not the quantity Fourier-transformed for component spectra.",
    }
    (output / "provenance.json.gz").write_bytes(
        gzip.compress(json.dumps(receipt, indent=2).encode(), mtime=0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
