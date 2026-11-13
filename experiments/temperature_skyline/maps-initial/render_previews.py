# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render lossless contiguous raster figures directly from the pointwise archive."""

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from samudra.metrics.spectra import SPATIAL_REGIONS
from samudra.skyline.map_report import render, subset

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
args = parser.parse_args()
parts: dict[str, list] = {}
with tarfile.open(args.archive) as archive:
    for member in archive.getmembers():
        if not member.name.endswith(".npz"):
            continue
        stream = archive.extractfile(member)
        assert stream is not None
        with np.load(io.BytesIO(stream.read())) as chunk:
            selected = (chunk["origin"] == 4273) & np.isin(chunk["depth"], [0, 7, 13])
            for key in chunk.files:
                parts.setdefault(key, []).append(chunk[key][selected])
data = {key: np.concatenate(values) for key, values in parts.items()}
regions = {"Global": data["global_lattice"]}
for name, lon, lat in SPATIAL_REGIONS:
    regions[name] = (
        (data["lon"] >= lon.start)
        & (data["lon"] < lon.stop)
        & (data["lat"] >= lat.start)
        & (data["lat"] < lat.stop)
    )
layouts = {}
for name, mask in regions.items():
    for depth in [0, 7, 13]:
        output = (
            Path(__file__).parent
            / "report"
            / f"map-{name.lower().replace(' ', '-')}-depth-{depth:02d}.png"
        )
        layout = render(
            subset(data, mask & (data["depth"] == depth)),
            name,
            depth,
            output,
            dpi=85,
            sample_stride=4 if name == "Global" else 1,
        )

        assert layout is not None
        with Image.open(output) as image:
            pixels = np.asarray(image.convert("RGB"))
        nr, nc = layout["shape"]
        block = layout["pixels_per_sample"]
        for left, top in layout["panel_origins"]:
            raster = pixels[top : top + nr * block, left : left + nc * block]
            cells = raster.reshape(nr, block, nc, block, 3)
            assert np.array_equal(
                cells, np.broadcast_to(cells[:, :1, :, :1], cells.shape)
            ), output
        layouts[output.name] = layout
(Path(__file__).parent / "raster-layout.json").write_text(
    json.dumps(layouts, indent=2) + "\n"
)
