# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render all twelve depth/region comparisons from saved point predictions."""

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from samudra.metrics.spectra import SPATIAL_REGIONS
from samudra.skyline.data import DEPTHS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=Path)
args = parser.parse_args()
root = Path(__file__).parent
parts: dict[str, list] = {}
with tarfile.open(args.archive) as archive:
    for member in archive.getmembers():
        if not member.name.endswith(".npz"):
            continue
        stream = archive.extractfile(member)
        assert stream is not None
        with np.load(io.BytesIO(stream.read())) as chunk:
            keep = (chunk["origin"] == 4273) & np.isin(chunk["depth"], [0, 7, 13])
            for key in (
                "row",
                "col",
                "lat",
                "lon",
                "depth",
                "area",
                "global_lattice",
                "gold_c",
                "regression_c",
                "samudra_c",
            ):
                parts.setdefault(key, []).append(chunk[key][keep])
data = {key: np.concatenate(values) for key, values in parts.items()}
regions = {"Global": data["global_lattice"]}
for name, lon, lat in SPATIAL_REGIONS:
    regions[name] = (
        (data["lon"] >= lon.start)
        & (data["lon"] < lon.stop)
        & (data["lat"] >= lat.start)
        & (data["lat"] < lat.stop)
    )
with (root / "report/per-date.csv").open() as csv_stream:
    reference = {
        (int(r["depth"]), r["region"]): r
        for r in csv.DictReader(csv_stream)
        if int(r["origin"]) == 4273
    }

# Each sample has the same exact 2x2 pixel footprint, including across rows.
# Regional panels are consequently smaller than the global panels.
width, height, dpi = 4420, 4300, 100
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
fig.suppressComposite = True
plt.rcParams.update({"font.size": 14})
fig.text(
    0.5,
    0.989,
    "Temperature · +5 days · 16 → 21 July 2016",
    ha="center",
    va="top",
    fontsize=25,
)
fig.text(
    0.5,
    0.976,
    "Initial-run weights without EMA · OM4 is a model reference · all 19 depths evaluated; three shown",
    ha="center",
    va="top",
    fontsize=17,
)
centers = [620 + 770 * i for i in range(5)]
for center, title in zip(
    centers,
    ["OM4", "Skyline regression", "Samudra", "Skyline error", "Samudra error"],
    strict=True,
):
    fig.text(center / width, 0.958, title, ha="center", fontsize=21, weight="bold")

records = []
top = 230
for depth in [0, 7, 13]:
    for region, mask in regions.items():
        d = {key: value[mask & (data["depth"] == depth)] for key, value in data.items()}
        stride = 4 if region == "Global" else 1
        rr = (d["row"] - d["row"].min()) // stride
        cc = (d["col"] - d["col"].min()) // stride
        shape = (int(rr.max()) + 1, int(cc.max()) + 1)
        assert len(np.unique(rr * shape[1] + cc)) == len(rr)
        ih, iw = shape[0] * 2, shape[1] * 2
        row_height = 435 if region == "Global" else 280
        image_top = top + 55
        if region != "Global":
            image_top += 15
        bottom = height - image_top - ih
        temperatures = [d[k] for k in ("gold_c", "regression_c", "samudra_c")]
        errors = [
            d[k].astype(np.float64) - d["gold_c"] for k in ("regression_c", "samudra_c")
        ]
        tmin, tmax = np.quantile(np.concatenate(temperatures), [0.01, 0.99])
        emax = float(np.quantile(np.abs(np.concatenate(errors)), 0.99))
        norms = [Normalize(tmin, tmax)] * 3 + [Normalize(-emax, emax)] * 2
        weights = d["area"].astype(np.float64)
        rmses = [float(np.sqrt(np.average(e * e, weights=weights))) for e in errors]
        for model, rmse in zip(("regression", "samudra"), rmses, strict=True):
            assert np.isclose(
                rmse, float(reference[(depth, region)][model + "_rmse_c"]), rtol=1e-6
            )
        fig.text(
            25 / width,
            (height - image_top - ih / 2) / height,
            f"{region}\n{DEPTHS[depth]:g} m\n"
            + ("1° lattice" if region == "Global" else "¼° grid"),
            va="center",
            fontsize=18,
        )
        row: dict[str, Any] = dict(
            region=region,
            depth_m=float(DEPTHS[depth]),
            points=len(rr),
            regression_rmse_c=rmses[0],
            samudra_rmse_c=rmses[1],
            temperature_limits_c=[float(tmin), float(tmax)],
            error_limit_c=emax,
            shape=shape,
            pixels_per_sample=2,
            panel_origins=[],
        )
        for col, (values, norm) in enumerate(
            zip(temperatures + errors, norms, strict=True)
        ):
            left = centers[col] - iw // 2
            raster = np.full(shape, np.nan)
            raster[rr, cc] = values
            cmap = plt.get_cmap("viridis" if col < 3 else "RdBu_r")
            rgba = cmap(norm(np.ma.masked_invalid(raster)), bytes=True)
            fig.figimage(
                rgba.repeat(2, 0).repeat(2, 1),
                xo=left,
                yo=bottom,
                origin="lower",
                interpolation="none",
                resample=False,
            )
            row["panel_origins"].append([left, image_top])
            if col >= 3:
                fig.text(
                    centers[col] / width,
                    (height - image_top + 18) / height,
                    f"RMSE {rmses[col - 3]:.4f} °C",
                    ha="center",
                    fontsize=17,
                )
            # Labels outside the image cannot antialias onto sample pixels.
            fig.text(
                left / width,
                (bottom - 23) / height,
                f"{d['lon'].min():.1f}°E",
                ha="left",
                va="top",
                fontsize=12,
            )
            fig.text(
                (left + iw) / width,
                (bottom - 23) / height,
                f"{d['lon'].max():.1f}°E",
                ha="right",
                va="top",
                fontsize=12,
            )
        for x, norm, colorbar_cmap, label in [
            (4090, norms[0], "viridis", "Temperature °C"),
            (4270, norms[3], "RdBu_r", "Prediction − OM4 °C"),
        ]:
            bar_height = min(ih, 210)
            ax = fig.add_axes(
                (
                    x / width,
                    (bottom + (ih - bar_height) / 2) / height,
                    18 / width,
                    bar_height / height,
                )
            )
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=colorbar_cmap),
                cax=ax,
                label=label,
                extend="both",
            )
            ax.tick_params(labelsize=12)
        records.append(row)
        top += row_height
fig.text(
    0.5,
    0.025,
    "RMSE: native-cell-area weighted over the displayed wet samples, using unclipped errors. Error = prediction − OM4.",
    ha="center",
    fontsize=17,
)
fig.text(
    0.5,
    0.014,
    "Per row: three temperature panels share pooled 1st–99th percentile limits; two error panels share pooled ±99th percentile |error| limits.",
    ha="center",
    fontsize=17,
)
fig.text(
    0.5,
    0.003,
    "Every sample is a solid 2 × 2 pixel block, without interpolation. Blank cells are land or missing samples; regional panels contain fewer samples.",
    ha="center",
    fontsize=17,
)
output = root / "report/comparison-all-depths-regions.png"
fig.savefig(output, dpi=dpi)
plt.close(fig)
with Image.open(output) as image:
    pixels = np.asarray(image.convert("RGB"))
for row in records:
    nr, nc = row["shape"]
    for left, top in row["panel_origins"]:
        cells = pixels[top : top + nr * 2, left : left + nc * 2].reshape(
            nr, 2, nc, 2, 3
        )
        assert np.array_equal(cells, np.broadcast_to(cells[:, :1, :, :1], cells.shape))
(root / "comparison-metrics.json").write_text(json.dumps(records, indent=2) + "\n")
print(f"{output}: verified all 60 panels and 24 RMSE labels against per-date.csv")
