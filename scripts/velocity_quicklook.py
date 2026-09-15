#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render one fixed validation example without waiting for the full evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from samudra.experiments.velocity_transfer.data import VelocitySource, eligible_anchors
from samudra.experiments.velocity_transfer.model import VelocitySamudra, rollout
from samudra.experiments.velocity_transfer.train import precision


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    from samudra_rust_loader import FlatOm4ReadPool

    source = VelocitySource(args.data, FlatOm4ReadPool(4))
    anchors = eligible_anchors(source.times, "validation")
    anchor = next(
        int(i)
        for i in anchors
        if source.times[i + 2] - source.times[i] == np.timedelta64(10, "D")
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    assert config["variant"] == "D0" and config["seed"] == 15
    device = torch.device(args.device)
    model = VelocitySamudra(config["widths"], checkpointing=None).to(device).eval()
    model.load_state_dict(state["model"])
    batch = source.batch(anchor, 2, device)
    print(
        "Loaded checkpoint and validation window; running two forecast steps",
        flush=True,
    )
    with precision(device):
        predictions = rollout(model, batch, "duacs", 2)
    assert torch.isfinite(predictions).all()
    forecast = (predictions[:, 1] * batch["std"] + batch["mean"])[0].cpu().numpy()
    initial = (batch["history"][:, -1] * batch["std"] + batch["mean"])[0].cpu().numpy()
    truth = (batch["targets"][:, 1] * batch["std"] + batch["mean"])[0].cpu().numpy()
    valid = batch["target_valid"][0, 1, 0].cpu().numpy().astype(bool)
    area = batch["area"][0, 0].cpu().numpy().astype(np.float64)
    dates = [
        str(t.astype("datetime64[D]")) for t in source.times[anchor - 3 : anchor + 3]
    ]

    def rmse(value, mask):
        weight = area * mask
        return float(
            np.sqrt(
                (((value - truth).astype(np.float64) ** 2).sum(0) * weight).sum()
                / weight.sum()
            )
        )

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_update": state["update"],
        "config": config,
        "anchor_index": anchor,
        "selection": "First eligible validation anchor with an exact ten-day lead",
        "history_dates": dates[:4],
        "initial_date": dates[3],
        "target_date": dates[5],
        "lead_days": 10,
        "precision": "GPU bfloat16" if device.type == "cuda" else "CPU float32",
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "global_vector_rmse_m_per_s": rmse(forecast, valid),
        "global_persistence_rmse_m_per_s": rmse(initial, valid),
        "valid_cells": int(valid.sum()),
    }
    for region, bounds in {
        "global": (0, 360, -80, 80),
        "gulf_stream": (280, 330, 20, 50),
    }.items():
        x0, x1, y0, y1 = bounds
        xi = np.flatnonzero((source.x >= x0) & (source.x <= x1))
        yi = np.flatnonzero((source.y >= y0) & (source.y <= y1))
        crop = np.ix_(yi, xi)
        masked = valid[crop]
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.6), layout="constrained")
        panels = [initial, forecast, truth, forecast - truth]
        titles = [
            f"Initial observation\n{dates[3]}",
            f"D0 forecast: +10 days\n{dates[5]}",
            f"Observed gold: +10 days\n{dates[5]}",
            "Forecast minus gold",
        ]
        for component, label in enumerate(["Eastward u", "Northward v"]):
            values = np.concatenate(
                [p[component][crop][masked] for p in [initial, truth]]
            )
            limit = max(float(np.quantile(np.abs(values), 0.99)), 0.05)
            for column, (panel, title) in enumerate(zip(panels, titles, strict=True)):
                ax = axes[component, column]
                field = np.where(masked, panel[component][crop], np.nan)
                artist = ax.pcolormesh(
                    source.x[xi],
                    source.y[yi],
                    field,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    shading="auto",
                    rasterized=True,
                )
                ax.set_facecolor("#dddddd")
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                ax.set_xlabel("Longitude (°E)")
                if column == 0:
                    ax.set_ylabel(f"{label}\nLatitude (°N)")
                if component == 0:
                    ax.set_title(title, fontsize=11)
            fig.colorbar(
                artist,
                ax=list(axes[component]),
                label="m/s",
                shrink=0.85,
                extend="both",
            )
        fig.suptitle(
            f"D0 seed 15 · best validation checkpoint (update {state['update']}) · {region.replace('_', ' ')}",
            fontsize=14,
        )
        fig.supxlabel(
            "Shared color scale within each row; limits clip the largest 1% of observed magnitudes. Grey = masked/excluded.\nInitial panel is the latest of four input maps. Single validation case; see metadata.json for inference precision.",
            fontsize=9,
        )
        fig.savefig(
            args.output / f"d0-10day-{region}.jpg",
            dpi=120,
            pil_kwargs={"quality": 65, "optimize": True},
        )
        plt.close(fig)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
