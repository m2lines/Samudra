# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Save per-date, per-region velocity forecasts and matched cheap baseline scores."""

import argparse
import csv
import json
import time
from pathlib import Path

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
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--linear-damping", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.stride < 1 or not 0 <= args.linear_damping <= 1:
        parser.error("Require positive stride and damping in [0,1]")
    start = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    from samudra_rust_loader import FlatOm4ReadPool

    source = VelocitySource(args.data, FlatOm4ReadPool(4))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = (
        VelocitySamudra(
            config["widths"], checkpointing=None, use_geometry=config["variant"] != "D4"
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(state["model"])
    anchors = eligible_anchors(source.times, args.split)[:: args.stride]
    if not len(anchors):
        raise ValueError("No complete evaluation windows")
    lat, lon = np.meshgrid(source.y, source.x, indexing="ij")
    regions = {
        "global_outside_5deg": np.ones_like(lat, dtype=bool),
        "gulf_stream": (lat >= 20) & (lat <= 50) & (lon >= 280) & (lon <= 330),
        "kuroshio": (lat >= 20) & (lat <= 45) & (lon >= 120) & (lon <= 180),
        "southern_ocean": (lat >= -60) & (lat <= -35),
        "north_pacific_gyre": (lat >= 15) & (lat <= 35) & (lon >= 200) & (lon <= 240),
    }
    args.output.mkdir(parents=True)
    fields = [
        "anchor",
        "target",
        "lead_step",
        "actual_lead_days",
        "region",
        "method",
        "weighted_squared_error",
        "weight",
        "vector_rmse_m_per_s",
        "bias_u_m_per_s",
        "bias_v_m_per_s",
        "valid_cell_count",
    ]
    with (args.output / "scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for n, anchor in enumerate(anchors):
            batch = source.batch(int(anchor), 6, device)
            with precision(device):
                predicted = rollout(model, batch, "duacs", 6)
            predicted = (
                predicted.float() * batch["std"][:, None] + batch["mean"][:, None]
            )
            truth = batch["targets"] * batch["std"][:, None] + batch["mean"][:, None]
            history = batch["history"] * batch["std"][:, None] + batch["mean"][:, None]
            leads = batch["day_offsets"][0, 4:]
            velocity_trend = (history[:, -1] - history[:, -2]) / (
                -batch["day_offsets"][0, 2]
            )
            forecasts = {
                "samudra": predicted,
                "persistence": history[:, -1:].expand_as(predicted),
                "damped_linear": history[:, -1:]
                + args.linear_damping
                * leads[None, :, None, None, None]
                * velocity_trend[:, None],
            }
            months = (
                source.times[anchor + 1 : anchor + 7]
                .astype("datetime64[M]")
                .astype(int)
                % 12
            )
            forecasts["seasonal_climatology"] = torch.from_numpy(
                source.monthly_mean[months]
            ).to(device)[None]
            for step in (1, 2, 4, 6):
                for region, region_mask in regions.items():
                    weights = (
                        batch["target_valid"][:, step - 1].float()
                        * batch["area"]
                        * torch.from_numpy(region_mask).to(device)
                    )
                    denom = weights.sum().item()
                    if denom <= 0:
                        continue
                    for method, forecast in forecasts.items():
                        error = forecast[:, step - 1] - truth[:, step - 1]
                        squared_error = (error.square() * weights).sum().item()
                        bias = (error * weights).sum(
                            dim=(0, 2, 3)
                        ).cpu().numpy() / denom
                        writer.writerow(
                            {
                                "anchor": str(source.times[anchor]),
                                "target": str(source.times[anchor + step]),
                                "lead_step": step,
                                "actual_lead_days": float(leads[step - 1]),
                                "region": region,
                                "method": method,
                                "weighted_squared_error": squared_error,
                                "weight": denom,
                                "vector_rmse_m_per_s": np.sqrt(squared_error / denom),
                                "bias_u_m_per_s": bias[0],
                                "bias_v_m_per_s": bias[1],
                                "valid_cell_count": int((weights > 0).sum()),
                            }
                        )
            stream.flush()
            print(
                json.dumps(
                    {
                        "evaluated": n + 1,
                        "total": len(anchors),
                        "anchor": str(source.times[anchor]),
                    }
                ),
                flush=True,
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": state["update"],
        "split": args.split,
        "anchors": len(anchors),
        "linear_damping": args.linear_damping,
        "seconds": time.monotonic() - start,
        "gpu_hours": (time.monotonic() - start) / 3600 if device.type == "cuda" else 0,
        "config": config,
        "complete": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
