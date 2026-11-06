# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed, paired validation locations and full-grid MSE estimates by channel."""

import numpy as np
import torch
import torch.distributed as dist

from .data import CHANNELS, DEPTHS
from .run import to_device


@torch.no_grad()
def evaluate_comparison(model, source, config, device, step, rank=0, world=1):
    model.eval()
    origins = source.origins["validation"]
    origins = origins[
        np.linspace(0, len(origins) - 1, config["validation_dates"], dtype=int)
    ]
    rng = np.random.default_rng(config["validation_seed"])
    # count, full-grid squared error, persistence squared error, area-wet error
    sums = np.zeros((77, 4), dtype=np.float64)
    for index, origin in enumerate(origins):
        depths = np.tile(np.arange(19), config["validation_points_per_depth"])
        if index % world != rank:
            rng.random(len(depths))
            continue
        frame = source.read(int(origin))
        for start in range(0, len(depths), config["microbatch"]):
            selected = depths[start : start + config["microbatch"]]
            batch = to_device(
                source.sample(frame, rng, len(selected), config["crop"], selected),
                device,
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                prediction = model(batch)
            error = (prediction - batch["target"]).square().cpu().numpy()
            persistence = (
                (batch["query_current"] - batch["target"]).square().cpu().numpy()
            )
            channels = batch["query_channel"].cpu().numpy()
            query_depth = batch["query_depth"].cpu().numpy()
            valid = batch["query_valid"].cpu().numpy()
            # Whole-column outputs are scored on precisely the same per-depth
            # locations as single-depth outputs, discarding extra predictions.
            valid &= query_depth == selected[:, None]
            rows, cols = (batch[k].cpu().numpy() for k in ("row", "col"))
            weight = source.wet_area[selected] / (
                source.area.size * source.area[rows, cols]
            )
            for i, q in zip(*np.where(valid), strict=True):
                c = channels[i, q]
                sums[c] += (
                    1,
                    weight[i] * error[i, q],
                    weight[i] * persistence[i, q],
                    error[i, q],
                )
    totals = torch.tensor(sums, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals)
    sums = totals.cpu().numpy()
    rows = []
    for c in range(77):
        count = sums[c, 0]
        if count == 0:
            continue
        z = 0 if c == 76 else c % 19
        nmse, persistence, area_nmse = sums[c, 1:] / count
        fraction = float(source.wet[z].mean())
        rows.append(
            dict(
                channel=CHANNELS[c],
                channel_index=c,
                depth=z,
                depth_m=float(DEPTHS[z]),
                count=int(count),
                wet_fraction=fraction,
                full_grid_nmse=float(nmse),
                persistence_full_grid_nmse=float(persistence),
                wet_cell_rmse=float(np.sqrt(nmse / fraction) * source.std[c]),
                area_wet_rmse=float(np.sqrt(area_nmse) * source.std[c]),
            )
        )
    expected = 19 if source.arm == "temperature" else 77
    if len(rows) != expected:
        raise ValueError("Validation lacks requested channels")
    by_field = {}
    for field in ("uo", "vo", "thetao", "so", "zos"):
        chosen = [r for r in rows if r["channel"].split("_")[0] == field]
        if chosen:
            by_field[field] = float(np.mean([r["full_grid_nmse"] for r in chosen]))
    return dict(
        step=step,
        origins=origins.tolist(),
        per_channel=rows,
        per_field_nmse=by_field,
        full_grid_nmse=float(np.mean([r["full_grid_nmse"] for r in rows])),
        temperature_full_grid_nmse=by_field["thetao"],
        persistence_full_grid_nmse=float(
            np.mean([r["persistence_full_grid_nmse"] for r in rows])
        ),
        metric="Equal normalized-channel masked MSE, full rectangular grid denominator",
    )
