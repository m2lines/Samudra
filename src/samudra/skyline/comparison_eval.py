# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed, paired validation locations and full-grid MSE estimates by channel."""

import numpy as np
import torch
import torch.distributed as dist

from .data import CHANNELS, DEPTHS
from .run import to_device


@torch.no_grad()
def evaluate_comparison(
    model, source, config, device, step, rank=0, world=1, vocab=None
):
    model.eval()
    origins = source.origins["validation"]
    origins = origins[
        np.linspace(0, len(origins) - 1, config["validation_dates"], dtype=int)
    ]
    rng = np.random.default_rng(config["validation_seed"])
    # count, full-grid squared error, persistence squared error, area-wet error
    discrete_only = config.get("discrete_readout_only", False)
    sums = np.zeros((77, 12 if vocab is not None else 4), dtype=np.float64)
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
            extra = None
            if vocab is not None:
                logits = prediction
                channels = batch["query_channel"]
                centers = vocab.centers[channels]
                ids = logits.argmax(-1)
                prediction = centers.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
                mean = (
                    prediction
                    if discrete_only
                    else (centers * logits.float().softmax(-1)).sum(-1)
                )
                target_ids = vocab.encode(batch["target"], channels)
                coarse_bins = config.get("model", {}).get("coarse_bins", 0)
                greedy = prediction
                coarse_wrong = torch.zeros_like(prediction)
                fine_wrong = torch.zeros_like(prediction)
                if coarse_bins:
                    fine_bins = logits.shape[-1] // coarse_bins
                    joint = logits.float().unflatten(-1, (coarse_bins, fine_bins))
                    coarse = joint.logsumexp(-1).argmax(-1)
                    branch = joint.gather(
                        -2, coarse[..., None, None].expand(*coarse.shape, 1, fine_bins)
                    ).squeeze(-2)
                    greedy_ids = coarse * fine_bins + branch.argmax(-1)
                    greedy = centers.gather(-1, greedy_ids.unsqueeze(-1)).squeeze(-1)
                    true_coarse = target_ids // fine_bins
                    true_branch = joint.gather(
                        -2,
                        true_coarse[..., None, None].expand(
                            *coarse.shape, 1, fine_bins
                        ),
                    ).squeeze(-2)
                    coarse_wrong = (coarse != true_coarse).float()
                    fine_wrong = (
                        true_branch.argmax(-1) != target_ids % fine_bins
                    ).float()
                extra = (
                    torch.stack(
                        (
                            (mean - batch["target"]).square(),
                            vocab.nll(logits, batch["target"], batch["query_channel"]),
                            vocab.tails(
                                batch["target"], batch["query_channel"]
                            ).float(),
                            (
                                vocab.quantized(batch["target"], batch["query_channel"])
                                - batch["target"]
                            ).square(),
                            (greedy - batch["target"]).square(),
                            coarse_wrong,
                            fine_wrong,
                            (ids != target_ids).float(),
                        ),
                        -1,
                    )
                    .cpu()
                    .numpy()
                )
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
                sums[c, :4] += (
                    1,
                    weight[i] * error[i, q],
                    weight[i] * persistence[i, q],
                    error[i, q],
                )
                if extra is not None:
                    sums[c, 4:] += weight[i] * extra[i, q]
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
        nmse, persistence, area_nmse = sums[c, 1:4] / count
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
        if vocab is not None:
            mean_nmse, nll, tail, quantization = sums[c, 4:8] / count
            rows[-1].update(
                probability_mean_full_grid_nmse=float(mean_nmse),
                probability_mean_wet_cell_rmse=float(
                    np.sqrt(mean_nmse / fraction) * source.std[c]
                ),
                full_grid_nll=float(nll),
                wet_cell_nll=float(nll / fraction),
                wet_cell_tail_fraction=float(tail / fraction),
                quantization_full_grid_nmse=float(quantization),
            )
            greedy, coarse_wrong, fine_wrong, leaf_wrong = sums[c, 8:] / count
            rows[-1].update(
                greedy_full_grid_nmse=float(greedy),
                wet_cell_leaf_error_rate=float(leaf_wrong / fraction),
            )
            if config.get("model", {}).get("coarse_bins", 0):
                rows[-1].update(
                    wet_cell_coarse_error_rate=float(coarse_wrong / fraction),
                    wet_cell_teacher_coarse_fine_error_rate=float(
                        fine_wrong / fraction
                    ),
                )
            if discrete_only:
                rows[-1].pop("probability_mean_full_grid_nmse")
                rows[-1].pop("probability_mean_wet_cell_rmse")
    expected = 19 if source.arm == "temperature" else 77
    if len(rows) != expected:
        raise ValueError("Validation lacks requested channels")
    by_field = {}
    for field in ("uo", "vo", "thetao", "so", "zos"):
        chosen = [r for r in rows if r["channel"].split("_")[0] == field]
        if chosen:
            by_field[field] = float(np.mean([r["full_grid_nmse"] for r in chosen]))
    result = dict(
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
    if vocab is not None:
        for key in (
            "full_grid_nll",
            "probability_mean_full_grid_nmse",
            "quantization_full_grid_nmse",
        ):
            if key not in rows[0]:
                continue
            result[key] = float(np.mean([r[key] for r in rows]))
            result["temperature_" + key] = float(
                np.mean([r[key] for r in rows if r["channel"].startswith("thetao_")])
            )
        result["point_prediction"] = (
            "joint MAP leaf center; greedy coarse/fine metrics reported separately"
            if discrete_only
            else "argmax bin center; probability-mean metrics reported separately"
        )
        result["likelihood"] = (
            "Categorical probability mass, not continuous density; open-ended edge bins"
        )
    return result
