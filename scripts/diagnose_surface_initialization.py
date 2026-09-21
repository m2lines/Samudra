#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Held-out, inference-only initialization diagnostics using pinned wave-two models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from samudra.experiments.surface_adaptation import Adaptation
from samudra.experiments.surface_state import channel_mse
from samudra.experiments.surface_wave import batches
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


def masked_box(value, mask, size):
    """Wet-normalized box average: periodic longitude, truncated latitude edges."""
    mask = mask.to(dtype=value.dtype)
    radius = size // 2

    def average(x):
        x = F.pad(x, (radius, radius, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, radius, radius))
        return F.avg_pool2d(x, size, stride=1)

    return average(value * mask) / average(mask).clamp_min(1e-12) * mask


def temporal_decomposition(sums, count):
    """Population MSE = squared mean bias + amplitude mismatch + covariance mismatch."""
    p, t, p2, t2, pt = sums / count
    vp = (p2 - p.square()).clamp_min(0)
    vt = (t2 - t.square()).clamp_min(0)
    scale = (vp * vt).sqrt()
    bias = (p - t).square()
    amplitude = (vp.sqrt() - vt.sqrt()).square()
    pattern = 2 * (scale - (pt - p * t))
    mse = p2 + t2 - 2 * pt
    torch.testing.assert_close(bias + amplitude + pattern, mse, atol=1e-10, rtol=1e-8)
    return torch.stack((mse, bias, amplitude, pattern, vp, vt), 0)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--wave1-root", default="/scratch/jr7309/runs/2026-09-18-surface-wave1"
    )
    parser.add_argument(
        "--selected-root",
        default="/scratch/jr7309/runs/2026-09-20-surface-wave2-rate-control/E",
    )
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    cfg = dict(
        output=str(out),
        name=out.name,
        data_root="/scratch/jr7309/data/om4_onedeg_v3",
        seed=1729,
        widths=[128, 192, 256, 384],
        readers=8,
        batch_size=2,
        device_cache=True,
        device_cache_reserve_gib=24,
        wandb_mode="disabled",
        wave1_root=args.wave1_root,
    )
    experiment = Adaptation(SimpleNamespace(**cfg))
    checkpoint = Path(args.selected_root) / "adapt-best.pt"
    audit = json.loads((Path(args.selected_root) / "checkpoint-audit.json").read_text())
    assert file_hash(checkpoint) == audit["selected"]["sha256"]
    selected = experiment.make_model(checkpoint).eval().requires_grad_(False)
    original = experiment.initializer.eval().requires_grad_(False)
    original.load_state_dict(
        torch.load(
            experiment.initializer_path,
            map_location=experiment.device,
            weights_only=False,
        )["model"]
    )
    climatology = torch.load(
        experiment.climatology_path, map_location=experiment.device, weights_only=False
    )["monthly"]
    source = native_om4_source(
        experiment.bundle.inference_source,
        LocalLocation(path=Path(cfg["data_root"]) / "OM4.zarr"),
        create_rust_io_runtime(8),
    )
    dataset = experiment.dataset(source)
    indices = list(range(0, len(dataset), 6))
    assert len(indices) == 99
    sampler = batches(indices, 2)
    regions = {
        "global": torch.ones_like(experiment.lat, dtype=torch.bool),
        "tropics": experiment.lat.abs() <= 20,
        "extratropics": experiment.lat.abs() > 20,
    }
    weights = {r: experiment.weights * m[None, :, None] for r, m in regions.items()}
    moments = {
        m: torch.zeros(
            (5, *experiment.mask.shape), device=experiment.device, dtype=torch.float64
        )
        for m in ["original", "E"]
    }
    scale_sums: dict[tuple[str, str, int], torch.Tensor] = {}
    origin_rows = []
    snapshot = {
        "latitude": experiment.lat.cpu().numpy(),
        "longitude": experiment.source.resolution[1].cpu().numpy(),
        "depths_m": np.array(experiment.bundle.data_layout.depth_levels),
    }
    snapshot_channels = ["thetao_5", "thetao_9", "so_5", "so_9"]
    snapshot_ids = [indices[0], indices[len(indices) // 2], indices[-1]]
    count = 0
    with torch.no_grad():
        for ids, batch in zip(
            sampler, experiment.loader(dataset, sampler), strict=True
        ):
            history, _, context, labels = experiment.inputs(batch, dataset, ids)
            truth = history.reshape(
                len(ids), 6, experiment.channels, *history.shape[-2:]
            )[:, -1].float()
            months = [source.time.values[i + 5].month - 1 for i in ids]
            climate = climatology[
                torch.tensor(months, device=experiment.device)
            ].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = {
                    "original": original(history, context, experiment.mask)[:, -1],
                    "E": selected.initializer(history, context, experiment.mask)[:, -1],
                }
            for model, p in pred.items():
                p, t = p.double(), truth.double()
                moments[model] += torch.stack((p, t, p * p, t * t, p * t)).sum(1)
                pa, ta = p.float() - climate, truth - climate
                for scale in [0, 3, 9]:
                    a, b = (
                        (pa, ta)
                        if scale == 0
                        else (
                            pa - masked_box(pa, experiment.mask, scale),
                            ta - masked_box(ta, experiment.mask, scale),
                        )
                    )
                    for region, w in weights.items():
                        denominator = w.sum((-2, -1)).clamp_min(1e-12)
                        vals = torch.stack(
                            (
                                (a * a * w).sum((-2, -1)) / denominator,
                                (b * b * w).sum((-2, -1)) / denominator,
                                (a * b * w).sum((-2, -1)) / denominator,
                                ((a - b).square() * w).sum((-2, -1)) / denominator,
                            )
                        )
                        key = model, region, scale
                        scale_sums[key] = scale_sums.get(
                            key, torch.zeros_like(vals[:, 0], dtype=torch.float64)
                        ) + vals.double().sum(1)
            estimates = {
                **pred,
                "climatology": climate,
                "truth": truth,
                "blur3_truth": masked_box(truth, experiment.mask, 3),
                "blur9_truth": masked_box(truth, experiment.mask, 9),
            }
            for model, p in estimates.items():
                for region, w in weights.items():
                    # Lead zero is contemporaneous reconstruction. Other leads hold t0 fixed.
                    target = torch.cat((truth[:, None], labels), 1)
                    mse = channel_mse(p[:, None], target, w).cpu().numpy()
                    for bi, index in enumerate(ids):
                        for li, lead in enumerate([0, 5, 10, 15, 20, 25, 30]):
                            for ci, name in enumerate(experiment.names):
                                origin_rows.append(
                                    dict(
                                        model=model,
                                        region=region,
                                        origin_index=index,
                                        origin_time=str(source.time.values[index + 5]),
                                        lead_days=lead,
                                        channel=name,
                                        normalized_mse=float(mse[bi, li, ci]),
                                    )
                                )
            for bi, index in enumerate(ids):
                if index in snapshot_ids:
                    for model, p in {
                        "truth": truth,
                        "climatology": climate,
                        **pred,
                    }.items():
                        for name in snapshot_channels:
                            ci = experiment.names.index(name)
                            field = p[bi, ci].float().cpu().numpy()
                            snapshot[f"{index}_{model}_{name}"] = np.where(
                                experiment.mask[ci].cpu().numpy(), field, np.nan
                            )
            count += len(ids)
            if count % 20 == 0 or count == 99:
                experiment.emit({"event": "diagnostic_progress", "origins": count})
    write_csv(out / "persistence-origins.csv", origin_rows)
    temporal = []
    for model, sums in moments.items():
        fields = temporal_decomposition(sums, count)
        for region, w in weights.items():
            values = (fields * w).sum((-2, -1)) / w.sum((-2, -1)).clamp_min(1e-12)
            for ci, name in enumerate(experiment.names):
                temporal.append(
                    dict(
                        model=model,
                        region=region,
                        channel=name,
                        **{
                            k: float(values[i, ci])
                            for i, k in enumerate(
                                [
                                    "mse",
                                    "mean_bias_mse",
                                    "amplitude_mse",
                                    "pattern_mse",
                                    "prediction_temporal_variance",
                                    "target_temporal_variance",
                                ]
                            )
                        },
                    )
                )
    write_csv(out / "temporal-decomposition.csv", temporal)
    scales = []
    for (model, region, scale), sums in scale_sums.items():
        for ci, name in enumerate(experiment.names):
            scales.append(
                dict(
                    model=model,
                    region=region,
                    box_width_cells=scale,
                    channel=name,
                    **{
                        k: float(sums[i, ci] / count)
                        for i, k in enumerate(
                            [
                                "prediction_anomaly_energy",
                                "target_anomaly_energy",
                                "cross_anomaly_moment",
                                "anomaly_mse",
                            ]
                        )
                    },
                )
            )
    write_csv(out / "spatial-scales.csv", scales)
    np.savez_compressed(out / "snapshots.npz", **snapshot)
    metadata = dict(
        origins=count,
        origin_indices=indices,
        snapshot_indices=snapshot_ids,
        snapshot_channels=snapshot_channels,
        selected_checkpoint=str(checkpoint),
        selected_checkpoint_sha256=file_hash(checkpoint),
        original_initializer_sha256=file_hash(experiment.initializer_path),
        diagnostic_script_sha256=file_hash(__file__),
        training_producer=__import__("os").environ.get("SAMUDRA_CODE_COMMIT"),
        notes="No fitting or checkpoint selection. Last reconstructed frame at t0; gold initialization in existing forecast uses two true history frames. Spatial scales use monthly-training-climatology anomalies and wet-normalized box high-pass filters in grid cells, not isotropic physical wavelengths. Temporal components use population moments across all 99 origins at each wet cell; mean bias may include persistent spatial structure, pattern includes correlation mismatch and is not uniquely position error.",
    )
    (out / "COMPLETE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    experiment.emit({"event": "diagnostic_complete", "origins": count})
    if experiment.run:
        experiment.run.finish()


if __name__ == "__main__":
    main()
