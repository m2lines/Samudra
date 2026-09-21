# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming, fixed-climatology spatial and temporal reconstruction diagnostics."""

import csv

import numpy as np
import torch
from torch.nn import functional as F


def masked_box(value, mask, size):
    radius = size // 2

    def average(x):
        x = F.pad(x, (radius, radius, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, radius, radius))
        return F.avg_pool2d(x, size, stride=1)

    mask = mask.to(value.dtype)
    return average(value * mask) / average(mask[None]).clamp_min(1e-12) * mask


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReconstructionDiagnostics:
    def __init__(self, experiment, source, indices):
        self.experiment = experiment
        self.source = source
        self.sums = torch.zeros(
            (5, *experiment.mask.shape), device=experiment.device, dtype=torch.float64
        )
        self.count = torch.zeros((), device=experiment.device, dtype=torch.float64)
        self.scales = torch.zeros(
            (3, 3, 4, experiment.channels),
            device=experiment.device,
            dtype=torch.float64,
        )
        self.regions = {
            "global": torch.ones_like(experiment.lat, dtype=torch.bool),
            "tropics": experiment.lat.abs() <= 20,
            "extratropics": experiment.lat.abs() > 20,
        }
        self.climate = torch.load(
            experiment.pretrain.parent.parent / "initializer/climatology.pt",
            map_location=experiment.device,
            weights_only=False,
        )["monthly"]
        self.snapshot_indices = {indices[0], indices[len(indices) // 2], indices[-1]}
        self.channels = [
            experiment.names.index(n) for n in ["thetao_5", "thetao_9", "so_5", "so_9"]
        ]
        self.snapshots = {
            "latitude": experiment.lat.cpu().numpy(),
            "longitude": experiment.source.resolution[1].cpu().numpy(),
            "channels": np.array([experiment.names[i] for i in self.channels]),
            "depths_m": np.array(experiment.bundle.data_layout.depth_levels),
        }

    def add(self, prediction, truth, ids):
        e = self.experiment
        p, t = prediction.double(), truth.double()
        self.sums += torch.stack((p, t, p * p, t * t, p * t)).sum(1)
        self.count += len(ids)
        months = torch.tensor(
            [self.source.time.values[i + 18].month - 1 for i in ids], device=e.device
        )
        climate = self.climate[months].float()
        pa, ta = prediction.float() - climate, truth.float() - climate
        for s, size in enumerate([0, 3, 9]):
            a, b = (
                (pa, ta)
                if size == 0
                else (
                    pa - masked_box(pa, e.mask, size),
                    ta - masked_box(ta, e.mask, size),
                )
            )
            for r, selector in enumerate(self.regions.values()):
                weight = e.weights * selector[None, :, None]
                denominator = weight.sum((-2, -1)).clamp_min(1e-12)
                moments = torch.stack(
                    (
                        (a * a * weight).sum((-2, -1)) / denominator,
                        (b * b * weight).sum((-2, -1)) / denominator,
                        (a * b * weight).sum((-2, -1)) / denominator,
                        ((a - b).square() * weight).sum((-2, -1)) / denominator,
                    )
                )
                self.scales[r, s] += moments.double().sum(1)
        for j, index in enumerate(ids):
            if index in self.snapshot_indices:
                self.snapshots[f"{index}_date"] = np.array(
                    str(self.source.time.values[index + 18])
                )
                self.snapshots[f"{index}_prediction"] = (
                    prediction[j, self.channels].float().cpu().numpy()
                )
                self.snapshots[f"{index}_truth"] = (
                    truth[j, self.channels].float().cpu().numpy()
                )
                self.snapshots[f"{index}_climatology"] = (
                    climate[j, self.channels].cpu().numpy()
                )

    def finish(self):
        e = self.experiment
        e.reduce(self.sums)
        e.reduce(self.count)
        e.reduce(self.scales)
        np.savez_compressed(e.out / f"snapshots-rank{e.rank}.npz", **self.snapshots)
        if e.rank:
            return
        p, t, pp, tt, pt = self.sums / self.count
        vp, vt = (pp - p * p).clamp_min(0), (tt - t * t).clamp_min(0)
        root = (vp * vt).sqrt()
        fields = {
            "mse": pp + tt - 2 * pt,
            "mean_bias_mse": (p - t).square(),
            "temporal_amplitude_mse": vp + vt - 2 * root,
            "temporal_pattern_mse": 2 * (root - (pt - p * t)),
        }
        rows = []
        for region, selector in self.regions.items():
            weights = e.weights.double() * selector[None, :, None]
            values = {
                name: (
                    (value * weights).sum((-2, -1))
                    / weights.sum((-2, -1)).clamp_min(1e-12)
                )
                .cpu()
                .numpy()
                for name, value in fields.items()
            }
            for c, channel in enumerate(e.names):
                rows.append(
                    {
                        "region": region,
                        "channel": channel,
                        **{name: float(value[c]) for name, value in values.items()},
                        "origins": int(self.count),
                    }
                )
        write_rows(e.out / "temporal-decomposition.csv", rows)
        rows = []
        scale_values = (self.scales / self.count).cpu().numpy()
        for r, region in enumerate(self.regions):
            for s, size in enumerate([0, 3, 9]):
                for c, channel in enumerate(e.names):
                    pp, tt, pt, mse = scale_values[r, s, :, c]
                    rows.append(
                        dict(
                            region=region,
                            box_width_cells=size,
                            channel=channel,
                            predicted_anomaly_second_moment=float(pp),
                            target_anomaly_second_moment=float(tt),
                            anomaly_cross_moment=float(pt),
                            anomaly_mse=float(mse),
                            origins=int(self.count),
                        )
                    )
        write_rows(e.out / "spatial-scales.csv", rows)
