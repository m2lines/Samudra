# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audited, training-only surface subcell variance supervision."""

import hashlib
import json

import numpy as np
import torch
import zarr  # type: ignore[import-untyped]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VarianceTargets:
    def __init__(self, config, source, prognostic_names):
        root = config.target_root
        ready = json.loads((root / "READY.json").read_text())
        for key, filename in (
            ("manifest", "manifest.json"),
            ("grid", "grid.npz"),
            ("normalization", "normalization.json"),
        ):
            if digest(root / filename) != ready[key + "_sha256"]:
                raise ValueError("KE target metadata hash mismatch: " + filename)
        grid = np.load(root / "grid.npz", allow_pickle=False)
        norm = json.loads((root / "normalization.json").read_text())
        self.dates = [str(t) for t in grid["dates"]]
        self.lookup = {date: i for i, date in enumerate(self.dates)}
        if len(self.lookup) != len(self.dates) or len(self.dates) != ready["frames"]:
            raise ValueError("Duplicated or incomplete KE dates")
        train_dates = {str(t)[:10] for t in source.time.values}
        if not set(self.dates).issubset(train_dates):
            raise ValueError("KE normalization includes dates outside training")
        missing = train_dates - set(self.dates)
        if missing:
            raise ValueError(f"KE targets missing training dates: {sorted(missing)}")
        for expected, actual in zip(
            (grid["lat"], grid["lon"]), source.resolution, strict=True
        ):
            np.testing.assert_allclose(
                expected, actual.cpu().numpy(), atol=1e-5, rtol=0
            )
        store = zarr.open_consolidated(str(root / "targets.zarr"))
        if not store["verified"][:].all():
            raise ValueError("Unverified KE target frames")
        valid = grid["mask"].astype(bool)
        mask = source.masks.prognostic_for_steps(1).cpu().numpy()
        for name in ("uo_0", "vo_0"):
            if (valid & ~mask[list(prognostic_names).index(name)]).any():
                raise ValueError("KE support extends outside coarse wet velocities")
        raw = store["variance"][:]
        if not np.isfinite(raw[:, valid]).all() or (raw[:, valid] < 0).any():
            raise ValueError("Nonfinite or negative wet KE target")
        values = (
            np.log1p(np.where(valid, raw, 0) / norm["variance_scale"])
            - norm["log_mean"]
        ) / norm["log_std"]
        values[:, ~valid] = 0
        if config.mode == "seasonal":
            months = np.array([int(d[5:7]) for d in self.dates])
            for month in range(1, 13):
                selected = months == month
                values[selected] = values[selected].mean(axis=0, dtype=np.float64)
        self.values = torch.from_numpy(values.astype(np.float32))
        weights = grid["area"] * valid
        self.weights = torch.from_numpy((weights / weights.sum()).astype(np.float32))
        self.coefficient = config.coefficient
        self.provenance = dict(
            mode=config.mode,
            coefficient=config.coefficient,
            frames=len(self.dates),
            first=self.dates[0],
            last=self.dates[-1],
            valid_cells=int(valid.sum()),
            metadata=ready,
            values_sha256=hashlib.sha256(self.values.numpy().tobytes()).hexdigest(),
        )

    def attach(self, batch, device):
        if len(batch.label_times) != len(batch):
            raise ValueError("KE supervision requires exact label timestamps")
        targets = []
        for times in batch.label_times:
            indices = np.array([self.lookup[str(t)[:10]] for t in times.flat]).reshape(
                times.shape
            )
            targets.append(self.values[torch.from_numpy(indices)].to(device))
        batch.auxiliary_targets = targets
        batch.auxiliary_weights = self.weights.to(device)
        batch.auxiliary_coefficient = self.coefficient
