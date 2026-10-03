# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-only targets for subcell variance of five-day mean velocity."""

import json
from pathlib import Path

import numpy as np
import torch
import zarr  # type: ignore[import-untyped]

from samudra.experiments.observation_pilot import digest


class VarianceTargets:
    def __init__(self, root, data, coefficient):
        root = Path(root)
        ready = json.loads((root / "READY.json").read_text())
        for name in ("manifest", "normalization", "grid"):
            file = root / (name + (".npz" if name == "grid" else ".json"))
            if digest(file) != ready[name + "_sha256"]:
                raise ValueError(f"Accessory {name} hash differs")
        grid = np.load(root / "grid.npz", allow_pickle=False)
        norm = json.loads((root / "normalization.json").read_text())
        store = zarr.open_consolidated(str(root / "targets.zarr"))
        if not store["verified"][:].all():
            raise ValueError("Unverified accessory targets")
        dates = list(grid["dates"])
        times = data.trainset.sources[0].time.values
        lookup = {date: i for i, date in enumerate(dates)}
        if len(lookup) != len(dates) or len(dates) != ready["frames"]:
            raise ValueError("Accessory dates are duplicated or incomplete")
        mapping = [lookup[str(t)[:10]] for t in times]
        self.mapping = torch.tensor(mapping, device=data.device)
        np.testing.assert_allclose(
            grid["lat"], data.lat.detach().cpu().numpy(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(
            grid["lon"], data.lon.detach().cpu().numpy(), atol=1e-5, rtol=0
        )
        if grid["mask"].shape != tuple(data.mask.shape[-2:]):
            raise ValueError("Accessory grid shape differs")
        raw = store["variance"][:]
        valid = grid["mask"]
        if not np.isfinite(raw[:, valid]).all() or (raw[:, valid] < 0).any():
            raise ValueError("Nonfinite or negative wet accessory target")
        normalized = (
            np.log1p(np.nan_to_num(raw) / norm["variance_scale"]) - norm["log_mean"]
        ) / norm["log_std"]
        normalized[:, ~valid] = 0
        self.values = torch.as_tensor(
            normalized[:, None], device=data.device, dtype=torch.float32
        )
        weights = grid["area"] * valid
        self.weights = torch.as_tensor(
            weights / weights.sum(), device=data.device, dtype=torch.float32
        )
        self.coefficient = coefficient
        self.loss_records = []

    def loss(self, prediction, ids, lead):
        # The model consumes frames 0..18 and predicts frames 19..24.
        index = torch.tensor(ids, device=self.mapping.device) + 18 + lead
        target = self.values[self.mapping[index]]
        if prediction.shape != target.shape:
            raise ValueError("Accessory prediction/target shapes differ")
        loss = ((prediction - target).square() * self.weights).sum((-2, -1)).mean()
        self.loss_records.append(loss.detach())
        return loss
