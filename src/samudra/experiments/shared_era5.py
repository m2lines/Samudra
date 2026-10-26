# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verified ERA5 forcing on the unchanged native OM4 training windows."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class SharedEra5:
    def __init__(self, root, reader, data, proof_sha256, *, full_readback=False):
        root = Path(root)
        proof_path = root / "COMPLETE.json"
        if digest(proof_path) != proof_sha256:
            raise ValueError("ERA5 interval proof changed")
        proof = json.loads(proof_path.read_text())
        if not proof["all_intervals_readback_exact"]:
            raise ValueError("ERA5 interval audit did not pass")
        for path, sha in proof["contracts"].items():
            if digest(path) != sha:
                raise ValueError("Shared ERA5 grid/statistics contract changed")
        for name, sha in proof["files"].items():
            if (full_readback or name == "indices.npy") and digest(root / name) != sha:
                raise ValueError("Shared ERA5 cache changed")
        indices = np.load(root / "indices.npy")
        needed = sorted(
            {i for start, stop, _, _ in reader.windows for i in range(start - 19, stop)}
        )
        if indices.tolist() != needed or proof["windows"] != len(reader.windows):
            raise ValueError("Shared ERA5 changed the OM4 sample pool")
        for index, bounds in zip(indices, proof["bounds"], strict=True):
            expected = [pd.Timestamp(str(t)).isoformat() for t in reader.bounds[index]]
            if expected != bounds:
                raise ValueError("OM4/ERA5 interval bounds differ")
        self.positions = {int(i): j for j, i in enumerate(indices)}
        self.values = np.load(root / "values.npy", mmap_mode="r")
        if self.values.shape != (len(needed), 8, *data.grid["mask"].shape[-2:]):
            raise ValueError("ERA5 forcing shape differs")
        self.mean = data.stats["atmosphere_mean"][None, :, None, None]
        self.std = data.stats["atmosphere_std"][None, :, None, None]

    def load(self, reader, index, data, *, cpu=False):
        start, stop, _, _ = reader.windows[index]
        raw = np.array(
            self.values[[self.positions[i] for i in range(start - 19, stop)]]
        )
        atmosphere = (raw - self.mean) / self.std
        if not np.isfinite(atmosphere).all():
            raise ValueError("Nonfinite shared forcing")
        return (
            torch.as_tensor(atmosphere, dtype=torch.float32)
            if cpu
            else data.tensor(atmosphere)
        )[None]
