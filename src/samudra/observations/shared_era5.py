# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""ERA5 interval adapter, adapted from the latent-diffusion experiment.

The cache stores all eight raw fields. Both sources apply the observation-fitted
atmospheric scales at read time; no learned three-field projection is involved.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def required_indices(reader):
    return sorted(
        {i for start, stop, _, _ in reader.windows for i in range(start - 19, stop)}
    )


class SharedEra5:
    """Require exact native intervals, complete cohort and verified raw content."""

    def __init__(self, root, reader, data, proof_sha256):
        root = Path(root)
        proof_path = root / "COMPLETE.json"
        if digest(proof_path) != proof_sha256:
            raise ValueError("ERA5 interval proof changed")
        proof = json.loads(proof_path.read_text())
        if not proof["all_intervals_readback_exact"]:
            raise ValueError("ERA5 interval audit did not pass")
        for path, sha in proof["contracts"].items():
            if digest(path) != sha:
                raise ValueError("Shared ERA5 source contract changed")
        for name, sha in proof["files"].items():
            if digest(root / name) != sha:
                raise ValueError("Shared ERA5 cache content changed")
        indices = np.load(root / "indices.npy")
        if indices.tolist() != required_indices(reader) or proof["windows"] != len(
            reader.windows
        ):
            raise ValueError("Shared ERA5 changed the OM4 sample pool")
        for index, bounds in zip(indices, proof["bounds"], strict=True):
            expected = [pd.Timestamp(str(t)).isoformat() for t in reader.bounds[index]]
            if expected != bounds:
                raise ValueError("OM4/ERA5 interval bounds differ")
        self.positions = {int(i): j for j, i in enumerate(indices)}
        self.values = np.load(root / "values.npy", mmap_mode="r")
        if self.values.shape != (len(indices), 8, *data.grid["mask"].shape[-2:]):
            raise ValueError("ERA5 forcing shape differs")
        self.mean = data.stats["atmosphere_mean"][None, :, None, None]
        self.std = data.stats["atmosphere_std"][None, :, None, None]

    def load(self, start, stop):
        raw = np.asarray(self.values[[self.positions[i] for i in range(start, stop)]])
        atmosphere = (raw - self.mean) / self.std
        if not np.isfinite(atmosphere).all():
            raise ValueError("Nonfinite shared ERA5 forcing")
        return atmosphere.astype(np.float32)
