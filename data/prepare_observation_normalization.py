#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fit and record the shared observation-only normalization, separately from training.

Run from the repository with the existing data environment:
  PYTHONPATH=src data/.venv/bin/python data/prepare_observation_normalization.py PREPARED_ROOT
Only the 243 May 1993--July 2013 training samples enter these statistics. Repeated
history intervals retain the original weighting; finite wet cells at ALL latitudes
contribute. Historical latitude-restricted artifacts must be regenerated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from samudra.observations.archive import (
    NORMALIZATION_POLICY,
    NORMALIZATION_VERSION,
    SPLITS,
    fit_statistics,
)
from samudra.observations.constants import (
    MODEL_DEPTHS,
    OBSERVED_DEPTHS,
    SALINITY_OFFSET,
    STATE_CHANNELS,
    SURFACE_CHANNELS,
    TEMPERATURE_OFFSET,
)
from samudra.observations.prepare import save_atomic


def state_statistics(statistics):
    """Original U-global conventions for all 77 channels, including unobserved ones."""
    mean = np.zeros(STATE_CHANNELS, dtype=np.float32)
    std = np.ones(STATE_CHANNELS, dtype=np.float32)
    for offset, group in ((TEMPERATURE_OFFSET, 0), (SALINITY_OFFSET, 1)):
        mean[offset : offset + OBSERVED_DEPTHS] = statistics["interior_mean"][
            group * OBSERVED_DEPTHS : (group + 1) * OBSERVED_DEPTHS
        ]
        std[offset : offset + OBSERVED_DEPTHS] = statistics["interior_std"][
            group * OBSERVED_DEPTHS : (group + 1) * OBSERVED_DEPTHS
        ]
        mean[offset + OBSERVED_DEPTHS : offset + MODEL_DEPTHS] = mean[
            offset + OBSERVED_DEPTHS - 1
        ]
        std[offset + OBSERVED_DEPTHS : offset + MODEL_DEPTHS] = std[
            offset + OBSERVED_DEPTHS - 1
        ]
    mean[SURFACE_CHANNELS], std[SURFACE_CHANNELS] = (
        statistics["surface_mean"],
        statistics["surface_std"],
    )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid observation state scales")
    return mean, std


def prepare_normalization(root, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root
    output.mkdir(parents=True, exist_ok=True)
    with np.load(root / "grid.npz") as grid:
        fit_statistics(root / "train", dict(grid), output / "statistics.npz")
    with np.load(output / "statistics.npz") as stats:
        mean, std = state_statistics(stats)
    save_atomic(output / "normalization.npz", state_mean=mean, state_std=std)
    paths = [
        root / "grid.npz",
        output / "statistics.npz",
        output / "normalization.npz",
        *sorted((root / "train").glob("*.npz")),
    ]
    hashes = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    (output / "normalization.json").write_text(
        json.dumps(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "normalization_policy": NORMALIZATION_POLICY,
                "training_split": SPLITS["train"],
                "unobserved_velocity": "mean=0,std=1",
                "deep_ts": "repeat deepest observed mean/std",
                "surface_temperature": "OISST overrides profile thetao_0",
                "sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepare_normalization(args.root, args.output)
