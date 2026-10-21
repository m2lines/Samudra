#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fit and record the shared observation-only normalization, separately from training.

Run: uv run python data/prepare_observation_normalization.py PREPARED_ROOT
Only the 243 May 1993--July 2013 training samples enter these statistics. Repeated
history intervals intentionally receive the same weighting as in original U-global.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from samudra.observations.archive import SPLITS, fit_statistics
from samudra.observations.prepare import save_atomic


def state_statistics(statistics):
    """Original U-global conventions for all 77 channels, including unobserved ones."""
    mean, std = np.zeros(77, dtype=np.float32), np.ones(77, dtype=np.float32)
    for offset, group in ((38, 0), (57, 1)):
        mean[offset : offset + 14] = statistics["interior_mean"][
            group * 14 : (group + 1) * 14
        ]
        std[offset : offset + 14] = statistics["interior_std"][
            group * 14 : (group + 1) * 14
        ]
        mean[offset + 14 : offset + 19] = mean[offset + 13]
        std[offset + 14 : offset + 19] = std[offset + 13]
    mean[[38, 76]], std[[38, 76]] = (
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
                "training_split": SPLITS["train"],
                "weighting": "cos(latitude), abs(latitude)<=60, original repeated-history sample distribution",
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
