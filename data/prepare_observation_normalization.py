#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fit and record an observation-derived normalization.

Run from the repository with the data environment:
  PYTHONPATH=src python data/prepare_observation_normalization.py PREPARED_ROOT
The May 1993--July 2013 training samples supply temperature, salinity, SSH and
ERA5 statistics over finite wet cells globally, including repeated input histories.
Velocity has no matching observed 3-D field: its state slots use mean zero and
standard deviation one. Deeper T/S slots repeat the deepest observed scales.
Model wind stress and heat flux are normalized separately using OM4 statistics.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from samudra.observations.archive import (
    NORMALIZATION_POLICY,
    NORMALIZATION_VERSION,
    SPLITS,
    fit_statistics,
)
from samudra.observations.constants import (
    MODEL_DEPTHS,
    OBSERVED_DEPTHS,
    PROFILE_NAMES,
    PROFILE_VARIABLES,
    STATE_NAMES,
    SURFACE_NAMES,
)


def state_statistics(
    statistics: Mapping[str, NDArray],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Map observed field names into the complete physical-state channel layout.

    U/V use zero/unit reference scales; deeper T/S repeat the deepest observed
    profile level. Surface OISST replaces the profile surface-temperature scale.
    Shared channel names keep preprocessing independent of Torch's DataLayout
    tensors while using the same canonical field ordering.
    """
    mean = dict.fromkeys(STATE_NAMES, 0.0)
    std = dict.fromkeys(STATE_NAMES, 1.0)
    for channel, name in enumerate(PROFILE_NAMES):
        mean[name] = float(statistics["interior_mean"][channel])
        std[name] = float(statistics["interior_std"][channel])
    for variable in PROFILE_VARIABLES:
        deepest = f"{variable}_{OBSERVED_DEPTHS - 1}"
        for depth in range(OBSERVED_DEPTHS, MODEL_DEPTHS):
            mean[f"{variable}_{depth}"] = mean[deepest]
            std[f"{variable}_{depth}"] = std[deepest]
    for channel, name in enumerate(SURFACE_NAMES):
        mean[name] = float(statistics["surface_mean"][channel])
        std[name] = float(statistics["surface_std"][channel])
    means = np.array([mean[name] for name in STATE_NAMES], dtype=np.float32)
    stds = np.array([std[name] for name in STATE_NAMES], dtype=np.float32)
    if (
        not np.isfinite(means).all()
        or not np.isfinite(stds).all()
        or not (stds > 0).all()
    ):
        raise ValueError("Invalid observation state scales")
    return means, stds


def prepare_normalization(root: str | Path, output: str | Path | None = None) -> None:
    root = Path(root)
    output = Path(output) if output is not None else root
    output.mkdir(parents=True, exist_ok=True)
    with np.load(root / "grid.npz") as grid:
        fit_statistics(root / "train", dict(grid), output / "statistics.npz")
    with np.load(output / "statistics.npz") as stats:
        mean, std = state_statistics(stats)
    stores = [output / "observation_means.zarr", output / "observation_stds.zarr"]
    for path, values in zip(stores, (mean, std), strict=True):
        dataset = xr.Dataset(
            {
                name: xr.DataArray(value)
                for name, value in zip(STATE_NAMES, values, strict=True)
            }
        )
        dataset.attrs.update(
            normalization_source="observation training cohort",
            normalization_version=NORMALIZATION_VERSION,
        )
        dataset.to_zarr(path, mode="w", consolidated=True)

    paths = [
        root / "grid.npz",
        output / "statistics.npz",
        *(
            path
            for store in stores
            for path in sorted(store.rglob("*"))
            if path.is_file()
        ),
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
