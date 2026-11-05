#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fit and record an observation-derived normalization.

Run from the repository with the data environment:
  python data/prepare_observation_normalization.py PREPARED_ROOT --om4 OM4_DIRECTORY
The May 1993--July 2013 training samples supply temperature, salinity, SSH and
ERA5 statistics over finite wet cells globally, including repeated input histories.
Velocity and T/S below the observed profile use the matching fields from
OM4_means.zarr and OM4_stds.zarr in OM4_DIRECTORY.
Model wind stress and heat flux are normalized separately using OM4 statistics.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr
import zarr  # type: ignore[import-untyped]
from numpy.typing import NDArray

from samudra.observations.archive import (
    NORMALIZATION_POLICY,
    NORMALIZATION_VERSION,
    SPLITS,
    fit_statistics,
)
from samudra.observations.constants import PROFILE_NAMES, STATE_NAMES, SURFACE_NAMES

OBSERVED_STATE_NAMES = frozenset(PROFILE_NAMES) | frozenset(SURFACE_NAMES)
OM4_STATE_NAMES = tuple(
    name for name in STATE_NAMES if name not in OBSERVED_STATE_NAMES
)


def state_sources() -> dict[str, str]:
    """Identify the source of every shared state-normalization channel."""
    sources = dict.fromkeys(STATE_NAMES, "OM4")
    sources.update(dict.fromkeys(PROFILE_NAMES, "IAP"))
    sources.update(dict(zip(SURFACE_NAMES, ("OISST", "DUACS"), strict=True)))
    return sources


def read_om4_statistics(om4: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    """Read finite scalar means/stds for the state fields absent from observations."""
    result: list[dict[str, float]] = []
    for suffix in ("means", "stds"):
        path = Path(om4) / f"OM4_{suffix}.zarr"
        if not path.is_dir():
            raise ValueError(f"Required OM4 normalization store is missing: {path}")
        store = zarr.open_consolidated(str(path), mode="r")
        fields = {}
        for name in OM4_STATE_NAMES:
            if name not in store:
                raise ValueError(f"Missing OM4 {suffix} for unobserved field {name}")
            value = np.asarray(store[name][...])
            if value.size != 1 or not np.isfinite(value).all():
                raise ValueError(f"OM4 {suffix} for {name} must be one finite scalar")
            scalar = float(value.item())
            if suffix == "stds" and scalar <= 0:
                raise ValueError(f"OM4 standard deviation for {name} must be positive")
            fields[name] = scalar
        result.append(fields)
    return result[0], result[1]


def state_statistics(
    statistics: Mapping[str, NDArray],
    om4_means: Mapping[str, float],
    om4_stds: Mapping[str, float],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Combine observed field statistics with OM4 statistics for unobserved fields.

    Surface OISST replaces the profile surface-temperature scale. Shared channel
    names preserve the canonical ordering without importing Torch into preparation.
    """
    mean = {name: om4_means[name] for name in OM4_STATE_NAMES}
    std = {name: om4_stds[name] for name in OM4_STATE_NAMES}
    for channel, name in enumerate(PROFILE_NAMES):
        mean[name] = float(statistics["interior_mean"][channel])
        std[name] = float(statistics["interior_std"][channel])
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
        raise ValueError(
            "Shared state scales require finite means and positive finite standard deviations"
        )
    return means, stds


def prepare_normalization(
    root: str | Path,
    output: str | Path | None = None,
    *,
    om4: str | Path,
) -> None:
    om4 = Path(om4)
    om4_means, om4_stds = read_om4_statistics(om4)
    root = Path(root)
    output = Path(output) if output is not None else root
    output.mkdir(parents=True, exist_ok=True)
    with np.load(root / "grid.npz") as grid:
        fit_statistics(root / "train", dict(grid), output / "statistics.npz")
    with np.load(output / "statistics.npz") as stats:
        mean, std = state_statistics(stats, om4_means, om4_stds)
    stores = [output / "observation_means.zarr", output / "observation_stds.zarr"]
    for path, values in zip(stores, (mean, std), strict=True):
        dataset = xr.Dataset(
            {
                name: xr.DataArray(
                    value, attrs={"normalization_source": state_sources()[name]}
                )
                for name, value in zip(STATE_NAMES, values, strict=True)
            }
        )
        dataset.attrs.update(
            normalization_source="observations for observed fields; OM4 for unobserved fields",
            normalization_version=NORMALIZATION_VERSION,
        )
        dataset.to_zarr(path, mode="w", consolidated=True)

    om4_stores = [om4 / "OM4_means.zarr", om4 / "OM4_stds.zarr"]
    paths = [
        root / "grid.npz",
        output / "statistics.npz",
        *(
            path
            for store in stores + om4_stores
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
                "field_sources": state_sources(),
                "om4_normalization_inputs": {
                    "means": str(om4_stores[0].resolve()),
                    "stds": str(om4_stores[1].resolve()),
                },
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
    parser.add_argument(
        "--om4",
        type=Path,
        required=True,
        help="Directory containing OM4_means.zarr and OM4_stds.zarr",
    )
    args = parser.parse_args()
    prepare_normalization(args.root, args.output, om4=args.om4)
