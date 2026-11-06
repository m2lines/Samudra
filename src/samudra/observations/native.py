# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation intervals read by the same Rust runtime as native OM4 planes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from samudra.native_reader import PhysicalVariable
from samudra.observations.archive import ObservationArchive
from samudra.observations.constants import OBSERVATION_FORCING_CHANNELS, STEP_DAYS
from samudra.observations.shared_era5 import digest
from samudra.rust_reader import RustIoRuntime


class NativeObservationArchive(ObservationArchive):
    def __init__(
        self,
        daily: str | Path,
        cache: str | Path,
        runtime: RustIoRuntime,
        proof_sha256: str,
        *,
        verify_content: bool = True,
    ) -> None:
        super().__init__(daily)
        cache = Path(cache)
        if digest(cache / "COMPLETE.json") != proof_sha256:
            raise ValueError("Native observation proof changed")
        proof = json.loads((cache / "COMPLETE.json").read_text())
        expected = [
            "sst",
            "adt",
            "ugos",
            "vgos",
            *[f"era5_{i}" for i in range(OBSERVATION_FORCING_CHANNELS)],
        ]
        if (
            proof["version"] != 1
            or proof["interval_days"] != STEP_DAYS
            or proof["stride_days"] != 1
            or proof["channels"] != expected
            or not proof["all_intervals_rust_readback_exact"]
            or digest(self.root / "grid.npz") != proof["source_grid_sha256"]
        ):
            raise ValueError("Native observation interval contract differs")
        if verify_content:
            from samudra.observations.training import archive_identity

            if archive_identity(self.root) != proof["source_identity"]:
                raise ValueError("Native observation source changed")
            for name, sha in proof["files"].items():
                if digest(cache / name) != sha:
                    raise ValueError(
                        f"Native observation cache content changed: {name}"
                    )
        self.first = pd.Timestamp(proof["first_start"])
        self.last = pd.Timestamp(proof["last_start"])
        self.variables = [PhysicalVariable("values", i) for i in range(len(expected))]
        self.reader = runtime.open(cache / "intervals.zarr", self.variables)
        if (
            self.reader.shape[0] != proof["intervals"]
            or (self.last - self.first).days + 1 != proof["intervals"]
        ):
            raise ValueError("Native observation cache time dimension differs")

    def read_intervals(
        self, starts: pd.DatetimeIndex, ends: pd.DatetimeIndex
    ) -> dict[str, NDArray[np.float32]]:
        starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
        if (
            not ((ends - starts) == pd.Timedelta(days=STEP_DAYS)).all()
            or not (starts == starts.normalize()).all()
            or (starts < self.first).any()
            or (starts > self.last).any()
        ):
            raise ValueError(
                "Requested observation interval is outside the exact cache"
            )
        indices = (starts - self.first).days.tolist()
        values = np.empty(
            (len(indices), len(self.variables), *self.reader.shape[1:]), np.float32
        )
        self.reader.read_into(indices, self.variables, values)
        return {"sst": values[:, :1], "adt": values[:, 1:4], "era5": values[:, 4:]}

    def interval(
        self, product: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> NDArray[np.float32]:
        return self.read_intervals(pd.DatetimeIndex([start]), pd.DatetimeIndex([end]))[
            product
        ][0]
