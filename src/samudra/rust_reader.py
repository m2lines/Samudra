# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Rust-specific construction and adaptation of the optional Zarr extension."""

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from samudra.native_reader import PhysicalVariable, PlaneReader


def _load_extension() -> Any:
    try:
        return importlib.import_module("samudra_rust_loader")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Rust data loading requires the optional extension; in a source "
            "checkout run `uv sync --extra rust`, or install the matching "
            "samudra-rust-loader platform wheel."
        ) from error


@dataclass(frozen=True)
class _RustPlaneReader:
    reader: Any
    compact: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.reader.shape

    def read_into(
        self,
        time_indices: list[int],
        variables: Sequence[PhysicalVariable],
        output: np.ndarray,
    ) -> None:
        selectors: list[tuple[str, int | None]] | list[str]
        if self.compact:
            selectors = [(variable.name, variable.level) for variable in variables]
        else:
            selectors = [variable.name for variable in variables]
        self.reader.read_into(time_indices, selectors, output)


class RustIoRuntime:
    """One Rayon pool shared by all Rust readers in this process/rank."""

    def __init__(self, max_concurrent_reads: int) -> None:
        self._read_pool = _load_extension().ZarrReadPool(max_concurrent_reads)

    def open(self, path: Path, variables: Sequence[PhysicalVariable]) -> PlaneReader:
        extension = _load_extension()
        compact = any(variable.level is not None for variable in variables)
        if compact:
            reader = extension.CompactOm4Reader(
                path,
                [(variable.name, variable.level) for variable in variables],
                self._read_pool,
            )
        else:
            reader = extension.FlatOm4Reader(
                path, [variable.name for variable in variables], self._read_pool
            )
        return _RustPlaneReader(reader, compact)
