# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare xgcm face-boundary operations on source and Icechunk pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import icechunk as ic  # type: ignore[import-not-found]
import numpy as np
import xarray as xr
import xgcm  # type: ignore[import-untyped]

FACE_CONNECTIONS = {
    "face": {
        0: {"X": ((12, "Y", False), (3, "X", False)), "Y": (None, (1, "Y", False))},
        1: {
            "X": ((11, "Y", False), (4, "X", False)),
            "Y": ((0, "Y", False), (2, "Y", False)),
        },
        2: {
            "X": ((10, "Y", False), (5, "X", False)),
            "Y": ((1, "Y", False), (6, "X", False)),
        },
        3: {"X": ((0, "X", False), (9, "Y", False)), "Y": (None, (4, "Y", False))},
        4: {
            "X": ((1, "X", False), (8, "Y", False)),
            "Y": ((3, "Y", False), (5, "Y", False)),
        },
        5: {
            "X": ((2, "X", False), (7, "Y", False)),
            "Y": ((4, "Y", False), (6, "Y", False)),
        },
        6: {
            "X": ((2, "Y", False), (7, "X", False)),
            "Y": ((5, "Y", False), (10, "X", False)),
        },
        7: {
            "X": ((6, "X", False), (8, "X", False)),
            "Y": ((5, "X", False), (10, "Y", False)),
        },
        8: {
            "X": ((7, "X", False), (9, "X", False)),
            "Y": ((4, "X", False), (11, "Y", False)),
        },
        9: {"X": ((8, "X", False), None), "Y": ((3, "X", False), (12, "Y", False))},
        10: {
            "X": ((6, "Y", False), (11, "X", False)),
            "Y": ((7, "Y", False), (2, "X", False)),
        },
        11: {
            "X": ((10, "X", False), (12, "X", False)),
            "Y": ((8, "Y", False), (1, "X", False)),
        },
        12: {"X": ((11, "X", False), None), "Y": ((9, "Y", False), (0, "X", False))},
    }
}


def grid(dataset: xr.Dataset) -> xgcm.Grid:
    return xgcm.Grid(
        dataset,
        coords={
            "X": {"center": "i", "left": "i_g"},
            "Y": {"center": "j", "left": "j_g"},
        },
        periodic=False,
        face_connections=FACE_CONNECTIONS,
        autoparse_metadata=False,
    )


def boundary_values(array: xr.DataArray, axis: str) -> np.ndarray:
    candidates = ("i_g", "i") if axis == "X" else ("j_g", "j")
    dimension = next(candidate for candidate in candidates if candidate in array.dims)
    return np.asarray(array.isel({dimension: [0, 4319]}).values)


def compare(source: xr.DataArray, pilot: xr.DataArray, axis: str) -> int:
    expected = boundary_values(source, axis)
    actual = boundary_values(pilot, axis)
    populated = np.isfinite(actual)
    if not populated.any():
        raise AssertionError(f"no populated {axis} face-boundary results")
    np.testing.assert_equal(actual[populated], expected[populated])
    return int(populated.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("pilot_repo", type=Path)
    parser.add_argument("pilot_result", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.pilot_result.read_text())["snapshot"]
    repo = ic.Repository.open(ic.local_filesystem_storage(str(args.pilot_repo)))
    pilot = xr.open_zarr(
        repo.readonly_session(snapshot_id=snapshot).store,
        consolidated=False,
    ).isel(time=0, k=0)
    source = xr.open_zarr(args.source, consolidated=False).isel(time=0, k=0)
    source_grid = grid(source)
    pilot_grid = grid(pilot)

    result: dict[str, Any] = {"scalar": {}, "vector": {}}
    for axis in ("X", "Y"):
        source_diff = source_grid.diff(source.Theta, axis, boundary="fill")
        pilot_diff = pilot_grid.diff(pilot.Theta, axis, boundary="fill")
        result["scalar"][axis] = compare(source_diff, pilot_diff, axis)

    source_vector = {
        "X": source_grid.diff(
            source.U, "X", boundary="fill", other_component={"Y": source.V}
        ),
        "Y": source_grid.diff(
            source.V, "Y", boundary="fill", other_component={"X": source.U}
        ),
    }
    pilot_vector = {
        "X": pilot_grid.diff(
            pilot.U, "X", boundary="fill", other_component={"Y": pilot.V}
        ),
        "Y": pilot_grid.diff(
            pilot.V, "Y", boundary="fill", other_component={"X": pilot.U}
        ),
    }
    for axis in ("X", "Y"):
        result["vector"][axis] = compare(source_vector[axis], pilot_vector[axis], axis)
    result.update(
        {
            "snapshot": snapshot,
            "face_connections": 13,
            "exact_populated_boundary_results": True,
            "oracle": "xgcm LLC/ECCO 13-face connection table",
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
