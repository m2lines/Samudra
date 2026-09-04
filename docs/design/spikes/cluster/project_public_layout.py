# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Project target layouts and physical-object upper bounds for all arrays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def layout(dimensions: list[str], shape: list[int]) -> tuple[list[int], list[int]]:
    if dimensions[:2] == ["time", "k_p1"] and len(dimensions) == 5:
        return [1, 17, 1, 120, 120], [2, 68, 1, 1440, 1440]
    if dimensions[:2] == ["time", "k"] and len(dimensions) == 5:
        return [1, 17, 1, 120, 120], [2, 51, 1, 1440, 1440]
    if dimensions[0:1] == ["time"] and len(dimensions) == 4:
        return [1, 1, 360, 360], [24, 1, 1080, 1080]
    if dimensions[0:1] == ["k"] and len(dimensions) == 4:
        return [17, 1, 120, 120], [51, 1, 1440, 1440]
    if dimensions[0:1] == ["face"] and len(dimensions) == 3:
        return [1, 360, 360], [1, 1080, 1080]
    return shape, shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args()
    source = json.loads(args.inventory.read_text())
    arrays: dict[str, Any] = {}
    total_objects = 0
    total_inner_chunks = 0
    for name, metadata in source["arrays"].items():
        shape = metadata["shape"]
        chunks, shards = layout(metadata["dimensions"], shape)
        objects = product(
            [math.ceil(length / width) for length, width in zip(shape, shards)]
        )
        inner_chunks = product(
            [math.ceil(length / width) for length, width in zip(shape, chunks)]
        )
        total_objects += objects
        total_inner_chunks += inner_chunks
        arrays[name] = {
            "shape": shape,
            "dimensions": metadata["dimensions"],
            "chunks": chunks,
            "shards": shards,
            "physical_data_object_upper_bound": objects,
            "logical_inner_chunks": inner_chunks,
        }
    print(
        json.dumps(
            {
                "array_count": len(arrays),
                "physical_data_object_upper_bound": total_objects,
                "logical_inner_chunks": total_inner_chunks,
                "notes": [
                    "Upper bound includes all-fill shards that may be omitted.",
                    "Repository metadata and manifest objects are not included.",
                    "Byte and duration projections require destination measurements.",
                ],
                "arrays": arrays,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
