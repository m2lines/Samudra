# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Create a metadata-only inventory of the complete LLC4320 source store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr  # type: ignore[import-untyped]


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    group = zarr.open_group(str(args.source), mode="r")
    arrays: dict[str, Any] = {}
    decoded_bytes = 0
    mounts: dict[str, list[str]] = {}
    for name in sorted(group.array_keys()):
        array = group[name]
        source_path = args.source / name
        resolved = source_path.resolve()
        mount = str(Path(*resolved.parts[:5]))
        mounts.setdefault(mount, []).append(name)
        itemsize = np.dtype(array.dtype).itemsize
        logical_bytes = int(np.prod(array.shape, dtype=np.int64)) * itemsize
        decoded_bytes += logical_bytes
        arrays[name] = {
            "shape": list(array.shape),
            "chunks": list(array.chunks),
            "dtype": str(array.dtype),
            "decoded_bytes": logical_bytes,
            "fill_value": json_value(array.fill_value),
            "compressor": json_value(array.compressor.get_config())
            if array.compressor is not None
            else None,
            "filters": [json_value(item.get_config()) for item in array.filters or []],
            "dimensions": list(array.attrs.get("_ARRAY_DIMENSIONS", [])),
            "attributes": json_value(dict(array.attrs)),
            "source_path": str(source_path),
            "resolved_path": str(resolved),
            "is_symlink": source_path.is_symlink(),
        }
    result = {
        "source": str(args.source),
        "zarr_format": 2,
        "array_count": len(arrays),
        "decoded_bytes": decoded_bytes,
        "root_attributes": json_value(dict(group.attrs)),
        "resolved_mounts": mounts,
        "arrays": arrays,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
