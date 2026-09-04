# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Measure Zarr v3 sharding behavior for tile-plus-halo reads.

This deliberately uses a 1:10 spatial scale model of LLC4320: a 72x72 tile,
an 18x18 inner chunk, and a four-pixel halo.  The topology is a simple 3x3
plane; LLC face transformations are outside the scope of this storage spike.

Run with:

    uv run --isolated --with 'zarr>=3.1,<4' --with numpy \
        python docs/design/spikes/zarr_sharding_read_spike.py
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import zarr  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec  # type: ignore[import-untyped]
from zarr.storage import MemoryStore  # type: ignore[import-untyped]

TILE = 72
HALO = 4
INNER = 18
LEVELS = 4


class TracingMemoryStore(MemoryStore):
    """MemoryStore that records keys and byte ranges returned by reads."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[tuple[str, str, int]] = []

    async def get(self, key, prototype=None, byte_range=None):  # noqa: ANN001
        result = await super().get(key, prototype=prototype, byte_range=byte_range)
        size = 0 if result is None else len(result)
        self.reads.append((key, repr(byte_range), size))
        return result


def ocean_like_data() -> np.ndarray:
    """Return smooth float32 data with a deterministic land mask."""
    y, x = np.mgrid[: 3 * TILE, : 3 * TILE]
    fields = []
    for level in range(LEVELS):
        field = (
            np.sin(x / (7.0 + level)) + np.cos(y / (11.0 + level)) + level / 10
        ).astype("float32")
        field[(x + 2 * y + level) % 10 < 4] = np.nan
        fields.append(field)
    return np.stack(fields)[None, ...]


def stored_bytes(store: TracingMemoryStore) -> tuple[int, int]:
    """Count physical data objects and bytes, excluding metadata."""
    objects = [
        value for key, value in store._store_dict.items() if key.startswith("c/")
    ]
    return len(objects), sum(len(value) for value in objects)


def read_summary(store: TracingMemoryStore) -> dict[str, Any]:
    data_reads = [read for read in store.reads if read[0].startswith("c/")]
    return {
        "data_get_calls": len(data_reads),
        "unique_data_objects": len({key for key, _, _ in data_reads}),
        "bytes_returned": sum(size for _, _, size in data_reads),
        "byte_range_kinds": dict(
            Counter(byte_range for _, byte_range, _ in data_reads)
        ),
    }


def exercise(name: str, *, chunks: tuple[int, ...], shards: tuple[int, ...] | None):
    store = TracingMemoryStore()
    array = zarr.create_array(
        store=store,
        shape=(1, LEVELS, 3 * TILE, 3 * TILE),
        chunks=chunks,
        shards=shards,
        dtype="float32",
        fill_value=np.nan,
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")],
    )
    source = ocean_like_data()
    array[:] = source

    object_count, physical_bytes = stored_bytes(store)
    store.reads.clear()
    result = array[:, :, TILE - HALO : 2 * TILE + HALO, TILE - HALO : 2 * TILE + HALO]
    expected = source[
        :, :, TILE - HALO : 2 * TILE + HALO, TILE - HALO : 2 * TILE + HALO
    ]
    np.testing.assert_equal(result, expected)

    return {
        "configuration": name,
        "chunks": chunks,
        "shards": shards,
        "physical_data_objects": object_count,
        "physical_data_bytes": physical_bytes,
        "logical_data_bytes": source.nbytes,
        "read": read_summary(store),
    }


def main() -> None:
    configs = [
        exercise(
            "source-like-unsharded",
            chunks=(1, LEVELS, TILE, TILE),
            shards=None,
        ),
        exercise(
            "small-unsharded",
            chunks=(1, LEVELS, INNER, INNER),
            shards=None,
        ),
        exercise(
            "small-chunks-in-tile-shards",
            chunks=(1, LEVELS, INNER, INNER),
            shards=(1, LEVELS, TILE, TILE),
        ),
    ]
    print(json.dumps({"zarr_version": zarr.__version__, "results": configs}, indent=2))


if __name__ == "__main__":
    main()
