# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Write equivalent sharded arrays for the Rust compatibility spike."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec, ZstdCodec  # type: ignore[import-untyped]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    root = zarr.create_group(store=str(args.output), zarr_format=3)
    data = np.arange(4 * 216 * 216, dtype="float32").reshape(1, 4, 216, 216)
    codecs = {
        "blosc": BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"),
        "zstd": ZstdCodec(level=5),
    }
    for name, codec in codecs.items():
        array = root.create_array(
            name,
            shape=data.shape,
            chunks=(1, 4, 18, 18),
            shards=(1, 4, 72, 72),
            dtype=data.dtype,
            compressors=[codec],
            dimension_names=("time", "k", "j", "i"),
        )
        array[:] = data
        print(
            f"{name}_stored_bytes={sum(p.stat().st_size for p in (args.output / name / 'c').rglob('*') if p.is_file())}"
        )


if __name__ == "__main__":
    main()
