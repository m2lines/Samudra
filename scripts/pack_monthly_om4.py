#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Pack existing OM4 training frames without changing any physical values."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from samudra.experiments.monthly_model_data import ModelMonths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--om4", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise FileExistsError("Packed store already complete")
    reader = ModelMonths(Path(a.om4) / "OM4.zarr", "om4", dict(np.load(a.grid)), a.om4)
    indices = sorted(
        {i for first, stop, _, _ in reader.windows for i in range(first - 19, stop)}
    )
    shape = (len(indices), 77, *reader.mask.shape[-2:])
    frames = np.lib.format.open_memmap(
        out / "frames.partial.npy", mode="w+", dtype=np.float32, shape=shape
    )

    def read(index):
        value = np.stack([reader.store[name][index] for name in reader.names])
        if value.dtype != np.float32:
            raise ValueError("Source dtype differs")
        return value

    digests = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        # Bounded batches keep at most workers frames resident.
        for offset in range(0, len(indices), a.workers):
            batch = indices[offset : offset + a.workers]
            for position, value in enumerate(pool.map(read, batch), start=offset):
                frames[position] = value
                digests.append(hashlib.sha256(value.tobytes()).hexdigest())
            if offset % (a.workers * 10) == 0:
                print(
                    json.dumps(
                        dict(
                            packed=min(offset + a.workers, len(indices)),
                            total=len(indices),
                        )
                    ),
                    flush=True,
                )
    frames.flush()
    del frames
    saved = np.load(out / "frames.partial.npy", mmap_mode="r")
    for i, expected in enumerate(digests):
        if hashlib.sha256(saved[i].tobytes()).hexdigest() != expected:
            raise ValueError("Packed frame read-back differs")
    del saved
    (out / "frames.partial.npy").rename(out / "frames.npy")
    result = dict(
        source=str(Path(a.om4) / "OM4.zarr"),
        source_metadata_sha256=hashlib.sha256(
            (Path(a.om4) / "OM4.zarr" / ".zmetadata").read_bytes()
        ).hexdigest(),
        grid_sha256=hashlib.sha256(Path(a.grid).read_bytes()).hexdigest(),
        indices=indices,
        shape=shape,
        dtype="float32",
        frame_sha256=digests,
        verification="every packed frame read back and byte-hash matched to source",
        bytes=(out / "frames.npy").stat().st_size,
    )
    (out / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("indices", "frame_sha256")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
