# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Lossless packed OM4 frames, indexed by the original native calendar."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import zarr  # type: ignore[import-untyped]
from numcodecs import Blosc  # type: ignore[import-untyped]

from samudra.native_reader import PhysicalVariable
from samudra.observations.shared_era5 import digest, required_indices


def interval_bounds(reader, indices):
    return [[str(t) for t in reader.bounds[i]] for i in indices]


class NativeOm4Planes:
    """Remap native time/channel selectors without changing ModelMonths windows."""

    def __init__(self, cache, original, runtime, proof_sha256, *, verify_content=True):
        cache = Path(cache)
        proof_path = cache / "COMPLETE.json"
        if digest(proof_path) != proof_sha256:
            raise ValueError("Packed OM4 proof changed")
        proof = json.loads(proof_path.read_text())
        indices = required_indices(original)
        if (
            proof["version"] != 1
            or proof["channels"] != original.names
            or proof["source_metadata_sha256"] != original.source_metadata_sha256
            or proof["indices"] != indices
            or proof["bounds"] != interval_bounds(original, indices)
            or not proof["all_frames_rust_readback_exact"]
            or not proof["all_frames_python_readback_exact"]
        ):
            raise ValueError("Packed OM4 source, cohort, channels or intervals differ")
        if verify_content:
            for name, sha in proof["files"].items():
                if digest(cache / name) != sha:
                    raise ValueError(f"Packed OM4 cache content changed: {name}")
        self.positions = {i: j for j, i in enumerate(indices)}
        self.variables = {
            name: PhysicalVariable("values", i) for i, name in enumerate(original.names)
        }
        self.reader = runtime.open(cache / "frames.zarr", list(self.variables.values()))
        if self.reader.shape != (len(indices), *original.mask.shape[-2:]):
            raise ValueError("Packed OM4 cache shape differs")

    def read_into(self, indices, variables, output):
        if any(v.level is not None or v.name not in self.variables for v in variables):
            raise ValueError("Packed OM4 requires original flat physical channels")
        self.reader.read_into(
            [self.positions[i] for i in indices],
            [self.variables[v.name] for v in variables],
            output,
        )


def prepare_native_om4(source, original, output, runtime):
    """Pack every eligible frame; independently decode each with Python and Rust."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace OM4 cache: {output}")
    indices = required_indices(original)
    if not indices:
        raise ValueError("Empty OM4 training cohort")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent)
    )
    store = staging / "frames.zarr"
    group = zarr.open_group(str(store), mode="w")
    shape = (len(original.names), *original.mask.shape[-2:])
    array = group.create_dataset(
        "values",
        shape=(len(indices), *shape),
        chunks=(1, *shape),
        dtype="f4",
        fill_value=np.nan,
        compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "lev", "y", "x"]
    flat_variables = [PhysicalVariable(n) for n in original.names]
    packed_variables = [
        PhysicalVariable("values", i) for i in range(len(original.names))
    ]
    flat = runtime.open(source, flat_variables)
    packed = runtime.open(store, packed_variables)
    expected = np.empty((1, *shape), np.float32)
    actual = np.empty_like(expected)
    source_values = hashlib.sha256()
    for position, index in enumerate(indices):
        flat.read_into([index], flat_variables, expected)
        source_values.update(expected.tobytes())
        array[position] = expected[0]
        np.testing.assert_array_equal(array[position], expected[0], err_msg=str(index))
        packed.read_into([position], packed_variables, actual)
        np.testing.assert_array_equal(actual, expected, err_msg=str(index))
        if position % 100 == 0:
            print(
                json.dumps(
                    {"frame": position, "total": len(indices), "native_index": index}
                ),
                flush=True,
            )
    zarr.consolidate_metadata(str(store))
    if digest(source / ".zmetadata") != original.source_metadata_sha256:
        raise ValueError("OM4 source metadata changed during packing")
    proof = {
        "version": 1,
        "source": str(source),
        "source_metadata_sha256": original.source_metadata_sha256,
        "source_values_sha256": source_values.hexdigest(),
        "channels": original.names,
        "indices": indices,
        "bounds": interval_bounds(original, indices),
        "all_frames_rust_readback_exact": True,
        "all_frames_python_readback_exact": True,
        "files": {
            str(p.relative_to(staging)): digest(p)
            for p in sorted(store.rglob("*"))
            if p.is_file()
        },
    }
    (staging / "COMPLETE.json").write_text(json.dumps(proof, indent=2) + "\n")
    staging.rename(output)
    print(
        json.dumps(
            {"output": str(output), "proof_sha256": digest(output / "COMPLETE.json")}
        ),
        flush=True,
    )
