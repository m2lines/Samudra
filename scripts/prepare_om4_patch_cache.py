#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Lossless native-quarter-degree rechunking; no regridding or rescaling.

Read the verified transferred source, pack the 77 requested physical fields,
and retain separate surface histories for economical reads. Every frame is
read back and compared exactly before its resumable completion marker is set.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import cftime
import numcodecs  # type: ignore[import-untyped]
import numpy as np
import zarr  # type: ignore[import-untyped]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--transfer-success", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    numcodecs.blosc.set_nthreads(1)
    source = Path(args.source)
    success = json.loads(Path(args.transfer_success).read_text())
    if success.get("destination") != str(
        source
    ) or "Full rclone check --download" not in success.get("validation", ""):
        raise ValueError("Transfer verification does not cover this source")
    original = zarr.open_consolidated(str(source / "OM4.zarr"), mode="r")
    grid = dict(np.load(args.grid))
    names = grid["names"].tolist()
    forcing = ["tauuo", "tauvo", "hfds"]
    times = original["time"][:]
    attrs = dict(original["time"].attrs)
    dates = cftime.num2date(times, attrs["units"], attrs["calendar"])
    stamps = np.array([f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in dates])
    indices = np.flatnonzero((stamps >= "1975-01-03") & (stamps <= "2014-10-05"))
    if not len(indices) or not np.all(np.diff(indices) == 1):
        raise ValueError("Noncontiguous source time selection")
    if not np.allclose(np.diff(times[indices]), 5, rtol=0, atol=1e-8):
        raise ValueError("Expected exact five-day cadence")
    for name in names + forcing:
        if original[name].shape != (len(times), 720, 1440):
            raise ValueError(f"Unexpected native quarter-degree shape: {name}")
        if original[name].dtype != np.dtype("float32"):
            raise ValueError(
                f"Rechunking must preserve the source float32 dtype: {name}"
            )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(
        source=str(source),
        source_metadata_sha256=digest(source / "OM4.zarr/.zmetadata"),
        transfer_success_sha256=digest(args.transfer_success),
        grid_sha256=digest(args.grid),
        names=names,
        forcing=forcing,
        first=str(stamps[indices[0]]),
        last=str(stamps[indices[-1]]),
        frames=len(indices),
        shape=[720, 1440],
        source_indices=indices.tolist(),
        dtype="float32",
        spatial_chunks=[128, 128],
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        policy="Native values, coordinates and masks; no interpolation or normalization",
    )
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("Cache resume contract differs")
    if (output / "CACHE_READY.json").exists():
        ready = json.loads((output / "CACHE_READY.json").read_text())
        if ready["manifest_sha256"] != digest(manifest) or ready[
            "grid_sha256"
        ] != digest(output / "grid.npz"):
            raise ValueError("Completed cache was modified")
        return
    write_json(manifest, contract)
    group = zarr.open_group(str(output / "fields.zarr"), mode="a")
    codec = numcodecs.Blosc(cname="lz4", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    for key, channels in (("prognostic", 77), ("surface", 2), ("boundary", 3)):
        group.require_dataset(
            key,
            shape=(len(indices), channels, 720, 1440),
            chunks=(1, channels, 128, 128),
            dtype="f4",
            fill_value=np.nan,
            compressor=codec,
            exact=True,
        )
    # Per-frame markers have independent chunks so parallel writers never race.
    group.require_dataset(
        "verified", shape=(len(indices),), chunks=(1,), dtype="bool", fill_value=False
    )
    masks = np.stack(
        [
            original["mask_" + (name.rsplit("_", 1)[-1] if "_" in name else "0")][:]
            for name in names
        ]
    ).astype(bool)
    static = dict(
        mask=masks,
        lat=original["y"][:],
        lon=original["x"][:],
        time=times[indices],
        time_units=np.array(attrs["units"]),
        calendar=np.array(attrs["calendar"]),
        dates=stamps[indices],
        names=np.array(names),
        depth=grid["depth"],
    )
    np.savez(output / "grid.npz", **static)
    surface_ids = [names.index("thetao_0"), names.index("zos")]
    forcing_mask = masks[surface_ids[0]]

    def convert(i):
        if bool(group["verified"][i]):
            return i
        t = int(indices[i])
        state = np.stack([original[name][t] for name in names]).astype("f4")
        boundary = np.stack([original[name][t] for name in forcing]).astype("f4")
        if (
            not np.isfinite(state[masks]).all()
            or not np.isfinite(boundary[:, forcing_mask]).all()
        ):
            raise ValueError(f"Nonfinite wet source values at {stamps[t]}")
        for key, values in (
            ("prognostic", state),
            ("surface", state[surface_ids]),
            ("boundary", boundary),
        ):
            group[key][i] = values
            if not np.array_equal(group[key][i], values, equal_nan=True):
                raise ValueError(f"Cache readback mismatch {key}/{i}")
        group["verified"][i] = True
        return i

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, i in enumerate(pool.map(convert, range(len(indices))), 1):
            if done % 20 == 0 or done == len(indices):
                print(
                    json.dumps(
                        dict(
                            event="frame_verified",
                            completed=done,
                            frames=len(indices),
                            index=i,
                        )
                    ),
                    flush=True,
                )
    if not group["verified"][:].all():
        raise ValueError("Incomplete verified cache")
    zarr.consolidate_metadata(group.store)
    write_json(
        output / "CACHE_READY.json",
        dict(
            manifest_sha256=digest(manifest),
            grid_sha256=digest(output / "grid.npz"),
            frames=len(indices),
            verification="all physical frames read back exactly, including NaNs; wet finite checked",
        ),
    )


if __name__ == "__main__":
    main()
