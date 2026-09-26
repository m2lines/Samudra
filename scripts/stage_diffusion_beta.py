#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit and register existing campaign datasets without modifying their contents.

Run with the observation preparation Python environment (numpy/xarray/zarr).
The half-degree store is exposed as targets only. No source weights or GPU jobs
are selected by this program. It fails loudly on incomplete stores or payloads.
"""

import argparse
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_payload(root, name, expected):
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Payload escapes root: {name}")
    if digest(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return path.stat().st_size


def audit_store(path, grid):
    metadata_path = path / ".zmetadata"
    metadata = json.loads(metadata_path.read_text())["metadata"]
    missing, chunks, size = [], 0, 0
    for key, spec in metadata.items():
        if not key.endswith("/.zarray"):
            continue
        directory = path / key.removesuffix("/.zarray")
        separator = spec.get("dimension_separator", ".")
        counts = [
            math.ceil(n / c) for n, c in zip(spec["shape"], spec["chunks"], strict=True)
        ]
        for index in itertools.product(*(range(n) for n in counts)):
            item = directory / (separator.join(map(str, index)) if index else "0")
            if not item.is_file():
                missing.append(str(item.relative_to(path)))
            else:
                chunks += 1
                size += item.stat().st_size
    if missing:
        raise ValueError(f"{path}: {len(missing)} missing chunks: {missing[:10]}")
    # Keep numeric times and explicit CF metadata; no Gregorian approximation.
    data = xr.open_zarr(path, consolidated=True, decode_times=False)
    if (data.sizes["y"], data.sizes["x"]) != tuple(grid):
        raise ValueError(f"Wrong grid: {path}")
    probes = {}
    for name in ("thetao_0", "thetao_9", "so_9", "uo_9", "vo_9", "zos"):
        values = np.asarray(data[name].isel(time=[0, 1800, -1]).values)
        if not np.isfinite(values).any(axis=(-2, -1)).all():
            raise ValueError(f"Empty target sample: {path}/{name}")
        probes[name] = {
            "finite": int(np.isfinite(values).sum()),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
    result = {
        "path": str(path),
        "metadata_sha256": digest(metadata_path),
        "payload_bytes": size,
        "chunks_present": chunks,
        "grid": grid,
        "frames": data.sizes["time"],
        "time_metadata": dict(data.time.attrs),
        "time_first_last": data.time.values[[0, -1]].tolist(),
        "source": data.attrs,
        "sample_reads": probes,
        "verification": "All declared chunks present; six channels decoded at three times. This is not a full payload checksum audit.",
    }
    return result, data.time.values.copy(), dict(data.time.attrs)


def link(path, target):
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError(f"Existing link points elsewhere: {path}")
    elif path.exists():
        raise FileExistsError(path)
    else:
        path.symlink_to(target, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/home/jrusak/data/diffusion-interior-beta"),
    )
    parser.add_argument(
        "--half", type=Path, default=Path("/mnt/home/jrusak/data/om4_halfdeg")
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("/mnt/home/jrusak/data/obs_full_range/d-observation-pilot"),
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Verify half-degree and observation data while one-degree copy proceeds",
    )
    parser.add_argument(
        "--reuse-one-degree",
        type=Path,
        help="Use an existing, user-approved one-degree release; audit locally without claiming an OSN copy verification",
    )
    args = parser.parse_args()
    root = args.root
    (root / "data").mkdir(parents=True, exist_ok=True)
    if args.reuse_one_degree is not None:
        link(root / "data/om4_onedeg_v3", args.reuse_one_degree)
    link(root / "data/om4_halfdeg_targets", args.half)
    link(root / "data/observations", args.observations / "samples")
    link(root / "data/annual_observations", args.observations / "annual-instance-v1")
    half, half_time, half_attrs = audit_store(args.half / "OM4.zarr", [360, 720])
    obs = args.observations / "samples"
    entries = [
        line.split(maxsplit=1)
        for line in (obs / "SHA256SUMS").read_text().splitlines()
        if line.strip()
    ]
    obs_bytes = sum(
        verify_payload(obs, name.lstrip("*"), expected) for expected, name in entries
    )
    counts = {
        split: len(list((obs / split).glob("*.npz")))
        for split in ("train", "validation", "test")
    }
    if counts != {"train": 243, "validation": 9, "test": 96}:
        raise ValueError(f"Changed observational split: {counts}")
    annual_records, annual_bytes = 0, 0
    for manifest in sorted(
        (args.observations / "annual-instance-v1").glob("*/*/COMPLETE.json")
    ):
        records = json.loads(manifest.read_text())
        for record in records["records"] + records["monthly_interiors"]:
            annual_bytes += verify_payload(
                manifest.parent, record["file"], record["sha256"]
            )
            annual_records += 1
    if annual_records != 416:
        raise ValueError(f"Incomplete annual bundle: {annual_records}")
    report = {
        "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "half_degree_targets": half,
        "observations": {
            "path": str(obs),
            "files_verified": len(entries),
            "payload_bytes": obs_bytes,
            "split_counts": counts,
            "manifest_sha256": digest(obs / "SHA256SUMS"),
            "grid_sha256": digest(obs / "grid.npz"),
            "statistics_sha256": digest(obs / "statistics.npz"),
        },
        "annual": {
            "path": str(args.observations / "annual-instance-v1"),
            "files_verified": annual_records,
            "payload_bytes": annual_bytes,
        },
        "half_degree_policy": "Targets only; surface inputs, forcings, latent grid, observations and state normalization remain unchanged",
        "baseline_ready": False,
    }
    if not args.existing_only:
        if (
            args.reuse_one_degree is None
            and not (root / "data/OM4_COPY_VERIFIED.txt").is_file()
        ):
            raise ValueError("One-degree source read-back check is not complete")
        one, one_time, one_attrs = audit_store(
            root / "data/om4_onedeg_v3/OM4.zarr", [180, 360]
        )
        if one_attrs != half_attrs or not np.array_equal(one_time, half_time):
            raise ValueError("One- and half-degree CF times do not match exactly")
        report["one_degree_inputs_and_targets"] = one
        report["time_alignment"] = (
            "All 4745 numeric CF timestamps and their units/calendar are identical"
        )
        report["one_degree_copy_verification"] = (
            "Existing local release approved by user; chunk presence and sample decoding only"
            if args.reuse_one_degree is not None
            else "rclone check --download against the original OSN release"
        )
    name = "EXISTING_DATA_VERIFIED.json" if args.existing_only else "DATA_READY.json"
    temporary = root / (name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(root / name)
    print(
        json.dumps(
            {
                "report": str(root / name),
                "observation_files": len(entries),
                "annual_files": annual_records,
                "half_degree_chunks": half["chunks_present"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
