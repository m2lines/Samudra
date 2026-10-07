#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Read-only source audit and isolated experiment readiness manifests."""

import argparse
import hashlib
import json
from pathlib import Path

import cftime
import numpy as np
import zarr

from scripts.verify_extent_observations import main as verify_observations


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    paths = json.loads((root / "paths.json").read_text())
    obs = Path(paths["observations"])
    obs.mkdir(parents=True, exist_ok=True)
    origin = Path(paths["observation_source"])
    for name in [
        "grid.npz",
        "statistics.npz",
        "SHA256SUMS",
        "train",
        "validation",
        "test",
    ]:
        p = obs / name
        if not p.exists():
            p.symlink_to(origin / name)
        if p.resolve() != (origin / name).resolve():
            raise ValueError("Observation source changed")
    sys.argv = ["verify", str(obs)]
    verify_observations()
    expected = {
        "SHA256SUMS": "d6e6672ca11aec3491a4fc896d9e99d528075a889cc0e6187b580ef8a52500f3",
        "grid.npz": "ce96360964a6669d0186b9581abaaffc06f660c4e09e0ac813b3dd1b819e5167",
        "statistics.npz": "7688d483d4c87a8ce0e2f27a1a342db93eecb3e892187506475d6970e865c001",
    }
    for f, digest in expected.items():
        if sha(obs / f) != digest:
            raise ValueError("Observation contract differs: " + f)
    if (
        sha(root / "selection-reference.json")
        != "c94c0601e168858e4ba41423aed74d8604eaa1efbc68dd76370b32b8eeacd26c"
    ):
        raise ValueError("Selection reference differs from U-global")
    receipt = {"producer": paths["producer"], "metadata_sha256": {}, "sources": {}}
    times = []
    for key, path in [("coarse", paths["global_om4"]), ("fine", paths["fine_om4"])]:
        p = Path(path)
        store = zarr.open_consolidated(str(p / "OM4.zarr"))
        receipt[key] = str(p)
        receipt["metadata_sha256"][key] = sha(p / "OM4.zarr/.zmetadata")
        t = store["time"][:]
        times.append(t)
        a = dict(store["time"].attrs)
        dates = cftime.num2date(t, a["units"], a["calendar"])
        indices = np.flatnonzero(np.array([v.year for v in dates]) < 1975)
        if len(indices) != 1241 or indices[0] != 0:
            raise ValueError("Wrong early coverage")
        variables = [
            f"{v}_{z}" for v in ["uo", "vo", "thetao", "so"] for z in range(19)
        ] + ["zos", "tauuo", "tauvo", "hfds"]
        checked = 0
        bytes_present = 0
        for name in variables:
            meta = json.loads((p / "OM4.zarr" / name / ".zarray").read_text())
            if meta["compressor"] is not None or meta["chunks"][0] != 1:
                raise ValueError("Expected published uncompressed per-frame fields")
            size = int(np.prod(meta["chunks"])) * np.dtype(meta["dtype"]).itemsize
            for i in indices:
                f = p / "OM4.zarr" / name / f"{i}.0.0"
                if f.stat().st_size != size:
                    raise ValueError("Missing/truncated native chunk: " + str(f))
                checked += 1
                bytes_present += size
            for i in [0, 620, 1240]:
                value = store[name][i]
                level = (
                    0
                    if "_" not in name or name in ["tauuo", "tauvo"]
                    else int(name.rsplit("_", 1)[1])
                )
                mask = store["mask_" + str(level)][:].astype(bool)
                if not np.isfinite(value[mask]).all():
                    raise ValueError("Nonfinite wet source sample " + name)
        receipt["sources"][key] = {
            "first": str(dates[0]),
            "last": str(dates[-1]),
            "early_frames": 1241,
            "early_chunks_size_checked": checked,
            "early_payload_bytes": bytes_present,
            "finite_sample_frames": [0, 620, 1240],
        }
    np.testing.assert_array_equal(*times)
    early = Path(paths["early_data"])
    early.mkdir(parents=True, exist_ok=True)
    target = early / "READY.json"
    if target.exists() and json.loads(target.read_text()) != receipt:
        raise ValueError("Ready contract changed")
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
