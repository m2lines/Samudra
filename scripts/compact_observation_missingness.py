#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Extract lossless native-grid report arrays from completed evaluations."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

CHANNELS = [38, 47, 57, 66, 0, 19, 76]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((args.root / "PRODUCTION_DAG.json").read_text())["jobs"]
    producer = digest(Path(__file__))
    for arm in jobs:
        run = args.root / arm
        state = args.root / (arm + "-state-diagnostics")
        annual = args.root / (arm + "-selected-annual")
        completion = args.root / (arm + "-completion")
        if not all((p / "COMPLETE.json").exists() for p in (state, annual, completion)):
            continue
        if not (run / "TRAIN_COMPLETE.json").exists():
            raise ValueError("Training must finish before report extraction")
        best = json.loads((run / "best.json").read_text())["checkpoint_sha256"]
        signature = {"extractor_sha256": producer, "best_checkpoint_sha256": best}
        marker = args.output / (arm + ".json")
        if marker.exists():
            assert json.loads(marker.read_text())["signature"] == signature
            continue
        state_input = json.loads((state / "COMPLETE.json").read_text())
        annual_input = json.loads((annual / "COMPLETE.json").read_text())["inputs"]
        assert (
            state_input["checkpoints"]["best.pt"]
            == annual_input["checkpoint_sha256"]
            == best
        )
        destination = args.output / arm
        destination.mkdir(exist_ok=True)
        inputs = {}
        checkpoints = [
            "joint-00000",
            "joint-00010",
            "joint-00100",
            "joint-01000",
            "joint-04000",
            "best",
            "joint-08000",
        ]
        if (state / "joint-16000.npz").exists():
            checkpoints.append("joint-16000")
        if (state / "before-observation-finish.npz").exists():
            checkpoints.extend(
                ["before-observation-finish", "joint-06100", "joint-06500"]
            )
        for name in checkpoints:
            path = state / (name + ".npz")
            inputs[str(path.relative_to(args.root))] = digest(path)
            with np.load(path) as source:
                assert source["initial"].shape == (2, 77, 180, 360)
                np.savez_compressed(
                    destination / (name + ".npz"),
                    initial=source["initial"][:, CHANNELS],
                    day30=source["day30"][:, CHANNELS],
                    cases=source["cases"],
                )
                if name == "best":
                    np.savez_compressed(
                        destination / "grid.npz",
                        lat=source["lat"],
                        lon=source["lon"],
                        mask=source["mask"][CHANNELS],
                        names=source["names"][CHANNELS],
                        channels=CHANNELS,
                    )
        for origin in ("2015-01-01", "2018-01-01", "2021-01-01"):
            path = annual / (origin + ".npz")
            inputs[str(path.relative_to(args.root))] = digest(path)
            with np.load(path) as source:
                indices = [list(source["leads_days"]).index(day) for day in (30, 365)]
                np.savez_compressed(
                    destination / ("annual-" + origin + ".npz"),
                    initial=source["initial"][-1, CHANNELS],
                    state=source["state_at_leads"][indices][:, CHANNELS],
                    surface=source["surface"][[5, 72]],
                    reference=source["reference"][[5, 72]],
                    leads_days=[30, 365],
                )
        for path in completion.glob("best-*-natural.npz"):
            inputs[str(path.relative_to(args.root))] = digest(path)
            (destination / path.name).write_bytes(path.read_bytes())
        files = {
            p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in destination.glob("*.npz")
        }
        record = {
            "signature": signature,
            "source_files": inputs,
            "files": files,
            "scope": "Lossless channel/time subsetting only; 180x360 native grid. All three annual cases retained. No model execution or selection.",
        }
        (destination / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        temporary = args.output / (arm + ".tmp.tar")
        with tarfile.open(temporary, "w") as archive:
            archive.add(destination, arcname=arm)
        final = args.output / (arm + ".tar")
        temporary.replace(final)
        record["archive_sha256"] = digest(final)
        record["archive_bytes"] = final.stat().st_size
        marker.write_text(json.dumps(record, indent=2) + "\n")
        print(arm, record["archive_bytes"], record["archive_sha256"], flush=True)


if __name__ == "__main__":
    main()
