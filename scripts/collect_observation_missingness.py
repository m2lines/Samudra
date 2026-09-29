#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect completed missingness-wave metrics with explicit checkpoint lineage."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

ARMS = {
    "legacy-scratch": (0, 16000),
    "masked-scratch": (0, 16000),
    "masked-sequential": (8000, 8000),
    "masked-mixed-finish": (8000, 8000),
    "conditioned-mixed-finish": (8000, 8000),
    "conditioned-mixed": (8000, 8000),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(root):
    files: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    status: dict[str, Any] = {}

    def read(path):
        name = str(path.relative_to(root))
        files[name] = json.loads(path.read_text())
        hashes[name] = digest(path)
        return files[name]

    for arm, (om4, obs) in ARMS.items():
        run = root / arm
        if not (run / "TRAIN_COMPLETE.json").exists():
            status[arm] = "training incomplete"
            continue
        complete = read(run / "TRAIN_COMPLETE.json")
        assert complete["task_counts"] == {"om4": om4, "observation": obs}
        assert complete["global_step"] == om4 + obs
        assert not (run / "TRAIN_PARTIAL.json").exists()
        manifest = read(run / "manifest.json")
        best = read(run / "best.json")
        selected_hash = digest(run / "best.pt")
        assert (
            selected_hash
            == best["checkpoint_sha256"]
            == complete["best_checkpoint_sha256"]
        )
        if manifest["arguments"]["observation_finish"]:
            assert best["task_counts"]["om4"] == om4
            assert (
                best["task_counts"]["observation"]
                > obs - manifest["arguments"]["observation_finish"]
            )
        for path in run.glob("validation-*.json"):
            read(path)
        for name in ("best-anywhere.json", "selection-reference.json"):
            read(run / name)
        needed = [
            "selected-monthly",
            "fixed-endpoint-monthly",
            "selected-annual",
            "completion",
            "state-diagnostics",
        ]
        if not om4:
            needed.append("fixed8k-monthly")
        missing = []
        for suffix in needed:
            directory = root / (arm + "-" + suffix)
            if not (directory / "COMPLETE.json").exists():
                missing.append(suffix)
                continue
            done = read(directory / "COMPLETE.json")
            if suffix.endswith("monthly"):
                signature = read(directory / "evaluation-input.json")
                assert done["origins"] == 96 and done["split"] == "test"
                assert (
                    signature["data_manifest_sha256"]
                    == manifest["data_manifest_sha256"]
                )
                label = "selected" if suffix == "selected-monthly" else "fixed-budget"
                result = read(directory / (label + ".json"))
                assert result["sha256"] == signature["checkpoint_sha256"]
                assert result["metrics"]["origins"] == [
                    f"{year}-{month:02d}"
                    for year in range(2015, 2023)
                    for month in range(1, 13)
                ]
                if suffix == "selected-monthly":
                    assert (
                        signature["checkpoint_sha256"]
                        == done["selected_sha256"]
                        == selected_hash
                    )
                else:
                    step = 8000 if suffix == "fixed8k-monthly" else obs
                    assert signature["checkpoint_sha256"] == digest(
                        run / f"joint-{step:05d}.pt"
                    )
                    assert (
                        signature["fixed_budget_lineage"]
                        == done["fixed_budget_lineage"]
                    )
            elif suffix == "selected-annual":
                signature = read(directory / "input.json")
                assert signature == done["inputs"]
                assert signature["checkpoint_sha256"] == selected_hash
                assert set(done["origins"]) == {
                    "2015-01-01",
                    "2018-01-01",
                    "2021-01-01",
                }
                assert signature["training_manifest_sha256"] == digest(
                    run / "manifest.json"
                )
            else:
                signature = read(directory / "input.json")
                assert signature == done
                assert signature["manifest_sha256"] == digest(run / "manifest.json")
                assert signature["checkpoints"]["best.pt"] == selected_hash
                origin_key = (
                    "origins" if suffix == "completion" else "validation_origins"
                )
                assert len(signature[origin_key]) == 9
            for path in directory.glob("*.json"):
                read(path)
        status[arm] = (
            "verified complete" if not missing else {"missing_evaluations": missing}
        )
    return {"status": status, "files": files, "file_sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root)
    args.output.write_bytes(gzip.compress(json.dumps(result).encode(), mtime=0))
    print(json.dumps(result["status"], indent=2))
    print("bundle_sha256", digest(args.output))


if __name__ == "__main__":
    main()
