#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect global-domain comparisons with actual checkpoint and support provenance."""

import argparse
import datetime
import gzip
import hashlib
import json
import re
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

BASE = Path("/scratch/jr7309/runs")
GLOBAL = BASE / "2026-09-29-observation-global"
LATENT = BASE / "2026-09-29-observation-global-latent10"
ARMS = {
    "restricted": (
        BASE / "2026-09-28-observation-missingness/conditioned-mixed",
        GLOBAL,
        "control",
        "053e34947932fdb56c6baf2c717a23f2df3cc733",  # pragma: allowlist secret
    ),
    "global": (
        GLOBAL / "conditioned-mixed-global",
        GLOBAL,
        "global",
        "79025a163817a577ab81af95b401f5cb0563cd12",  # pragma: allowlist secret
    ),
    "latent10": (
        LATENT / "conditioned-mixed-global-latent10",
        LATENT,
        "global",
        "b95179b963d372b72333c7f0e7e521a67b51ff6e",  # pragma: allowlist secret
    ),
}
ORIGINS = ["2015-01-01", "2018-01-01", "2021-01-01"]
TEST_MONTHS = [
    f"{year}-{month:02d}" for year in range(2015, 2023) for month in range(1, 13)
]


@cache
def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect():
    files, hashes = {}, {}
    status: dict[str, Any] = {}

    def read(path, key):
        files[key] = json.loads(path.read_text())
        hashes[key] = {"path": str(path), "sha256": digest(path)}
        return files[key]

    for arm, (run, root, prefix, producer) in ARMS.items():
        if not (run / "TRAIN_COMPLETE.json").exists():
            status[arm] = "training incomplete"
            continue
        complete = read(run / "TRAIN_COMPLETE.json", arm + "/run/complete")
        assert complete["task_counts"] == {"om4": 8000, "observation": 8000}
        assert complete["global_step"] == 16000
        assert not (run / "TRAIN_PARTIAL.json").exists()
        manifest = read(run / "manifest.json", arm + "/run/manifest")
        options = manifest["arguments"]
        assert manifest["code_commit"] == producer
        assert options["seed"] == 1729 and options["ordering"] == "mixed"
        assert options.get("global_observations", False) == (arm != "restricted")
        assert options.get("latent_channels", 0) == (10 if arm == "latent10" else 0)
        assert options["normalization"] == "instance"
        assert (
            options["surface_policy"] == "observed-only"
            and options["surface_fill"] == "zero"
        )
        best = read(run / "best.json", arm + "/run/best")
        reference = read(run / "selection-reference.json", arm + "/run/reference")
        assert reference["protocol"]["version"] == (3 if arm == "restricted" else 4)
        assert reference["data_manifest_sha256"] == manifest["data_manifest_sha256"]
        assert digest(run / "best.pt") == best["checkpoint_sha256"]
        for path in sorted(run.glob("validation-*.json")):
            read(path, arm + "/validation/" + path.stem)
        missing = []
        for choice in ("endpoint", "selected"):
            checkpoint = run / ("joint-08000.pt" if choice == "endpoint" else "best.pt")
            for kind in ("monthly", "annual"):
                directory = root / f"{prefix}-{choice}-{kind}"
                key = f"{arm}/{choice}/{kind}"
                if not (directory / "COMPLETE.json").exists():
                    missing.append(str(directory))
                    continue
                marker = read(directory / "COMPLETE.json", key + "/complete")
                signature = read(
                    directory
                    / ("evaluation-input.json" if kind == "monthly" else "input.json"),
                    key + "/input",
                )
                assert signature["global_observations"] is True
                assert signature["checkpoint_sha256"] == digest(checkpoint)
                assert signature["split"] == "test"
                if choice == "endpoint":
                    lineage = signature["fixed_budget_lineage"]
                    assert lineage["task_counts"] == {"om4": 8000, "observation": 8000}
                    assert lineage["global_step"] == 16000
                    assert lineage["training_manifest_sha256"] == digest(
                        run / "manifest.json"
                    )
                if kind == "monthly":
                    assert marker["origins"] == 96
                    stem = "fixed-budget" if choice == "endpoint" else "selected"
                    for name in (
                        stem,
                        stem + "-inferred-persistence",
                        stem + "-inferred-anomaly-persistence",
                        "seasonal-climatology",
                    ):
                        record = read(directory / (name + ".json"), key + "/" + name)
                        metrics = record["metrics"] if name == stem else record
                        assert metrics["origins"] == TEST_MONTHS
                else:
                    assert sorted(marker["origins"]) == ORIGINS
                    assert marker["inputs"] == signature
                    for origin in ORIGINS:
                        read(directory / (origin + ".json"), key + "/" + origin)
        if arm != "restricted":
            read(root / "fit-conditioned/QUALIFIED.json", arm + "/qualification/fit")
            read(
                root / "probe-conditioned/JOINT_QUALIFIED.json",
                arm + "/qualification/resume",
            )
            assert all(files[arm + "/qualification/fit"]["gradient_reached"].values())
            assert files[arm + "/qualification/resume"]["resume_verified"]
        status[arm] = {"missing": missing} if missing else "verified complete"
    jobs = {"18811814", "18815268"}
    for root in (GLOBAL, LATENT):
        for path in root.glob("*-submission.json"):
            jobs.add(str(json.loads(path.read_text())["job"]))
    raw = subprocess.check_output(
        [
            "sacct",
            "--duplicates",
            "-nXP",
            "-j",
            ",".join(sorted(jobs)),
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Start,End",
        ],
        text=True,
    )
    accounting = []
    for line in raw.splitlines():
        job, state, code, seconds, tres, start, end, *_ = line.split("|")
        if job not in jobs:
            continue
        match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
        gpus = int(match[1]) if match else 0
        accounting.append(
            dict(
                job=job,
                state=state,
                exit_code=code,
                seconds=int(seconds),
                gpus=gpus,
                gpu_hours=int(seconds) * gpus / 3600,
                start=start,
                end=end,
            )
        )
    return dict(
        collected_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        files=files,
        hashes=hashes,
        status=status,
        accounting=accounting,
        allocated_gpu_hours=sum(r["gpu_hours"] for r in accounting),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        gzip.compress(json.dumps(result, allow_nan=False).encode(), mtime=0)
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "gpu_hours": result["allocated_gpu_hours"],
                "sha256": digest(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
