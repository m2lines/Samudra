#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, CPU-light cross-scheduler dependency bridge on the beta login host.

No GPU allocation, no experiment resubmission, and no conversation timer. Exit
once the exact recorded jobs are released, or after 48 hours with a failure file.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1])
    config = json.loads((root / "paths.json").read_text())
    jobs = json.loads((root / "jobs.json").read_text())
    receipt = root / "released.json"
    released = json.loads(receipt.read_text()) if receipt.exists() else []
    deadline = time.monotonic() + 48 * 3600
    while time.monotonic() < deadline:
        obs = Path(config["observations"])
        cache = Path(config["patch_cache"])
        ready = []
        if (obs / "DATA_READY.json").exists():
            record = json.loads((obs / "DATA_READY.json").read_text())
            if record["source_sha256sums"] != digest(obs / "SHA256SUMS"):
                raise ValueError("Observation readiness checksum differs")
            ready += jobs["fits"]
        if (cache / "CACHE_READY.json").exists():
            record = json.loads((cache / "CACHE_READY.json").read_text())
            if record["manifest_sha256"] != digest(cache / "manifest.json") or record[
                "grid_sha256"
            ] != digest(cache / "grid.npz"):
                raise ValueError("Patch cache readiness checksum differs")
            ready += jobs["patch_probes"]
        for job in ready:
            if job not in released:
                subprocess.run(["scontrol", "release", str(job)], check=True)
                released.append(job)
                receipt.write_text(json.dumps(released) + "\n")
                print(
                    json.dumps(dict(event="released_after_data_verification", job=job)),
                    flush=True,
                )
        if set(released) == set(jobs["fits"] + jobs["patch_probes"]):
            (root / "GATE_COMPLETE.json").write_text(
                json.dumps(dict(released=released)) + "\n"
            )
            return
        time.sleep(60)
    raise TimeoutError(
        "Data prerequisites did not finish within 48 hours; held jobs preserved"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (Path(sys.argv[1]) / "GATE_FAILED.json").write_text(
            json.dumps(dict(error=str(error))) + "\n"
        )
        raise
