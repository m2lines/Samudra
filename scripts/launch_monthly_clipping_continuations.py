#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Qualify all three forks before launching their matched continuations."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--parent", required=True)
    p.add_argument("--producer", required=True)
    a = p.parse_args()
    root = Path(a.root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    parent = Path(a.parent)
    paths = json.loads((parent / "A/protocol.json").read_text())

    def worker(gpu, arm, qualify):
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_monthly_clipping_continuation.py")),
            "--arm",
            arm,
            "--root",
            a.root,
            "--producer",
            a.producer,
            "--qualification",
            str(parent / "initial-qualification-214096/QUALIFIED.json"),
            "--source-checkpoint",
            str(parent / "A/endpoint-02000.pt"),
            "--source-sha256",
            "3888a09b570d1efcd362d07162f746ce484010b651335421732786cea3412bdf",  # pragma: allowlist secret
        ]
        for key in ("observations", "om4", "cm4"):
            command.extend(["--" + key, paths[key]])
        if qualify:
            command.append("--qualify")
        with (
            root
            / "logs"
            / f"{'qualification' if qualify else 'production'}-{arm}-{os.environ['SLURM_JOB_ID']}.out"
        ).open("w") as log:
            result = subprocess.run(
                command,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return arm, result.returncode

    for qualify in (True, False):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(worker, i, arm, qualify)
                for i, arm in enumerate(("clip1", "clip5", "adapter"))
            ]
            results = [f.result() for f in futures]
        print(json.dumps({"qualification": qualify, "results": results}), flush=True)
        if any(code for _, code in results):
            raise RuntimeError("Stage failed; no later stage launched")
    (root / "COMPLETE.json").write_text(
        json.dumps({"producer": a.producer, "arms": ["clip1", "clip5", "adapter"]})
    )


if __name__ == "__main__":
    main()
