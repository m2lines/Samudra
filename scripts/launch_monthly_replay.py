#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Qualify sequentially on beta_test; train four matched arms on one Beta node."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--qualify", action="store_true")
    a = p.parse_args()
    config = json.loads((a.root / "paths.json").read_text())
    (a.root / "logs").mkdir(exist_ok=True)

    def worker(index):
        name = config["runs"][index]["name"]
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_monthly_replay.py")),
            "--root",
            str(a.root),
            "--name",
            name,
        ]
        if a.qualify:
            command.append("--qualify")
        stage = "qualification" if a.qualify else "production"
        with (
            a.root / "logs" / f"{stage}-{name}-{os.environ['SLURM_JOB_ID']}.out"
        ).open("w") as stream:
            subprocess.run(
                command,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "0" if a.qualify else str(index),
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    if a.qualify:
        for index in range(4):
            worker(index)
    else:
        proofs = [
            json.loads(
                (a.root / "qualification" / row["name"] / "QUALIFIED.json").read_text()
            )
            for row in config["runs"]
        ]
        if len({p["initial_random_backbone_sha256"] for p in proofs}) != 1 or not all(
            p["exact_serialized_update"] for p in proofs
        ):
            raise ValueError("Qualification/initialization contracts differ")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, index) for index in range(4)]
            for future in futures:
                future.result()
        (a.root / "COMPLETE.json").write_text(
            json.dumps(dict(producer=config["producer"], runs=config["runs"]), indent=2)
            + "\n"
        )


if __name__ == "__main__":
    main()
