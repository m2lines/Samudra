#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Require annual audits and first-month qualification before parallel evaluation."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--evaluate-only", action="store_true")
    stages.add_argument("--qualify-only", action="store_true")
    a = parser.parse_args()
    config = json.loads((a.root / "paths.json").read_text())
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_monthly_annual.py")),
        "--root",
        str(a.root),
    ]
    if not a.evaluate_only:
        for stage in ["audit", "qualify"]:
            subprocess.run(command + ["--stage", stage], check=True)
    # Evaluation children independently recheck the qualified producer, audit
    # hash, configuration and checkpoint. A separate Slurm afterok dependency
    # lets short qualification run on beta_test without consuming a full node.
    qualified = json.loads((a.root / "QUALIFIED.json").read_text())
    if (
        qualified.get("first_month_readout_exact") is not True
        or qualified["producer"] != config["producer"]
    ):
        raise ValueError("Annual qualification has not passed")
    if a.qualify_only:
        return

    def worker(gpu):
        for index in range(gpu, len(config["models"]), 4):
            name = config["models"][index]["name"]
            with (a.root / "logs" / (name + ".out")).open("a") as log:
                subprocess.run(
                    command + ["--stage", "evaluate", "--index", str(index)],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, gpu) for gpu in range(4)]
        for future in futures:
            future.result()
    for row in config["models"]:
        proof = json.loads((a.root / row["name"] / "COMPLETE.json").read_text())
        if (
            proof["origins"] != config["origins"]
            or proof["producer"] != config["producer"]
        ):
            raise ValueError("Incomplete evaluation cohort")
    (a.root / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                producer=config["producer"],
                models=config["models"],
                origins=config["origins"],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
