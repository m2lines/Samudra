#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Four model-only pretraining arms, then their matched observation cooldowns."""

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
        arm = "ABCD"[index]
        command = [
            sys.executable,
            str(
                Path(__file__).with_name(
                    "qualify_monthly_pretraining.py"
                    if a.qualify
                    else "run_monthly_latent_pilot.py"
                )
            ),
            "--root",
            str(a.root),
            "--arm",
            arm,
            "--producer",
            config["producer"],
            "--qualification",
            config["qualification"],
        ]
        for key in ("observations", "om4", "cm4"):
            command.extend(["--" + key, config[key]])
        if not a.qualify:
            command.extend(
                [
                    "--model-only-pretraining",
                    "--pretraining-qualification",
                    str(a.root / "qualification" / arm / "QUALIFIED.json"),
                ]
            )
        stage = "qualification" if a.qualify else "production"
        with (a.root / "logs" / f"{stage}-{arm}-{os.environ['SLURM_JOB_ID']}.out").open(
            "w"
        ) as log:
            subprocess.run(
                command,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "0" if a.qualify else str(index),
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    if a.qualify:
        for index in range(4):
            worker(index)
    else:
        proofs = [
            json.loads((a.root / "qualification" / arm / "QUALIFIED.json").read_text())
            for arm in "ABCD"
        ]
        if len({p["initial_backbone_sha256"] for p in proofs}) != 1 or any(
            p["producer"] != config["producer"]
            or not p["exact_observation_cooldown_resume"]
            for p in proofs
        ):
            raise ValueError("All-four matching qualification has not passed")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, index) for index in range(4)]
            for future in futures:
                future.result()
        for arm in "ABCD":
            summary = json.loads(
                (a.root / "production" / arm / "TRAIN_COMPLETE.json").read_text()
            )
            if summary["step"] != 3000 or summary["counts"] != {
                "obs": 1000,
                "om4": 2000 if arm in "AB" else 1000,
                "cm4": 0 if arm in "AB" else 1000,
            }:
                raise ValueError("Completed data exposure differs from protocol")
        (a.root / "COMPLETE.json").write_text(
            json.dumps(
                dict(
                    producer=config["producer"],
                    arms=list("ABCD"),
                    pretraining=2000,
                    cooldown=1000,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
