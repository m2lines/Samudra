#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""One four-GPU OBS-loss ablation with a shared, fail-closed qualification gate."""

import argparse
import fcntl
import json
import math
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.root / "paths.json").read_text())
    if len(config["runs"]) != 1:
        raise ValueError("This launch is one authorized reweighting run")
    row = config["runs"][0]
    if row["mode"] != "shared-4k" or row["observation_loss_weights"] != [0.5, 0.5, 0.1]:
        raise ValueError("Unexpected experiment protocol")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(Path(__file__).with_name("run_monthly_replay.py")),
        "--root",
        str(args.root),
        "--name",
        row["name"],
    ]
    proof = args.root / "qualification" / row["name"] / "QUALIFIED.json"
    failure = args.root / "QUALIFICATION_FAILED.json"
    # Both schedulers may start; only one qualifier writes or executes at once.
    with (args.root / "qualification.lock").open("a") as lock:
        try:
            fcntl.flock(
                lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if args.qualification_only else 0)
            )
        except BlockingIOError:
            print(
                "Qualification already owned by production; no duplicate work.",
                flush=True,
            )
            return
        if failure.exists():
            raise RuntimeError(
                "Prior qualification failed; preserve evidence for recovery"
            )
        if not proof.exists():
            try:
                subprocess.run(command + ["--qualify"], check=True)
            except subprocess.CalledProcessError as error:
                failure.write_text(json.dumps(dict(returncode=error.returncode)) + "\n")
                raise
        qualification = json.loads(proof.read_text())
        if not qualification["exact_serialized_update"]:
            raise ValueError("Exact resume qualification missing")
        for record in qualification["records"]:
            if record["source"] == "obs":
                expected = sum(
                    w * record["train/obs/" + k]
                    for w, k in zip(
                        row["observation_loss_weights"],
                        ("interior", "surface", "reconstruction"),
                        strict=True,
                    )
                )
                if not math.isclose(
                    record["loss"], expected, rel_tol=2e-6, abs_tol=1e-6
                ):
                    raise ValueError(
                        "Actual OBS training weights differ from approved weights"
                    )
    if not args.qualification_only:
        # Runner independently checks exact producer/config/protocol equality.
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
