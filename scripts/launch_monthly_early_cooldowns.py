#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct verified control prefix, then run three matched early cooldowns."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ["root", "source", "parent"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--producer", required=True)
    a = p.parse_args()
    (a.root / "logs").mkdir(parents=True, exist_ok=True)
    source = a.source / "production"
    paths = json.loads((source / "clip1/protocol.json").read_text())
    stream = source / "clip1/endpoint-02000.pt"
    contracts = {
        str(f): sha(f)
        for f in [stream, source / "clip1/fork-00000.pt", source / "clip1/events.jsonl"]
    }
    contract_path = a.root / "SOURCE_CONTRACTS.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contracts:
        raise ValueError("Source changed")
    contract_path.write_text(json.dumps(contracts, indent=2))

    def execute(command, gpu, label):
        with (a.root / "logs" / (label + ".out")).open("a") as log:
            subprocess.run(
                command,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def train(root, gpu, extra):
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_monthly_clipping_continuation.py")),
            "--arm",
            "clip1",
            "--root",
            str(root),
            "--seed",
            "1730",
            "--producer",
            a.producer,
            "--qualification",
            str(a.parent / "initial-qualification-214096/QUALIFIED.json"),
        ]
        for key in ["observations", "om4", "cm4"]:
            command += ["--" + key, paths[key]]
        command += extra
        execute(command + ["--qualify"], gpu, root.name + "-qualify")
        execute(command, gpu, root.name + "-train")

    def replay_and_cooldowns():
        replay = a.root / "prefix-replay"
        train(
            replay,
            0,
            ["--reproduce-pilot", "--replay-prefix-reference", str(source / "clip1")],
        )
        proof = json.loads(
            (replay / "production/clip1/REPLAY_VERIFIED.json").read_text()
        )
        if proof["steps"] != 1000 or proof["producer"] != a.producer:
            raise ValueError("Prefix not verified")

        def cooldown(index, step):
            checkpoint = (
                source / "clip1/fork-00000.pt"
                if step == 0
                else replay / f"production/clip1/endpoint-{step:05d}.pt"
            )
            train(
                a.root / f"cooldown-{step}",
                [0, 2, 3][index],
                [
                    "--early-cooldown-step",
                    str(step),
                    "--source-checkpoint",
                    str(checkpoint),
                    "--source-sha256",
                    sha(checkpoint),
                    "--cooldown-stream-checkpoint",
                    str(stream),
                    "--cooldown-stream-sha256",
                    contracts[str(stream)],
                ],
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(cooldown, index, step)
                for index, step in enumerate([0, 100, 1000])
            ]
            for future in futures:
                future.result()

    diagnostic = [
        sys.executable,
        str(Path(__file__).with_name("diagnose_monthly_convergence.py")),
        "--source",
        str(source),
        "--out",
        str(a.root / "diagnostics"),
        "--producer",
        a.producer,
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(replay_and_cooldowns),
            pool.submit(execute, diagnostic, 1, "diagnostics"),
        ]
        for future in futures:
            future.result()
    (a.root / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                producer=a.producer,
                cooldown_starts=[0, 100, 1000],
                diagnostic_complete=True,
            )
        )
    )


if __name__ == "__main__":
    main()
