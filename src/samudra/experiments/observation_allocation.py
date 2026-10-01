# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run independent observation arms inside one four-GPU beta allocation."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-contract", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 4 or not all(visible):
        raise ValueError(f"Expected four allocated visible GPUs, got {visible}")
    base = [
        "--data",
        args.data,
        "--checkpoint",
        args.checkpoint,
        "--source-contract",
        args.source_contract,
    ]
    reference = [
        "--selection-reference",
        str(root / "fitting/selection-reference.json"),
    ]
    active = {}
    finished = set()
    failures = {}
    started = set()

    def emit(event, **values):
        record = {
            "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "event": event,
            **values,
        }
        with (root / "allocation-events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def launch(name, gpu, extra=(), evaluate=False):
        started.add(name)
        marker = (
            "COMPLETE.json"
            if evaluate
            else ("QUALIFIED.json" if "fitting" in name else "TRAIN_COMPLETE.json")
        )
        if (root / name / marker).exists():
            finished.add(name)
            emit("already_complete", name=name)
            return
        if evaluate:
            arm = name.removesuffix("-evaluation")
            arguments = ["--run", str(root / arm), "--output", str(root / name)]
            module = "samudra.experiments.observation_evaluate"
        else:
            arguments = (
                base
                + ["--output", str(root / name), "--name", "obs-D-beta-" + name]
                + list(extra)
            )
            module = "samudra.experiments.observation_pilot"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = visible[gpu]
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
        ):
            environment.pop(key, None)
        log = (root / (name + "-process.log")).open("a")
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", module, *arguments],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        active[name] = (process, log, marker)
        emit("started", name=name, gpu=gpu, device=visible[gpu], pid=process.pid)

    def collect():
        for name, (process, log, marker) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[name]
            if code == 0 and (root / name / marker).exists():
                finished.add(name)
            else:
                failures[name] = code
            emit("exited", name=name, returncode=code, successful=name in finished)

    launch("fitting", 0, ["--fit-probe", "--wandb-mode", "disabled"])
    while active:
        time.sleep(5)
        collect()
    if failures:
        raise RuntimeError(f"Observation fitting gate failed: {failures}")
    qualified = ["--qualification", str(root / "fitting/QUALIFIED.json")] + reference
    launch("primary", 0, qualified)
    launch("adapter-only", 1, qualified + ["--adapter-only"])
    launch(
        "scratch-fitting",
        2,
        ["--fit-probe", "--from-scratch", "--wandb-mode", "disabled"] + reference,
    )
    while True:
        collect()
        if "scratch-fitting" in finished and "scratch" not in started:
            launch(
                "scratch",
                2,
                [
                    "--from-scratch",
                    "--qualification",
                    str(root / "scratch-fitting/QUALIFIED.json"),
                    "--reconstruction-hours",
                    "4",
                    "--joint-hours",
                    "4",
                ]
                + reference,
            )
        if not any(name.endswith("-evaluation") for name in active):
            for arm in ("primary", "adapter-only", "scratch"):
                name = arm + "-evaluation"
                if arm in finished and name not in started:
                    launch(name, 3, evaluate=True)
                    break
        pending_evaluation = any(
            arm in finished and arm + "-evaluation" not in started
            for arm in ("primary", "adapter-only", "scratch")
        )
        if not active and not pending_evaluation:
            break
        time.sleep(5)
    emit("allocation_finished", finished=sorted(finished), failures=failures)
    if failures:
        raise RuntimeError(f"One or more pilot processes failed: {failures}")


if __name__ == "__main__":
    main()
