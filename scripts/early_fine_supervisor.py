#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Execute only qualified, pinned stages and forward preemption warnings."""

import argparse
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--stage", choices=["prepare", "qualify", "train"], required=True
    )
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    paths = json.loads((root / "paths.json").read_text())
    if paths["producer"] != os.environ["SAMUDRA_CODE_COMMIT"]:
        raise ValueError("Wrong producer")
    runtime = json.loads((root / "runtime-contract.json").read_text())
    for name, expected in runtime["modules"].items():
        module = importlib.import_module(name)
        if getattr(module, "__version__", None) != expected["version"]:
            raise ValueError("Runtime version drift: " + name)
        for path, digest in expected["binaries"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError("Runtime binary drift: " + path)

    def run(script, arguments):
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name(script)), *arguments]
        )
        signal.signal(signal.SIGUSR1, lambda *_: child.send_signal(signal.SIGUSR1))
        signal.signal(signal.SIGTERM, lambda *_: child.send_signal(signal.SIGUSR1))
        code = child.wait()
        if code:
            raise SystemExit(code)

    if args.stage == "prepare":
        run("prepare_early_fine_wave.py", ["--root", args.root])
        return
    import torch

    from samudra.rust_data import create_rust_io_runtime

    create_rust_io_runtime(4)
    print(
        json.dumps(
            {
                "event": "runtime_verified",
                "producer": paths["producer"],
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "job": os.environ.get("SLURM_JOB_ID"),
            }
        ),
        flush=True,
    )
    from scripts.run_early_fine_wave import ARMS

    architecture, latent = ARMS[args.arm]
    if args.stage == "qualify":
        for stage, marker in [
            ("fit", root / f"fit-{architecture}-{latent}" / "QUALIFIED.json"),
            ("probe", root / f"probe-{args.arm}" / "JOINT_QUALIFIED.json"),
        ]:
            # Throughput is written after the probe marker: rerun if interrupted there.
            complete = marker.exists() and (
                stage != "probe" or (marker.parent / "THROUGHPUT.json").exists()
            )
            if not complete:
                run(
                    "run_early_fine_wave.py",
                    ["--root", args.root, "--arm", args.arm, "--stage", stage],
                )
    else:
        run(
            "run_early_fine_wave.py",
            ["--root", args.root, "--arm", args.arm, "--stage", "train"],
        )


if __name__ == "__main__":
    main()
