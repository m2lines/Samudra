# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Hardware gate followed by the approved physical-state replay pilot."""

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["PHYSICAL_ROOT"])
reference = Path(os.environ["PHYSICAL_REFERENCE"])
config = json.loads(reference.read_text())
legacy = json.loads(Path(config["legacy_qualification"]).read_text())
base = [
    sys.executable,
    "-m",
    "samudra.experiments.physical_state_train",
    "--root",
    str(root),
    "--reference",
    str(reference),
    "--data",
    config["samples"],
    "--om4",
    config["om4"],
    "--packed-om4",
    legacy["packed_om4"],
    "--shared-era5",
    config["forcing_cache"],
    "--shared-era5-proof-sha256",
    config["forcing_proof_sha256"],
    "--effective-batch",
    "8",
    "--microbatch",
    "8",
    "--fused-adam",
    "--members",
    "2",
    "--sampling-steps",
    "8",
]
production = root / "production"
if not (root / "HARDWARE_QUALIFIED.json").exists():
    subprocess.run(
        base
        + [
            "--output",
            str(root / "hardware-check"),
            "--smoke",
            "--om4-warmup-updates",
            "0",
            "--obs-cooldown-updates",
            "0",
            "--updates-per-task",
            "1",
            "--checkpoint-every",
            "1",
            "--hours",
            "1",
        ],
        check=True,
    )
    events = [
        json.loads(line)
        for line in (root / "hardware-check/events.jsonl").read_text().splitlines()
    ]
    if [e["task"] for e in events] != ["om4", "observation"]:
        raise ValueError("Both hardware qualification tasks must complete")
    (root / "HARDWARE_QUALIFIED.json").write_text(json.dumps(events, indent=2))
command = base + [
    "--output",
    str(production),
    "--replay-cache",
    config["replay_cache"],
    "--om4-warmup-updates",
    "100",
    "--updates-per-task",
    "500",
    "--obs-cooldown-updates",
    "100",
    "--checkpoint-every",
    "50",
    "--hours",
    "23",
]
if (production / "last.pt").exists():
    command += ["--resume"]
subprocess.run(command, check=True)
