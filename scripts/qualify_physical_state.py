# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["PHYSICAL_ROOT"])
reference = Path(
    "/projects/ny/lz1955/multiscale/jrusak/runs/shared-era5-optimized-20261024/optimized/paths.json"
)
config = json.loads(reference.read_text())
legacy = json.loads(Path(config["legacy_qualification"]).read_text())
command = [
    sys.executable,
    "-m",
    "samudra.experiments.physical_state_train",
    "--root",
    str(root),
    "--output",
    str(root / "qualification"),
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
    "--om4-warmup-updates",
    "0",
    "--obs-cooldown-updates",
    "0",
    "--updates-per-task",
    "1",
    "--effective-batch",
    "1",
    "--checkpoint-every",
    "1",
    "--members",
    "2",
    "--sampling-steps",
    "8",
    "--max-new-updates",
    "2",
    "--hours",
    ".75",
    "--smoke",
]
subprocess.run(command, check=True)
events = [
    json.loads(line)
    for line in (root / "qualification/events.jsonl").read_text().splitlines()
]
if [e["task"] for e in events] != ["om4", "observation"]:
    raise ValueError("Both real tasks must complete")
(root / "QUALIFIED.json").write_text(
    json.dumps(
        {
            "events": events,
            "producer": os.environ["SAMUDRA_CODE_COMMIT"],
            "production_qualified": False,
            "scope": "batch1 full-grid forward/backward; production batch8 and replay not qualified",
        },
        indent=2,
    )
)
