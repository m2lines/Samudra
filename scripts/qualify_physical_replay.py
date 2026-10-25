# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Real-data replay qualification, including an optimizer/buffer resume."""

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["PHYSICAL_ROOT"])
reference = Path(os.environ["PHYSICAL_REFERENCE"])
config = json.loads(reference.read_text())
legacy = json.loads(Path(config["legacy_qualification"]).read_text())
command = [
    sys.executable,
    "-m",
    "samudra.experiments.physical_state_train",
    "--root",
    str(root),
    "--output",
    str(root / "replay-qualification"),
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
    "--replay-cache",
    config["replay_cache"],
    "--om4-warmup-updates",
    "0",
    "--obs-cooldown-updates",
    "0",
    "--updates-per-task",
    "4",
    "--effective-batch",
    "8",
    "--microbatch",
    "2",
    "--fused-adam",
    "--checkpoint-every",
    "2",
    "--members",
    "2",
    "--sampling-steps",
    "8",
    "--hours",
    "1",
    "--smoke",
]
subprocess.run(command + ["--max-new-updates", "4"], check=True)
subprocess.run(command + ["--resume", "--max-new-updates", "2"], check=True)
events = [
    json.loads(line)
    for line in (root / "replay-qualification/events.jsonl").read_text().splitlines()
]
last = events[-1]
if last["step"] != 6 or any(
    last["replay_examples"][task] == 0 for task in ("om4", "observation")
):
    raise ValueError("Both replay sources and resume must execute")
(root / "REPLAY_QUALIFIED.json").write_text(
    json.dumps(
        dict(
            events=events,
            producer=os.environ["SAMUDRA_CODE_COMMIT"],
            scope="six full-grid effective-batch-eight updates including both replay sources and resume; no long-horizon skill claim",
        ),
        indent=2,
    )
)
