# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Extend only a capped joint run's operational time allowance.

Launch beside the ORIGINAL qualified code overlay, not as its replacement.
The immutable training manifest continues to describe the original invocation;
RUNTIME_RECOVERY.json records the explicit runtime override and launcher hash.
"""

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extend_runtime(pilot, hours):
    """Leave the manifest, deadline, checkpoint state and science args untouched."""
    original = pilot.args.joint_hours
    if not math.isfinite(hours) or not original < hours <= original + 3:
        raise ValueError("Recovery may add at most three cumulative hours")
    if pilot.elapsed_seconds >= hours * 3600:
        raise ValueError("Recovery allowance already exhausted")
    # Pilot.manifest['arguments'] aliases vars(args); isolate the operational
    # override so old snapshots retain the exact same manifest/resume contract.
    pilot.args = copy.deepcopy(pilot.args)
    pilot.args.joint_hours = hours
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--hours", type=float, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if sha256(__file__) != args.launcher_sha256:
        raise ValueError("Runtime launcher hash differs")
    manifest_path = run / "manifest.json"
    if sha256(manifest_path) != args.manifest_sha256:
        raise ValueError("Original manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["code_commit"] != os.environ.get("SAMUDRA_CODE_COMMIT"):
        raise ValueError("Must use the original qualified training producer")
    if Path(manifest["arguments"]["output"]).resolve() != run:
        raise ValueError("Run path differs")
    if (run / "TRAIN_COMPLETE.json").exists():
        raise ValueError("Refusing to resume a completed run")
    deadline = datetime.datetime.fromisoformat(manifest["arguments"]["deadline"])
    if datetime.datetime.now(datetime.UTC) >= deadline:
        raise ValueError("Original deadline passed")
    partial = json.loads((run / "TRAIN_PARTIAL.json").read_text())
    if partial["reason"] != "wall-time limit":
        raise ValueError("Not a runtime-limited partial run")
    record_path = run / "RUNTIME_RECOVERY.json"
    contract = {
        "original_manifest_sha256": args.manifest_sha256,
        "launcher_sha256": args.launcher_sha256,
        "training_producer": manifest["code_commit"],
        "original_hours": manifest["arguments"]["joint_hours"],
        "recovery_hours": args.hours,
        "deadline": manifest["arguments"]["deadline"],
    }
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["contract"] != contract:
            raise ValueError("Recovery contract changed")
    else:
        archive = run / "before-runtime-recovery"
        archive.mkdir(exist_ok=True)
        for name in ["manifest.json", "TRAIN_PARTIAL.json", "joint-last.pt"]:
            target = archive / name
            if not target.exists():
                os.link(run / name, target)
        record = {
            "contract": contract,
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "original_partial": partial,
            "original_resume_sha256": sha256(archive / "joint-last.pt"),
        }
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(record_path)
    # These imports come from the unchanged, qualified, read-only code overlay.
    from samudra.experiments.observation_joint import JointPilot

    pilot = JointPilot(argparse.Namespace(**manifest["arguments"]))
    extend_runtime(pilot, args.hours)
    assert pilot.manifest == manifest
    pilot.emit(
        {"event": "runtime_recovery", **contract, "global_step": pilot.completed}
    )
    pilot.run_joint()
    assert pilot.manifest == manifest
    if (run / "TRAIN_COMPLETE.json").exists():
        # The archived partial marker remains; the live run is now complete.
        (run / "TRAIN_PARTIAL.json").unlink()


if __name__ == "__main__":
    main()
