# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Export compact scratch-wave provenance and diagnostic summaries for the report."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root, args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json.gz").write_bytes(
        (root / "scratch-summary/summary.json.gz").read_bytes()
    )
    native: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    curves: list[dict[str, Any]] = []
    for seed in (1729, 1730):
        name = f"B-{seed}"
        native[name], checkpoints[name] = {}, {}
        for phase, marker in (
            ("om4", "PRETRAIN_COMPLETE.json"),
            ("observation", "OBSERVATION_COMPLETE.json"),
        ):
            stage = root / "scratch-training-records" / name / phase
            complete = json.loads((stage / marker).read_text())
            if not complete["state"]["complete"]:
                raise ValueError("Incomplete training stage")
            selected = complete["best_checkpoint_sha256"]
            checkpoints[name][phase] = dict(
                checkpoint_prefix=selected[:16], **complete["state"]
            )
            stats = json.loads(
                (
                    root / "scratch-native-velocity" / name / (phase + "-summary.json")
                ).read_text()
            )
            if stats["protocol"]["checkpoint_sha256"] != selected:
                raise ValueError(
                    "Native diagnostics use a different selected checkpoint"
                )
            stats["protocol"] = {
                k: stats["protocol"][k] for k in ("origins", "members", "seed", "scope")
            }
            stats["checkpoint_prefix"] = selected[:16]
            native[name][phase] = stats
        for path in sorted(
            (root / "scratch-training-records" / name / "observation").glob(
                "validation-*.json"
            )
        ):
            record = json.loads(path.read_text())
            curves.append(
                dict(run=name, step=record["step"], objective=record["score"])
            )
    native_files = []
    for name, phases in native.items():
        for phase, values in phases.items():
            filename = f"native-velocity-{name}-{phase}.json.gz"
            (output / filename).write_bytes(
                gzip.compress(json.dumps(values, sort_keys=True).encode(), mtime=0)
            )
            native_files.append(filename)
    (output / "native-velocity-index.json").write_text(
        json.dumps(dict(files=native_files), indent=2) + "\n"
    )
    with (output / "validation-curves.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("run", "step", "objective"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(curves, key=lambda r: (r["run"], r["step"])))
    ledger = json.loads((root / "gpu-ledger.json").read_text())
    wave = json.loads((root / "scratch-wave.json").read_text())
    proofs = [
        json.loads((root / f"scratch-report-{seed}-verification.json").read_text())
        for seed in (1729, 1730)
    ]
    baseline = json.loads((root / "scratch-baseline-report/protocol.json").read_text())
    provenance = dict(
        training_producer=wave["producer"][:12],
        evaluator_producer=wave["evaluation_producer"][:12],
        baseline_evaluator_producer=wave["baseline_report_producer"][:12],
        velocity_evaluator_producer=wave["velocity_diagnostics"]["producer"][:12],
        baseline_checkpoint_prefix=baseline["checkpoint_sha256"][:16],
        training=checkpoints,
        start_utc=wave["start_utc"],
        deadline_utc=wave["deadline_utc"],
        allocations=[
            {
                k: v
                for k, v in a.items()
                if k
                in (
                    "job",
                    "state",
                    "gpus",
                    "elapsed_seconds",
                    "gpu_hours",
                    "attempt_start",
                )
            }
            for a in ledger["allocations"]
        ],
        completed_campaign_gpu_hours=ledger["total_gpu_hours"],
        scratch_wave_gpu_hours=ledger["total_gpu_hours"] - wave["prior_gpu_hours"],
        monthly_report_verification=[
            {k: p[k] for k in ("seed", "verified_files", "verified_bytes")}
            for p in proofs
        ],
        note="Short checkpoint identifiers for public navigation; full read-back SHA256 proofs retained with campaign outputs. Includes all earlier campaign allocations and preempted attempts.",
    )
    (output / "run-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
