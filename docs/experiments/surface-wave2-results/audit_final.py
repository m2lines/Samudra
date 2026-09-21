# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the published study's checkpoint lineage and all-attempt allocation ledger."""

import csv
import datetime as dt
import io
import json
import math
from pathlib import Path
from typing import TypedDict

from samudra.experiments.surface_adaptation_analysis import read_bytes

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
EXPECTED = {
    "18079867": "qualification-1",
    "18079868": "qualification-1",
    "18079869": "qualification-1",
    "18094385": "container-restore-cpu",
    "18096965": "qualification-2",
    "18096966": "qualification-2",
    "18096967": "qualification-2",
    "18097624": "A",
    "18097625": "B",
    "18097626": "C",
    "18097627": "D",
    "18097633": "E",
    "18160437": "F",
    "18160463": "checkpoint-audit-cpu",
}
PRODUCER = "b46eb446f153a967869bbfc7f9e93783de986579"  # pragma: allowlist secret -- public producer Git SHA
DEADLINE = dt.datetime.fromisoformat("2026-09-23T16:51:29+00:00")
LICENSE = "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"


class AccountingRow(TypedDict):
    job: str
    role: str
    state: str
    start_utc: str
    end_utc: str
    seconds: int
    gpus: int
    gpu_hours: float
    exit_code: str


def load(relative):
    return json.loads(read_bytes(RAW / relative))


def main():
    rows = list(
        csv.DictReader(
            io.StringIO(read_bytes(RAW / "final-accounting.csv").decode()),
            delimiter="|",
        )
    )
    assert len(rows) == len(EXPECTED)
    assert {row["JobID"] for row in rows} == set(EXPECTED)
    accounting: list[AccountingRow] = []
    events = []
    for row in rows:
        job = row["JobID"]
        state = row["State"].split()[0]
        assert state in {"COMPLETED", "CANCELLED", "FAILED"}, row
        if EXPECTED[job] != "qualification-1":
            assert state == "COMPLETED" and row["ExitCode"] == "0:0", row
        seconds = int(row["ElapsedRaw"])
        tres = dict(
            item.split("=", 1) for item in row["AllocTRES"].split(",") if "=" in item
        )
        gpus = int(tres.get("gres/gpu", 0))
        if seconds:
            start = dt.datetime.fromisoformat(row["Start"]).replace(tzinfo=dt.UTC)
            end = dt.datetime.fromisoformat(row["End"]).replace(tzinfo=dt.UTC)
            assert (end - start).total_seconds() == seconds, row
            assert end < DEADLINE, row
            if gpus:
                events.extend([(start, gpus, job), (end, -gpus, job)])
        accounting.append(
            dict(
                job=job,
                role=EXPECTED[job],
                state=state,
                start_utc=row["Start"],
                end_utc=row["End"],
                seconds=seconds,
                gpus=gpus,
                gpu_hours=seconds * gpus / 3600,
                exit_code=row["ExitCode"],
            )
        )
    current = peak = 0
    for _, delta, _ in sorted(events, key=lambda event: (event[0], event[1])):
        current += delta
        peak = max(peak, current)
        assert current >= 0
    assert current == 0 and peak <= 8
    audit = json.loads((ROOT / "analysis" / "analysis-audit.json").read_text())
    assert set(audit["arms"]) == set("ABCDEF")
    assert audit["channel_rows_per_arm"] == 8316
    assert audit["origin_rows_per_arm"] == 64152
    checkpoints = {}
    for arm in "ABCDEF":
        manifest = load(f"{arm}/manifest.json")
        marker = load(f"{arm}/COMPLETE.json")
        initial = load(f"{arm}/initialization.json")
        check = load(f"{arm}/checkpoint-audit.json")
        assert manifest["code_commit"] == PRODUCER
        assert EXPECTED[manifest["slurm_job_id"]] == arm
        assert check["completion"] == marker
        assert math.isclose(
            check["selected"]["state"]["best"], marker["selected_ts_mse"], rel_tol=1e-6
        )
        assert check["last"]["state"]["complete"]
        frozen = {"A": "evolution", "B": "initializer"}.get(arm)
        assert check["frozen_component_verified"] == frozen
        if frozen:
            assert (
                check["selected"][frozen + "_fingerprint"]
                == initial["initial_" + frozen + "_fingerprint"]
            )
        assert dt.datetime.fromisoformat(marker["time_utc"]) < DEADLINE
        checkpoints[arm] = check["selected"]
    gpu_hours = sum(row["gpu_hours"] for row in accounting)
    assert gpu_hours <= 112  # Four-arm envelope 80 plus two controls capped at 16 each.
    output = dict(
        terminal_jobs=len(rows),
        gpu_hours=gpu_hours,
        peak_concurrent_gpus=peak,
        gpu_hours_by_role={
            role: sum(r["gpu_hours"] for r in accounting if r["role"] == role)
            for role in sorted(set(EXPECTED.values()))
        },
        deadline_utc=DEADLINE.isoformat(),
        all_jobs_before_deadline=True,
        checkpoint_producer=PRODUCER,
        selected_checkpoints=checkpoints,
        boundary="CPU checkpoint loading verified on Torch; this local audit verifies its recorded evidence, not remote checkpoint bytes anew.",
    )
    target = ROOT / "completion-audit.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    with (ROOT / "accounting.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accounting[0]))
        writer.writeheader()
        writer.writerows(accounting)
    for path in [target, ROOT / "accounting.csv"]:
        Path(str(path) + ".license").write_text(LICENSE)
    print(
        f"PASS: {len(rows)} terminal jobs, {gpu_hours:.6f} GPU-hours, peak {peak} GPUs; six checkpoint lineages verified"
    )


if __name__ == "__main__":
    main()
