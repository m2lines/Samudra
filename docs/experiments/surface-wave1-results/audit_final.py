# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Validate final wave artifacts, completion and allocated GPU accounting."""

import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, TypedDict

from artifact_io import open_artifact, read_artifact

R = Path(__file__).resolve().parent
collection = json.loads((R / "collection.json").read_text())
for name, entry in collection["files"].items():
    assert hashlib.sha256(read_artifact(R / name)).hexdigest() == entry["sha256"], name
jobs = json.loads((R / "jobs.json").read_text())


class AccountingRow(TypedDict):
    job: str
    state: str
    start_utc: str
    end_utc: str
    seconds: int
    gpus: int
    gpu_hours: float
    exit_code: str


account: list[AccountingRow] = []
events = []
for line in (R / "final-accounting.psv").read_text().splitlines():
    job, state, start, end, seconds, tres, exitcode = line.split("|")[:7]
    gpu = next(
        (int(x.split("=")[1]) for x in tres.split(",") if x.startswith("gres/gpu=")), 0
    )
    assert state.split()[0] in {
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
    }, (job, state)
    if gpu:
        t0 = datetime.datetime.fromisoformat(start)
        t1 = datetime.datetime.fromisoformat(end)
        assert (t1 - t0).total_seconds() == int(seconds), (job, seconds)
        events.extend([(t0, gpu, job), (t1, -gpu, job)])
    account.append(
        dict(
            job=job,
            state=state,
            start_utc=start,
            end_utc=end,
            seconds=int(seconds),
            gpus=gpu,
            gpu_hours=gpu * int(seconds) / 3600,
            exit_code=exitcode,
        )
    )
by_id = {r["job"]: r for r in account}
assert len(by_id) == len(account)
assert {v["id"] for v in jobs.values()} <= set(by_id)
for key in (
    "initializer-finalization",
    "ar-pretrain-finalization",
    "ar-joint",
    "direct-joint",
    "report",
):
    assert by_id[jobs[key]["id"]]["state"] == "COMPLETED", key
    assert by_id[jobs[key]["id"]]["exit_code"] == "0:0", key
assert by_id["17959017"]["state"].startswith("CANCELLED")
current = peak = 0
active = set()
peak_example = None
for time, delta, job in sorted(events, key=lambda e: (e[0], e[1])):
    current += delta
    if delta > 0:
        active.add(job)
    else:
        active.remove(job)
    if current > peak:
        peak = current
        peak_example = {"time_utc": time.isoformat() + "Z", "jobs": sorted(active)}
assert current == 0 and peak <= 8
used = sum(r["gpu_hours"] for r in account)
assert used <= 576
assert math.isclose(
    used, json.loads((R / "report.json").read_text())["gpu_hours"], rel_tol=1e-12
)
models = {}
for task, jobkey in [
    ("initializer", "initializer-finalization"),
    ("ar", "ar-joint"),
    ("direct", "direct-joint"),
]:
    manifest = json.loads((R / task / "manifest.json").read_text())
    assert manifest["code_commit"] == "fe0f314ebb562f5b658e49fedea9c5d27062d245"
    assert manifest["slurm_job_id"] == jobs[jobkey]["id"]
    assert manifest["arguments"]["max_steps"] == 0
    assert json.loads((R / task / "COMPLETE.json").read_text())["task"] == task
    if task == "initializer":
        rows = list(csv.DictReader((R / task / "initializer_metrics.csv").open()))
        assert len(rows) == 154
        assert len({(r["mode"], r["channel"]) for r in rows}) == 154
        assert all(math.isfinite(float(r["normalized_rmse"])) for r in rows)
        continue
    rows = list(csv.DictReader(open_artifact(R / task / "heldout_metrics.csv")))
    expected = {
        (mode, region, str(lead), channel)
        for mode in ("inferred", "true", "inferred_persistence", "climatology")
        for region in ("global", "tropics", "extratropics")
        for lead in (5, 10, 15, 20, 25, 30)
        for channel in manifest["channels"]
    }
    actual = {(r["mode"], r["region"], r["lead_days"], r["channel"]): r for r in rows}
    assert set(actual) == expected and len(actual) == len(rows) == 5544
    assert {r["origins"] for r in rows} == {"99"}
    assert all(
        math.isfinite(float(r[k]))
        for r in rows
        for k in (
            "normalized_rmse",
            "physical_rmse",
            "prediction_second_moment",
            "target_second_moment",
        )
    )
    assert manifest["world_size"] == 4
    models[task] = {"manifest": manifest, "rows": actual}
assert models["ar"]["manifest"]["data"] == models["direct"]["manifest"]["data"]
assert models["ar"]["manifest"]["channels"] == models["direct"]["manifest"]["channels"]
max_target_difference = max(
    abs(
        float(row["target_second_moment"])
        - float(models["direct"]["rows"][key]["target_second_moment"])
    )
    for key, row in models["ar"]["rows"].items()
)
assert max_target_difference == 0
for key, row in models["ar"]["rows"].items():
    if key[0] == "climatology":
        for field in (
            "normalized_rmse",
            "physical_rmse",
            "prediction_second_moment",
            "target_second_moment",
        ):
            assert row[field] == models["direct"]["rows"][key][field]
summary: dict[str, Any] = {
    "verified_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "scope": "OM4-only five-day 5–30-day hindcasts, not observations or continuous long rollouts",
    "source_files_sha256_verified": len(collection["files"]),
    "heldout_rows_per_model": 5544,
    "heldout_origins_per_model": 99,
    "target_second_moments_max_difference": max_target_difference,
    "climatology_metrics_identical": True,
    "terminal_jobs": len(account),
    "gpu_hours": used,
    "budget_gpu_hours": 576,
    "peak_concurrent_gpus": peak,
    "peak_example": peak_example,
    "ar_gpu_hours": sum(
        by_id[j]["gpu_hours"] for j in ["17977416", "18068421", "18068422", "18071751"]
    ),
    "direct_gpu_hours": sum(
        by_id[j]["gpu_hours"] for j in ["17977417", "18006099", "18006100", "18013772"]
    ),
    "initializer_gpu_hours": sum(
        by_id[j]["gpu_hours"] for j in ["17958872", "17971844", "17977027"]
    ),
    "passive_monitor_cancelled": True,
    "next_wave_submitted": False,
}
summary["other_smoke_and_recovery_gpu_hours"] = (
    used
    - summary["ar_gpu_hours"]
    - summary["direct_gpu_hours"]
    - summary["initializer_gpu_hours"]
)
(R / "completion-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
with (R / "accounting.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(account[0]))
    writer.writeheader()
    writer.writerows(account)
print(json.dumps(summary, indent=2))
lookup = {
    (r["model"], r["mode"], r["region"], int(r["lead_days"]), r["variable"]): float(
        r["normalized_rmse"]
    )
    for r in csv.DictReader((R / "grouped_metrics.csv").open())
}
for lead in (5, 15, 30):
    for var in ("thetao", "so", "zos", "sst"):
        ar = lookup["ar", "inferred", "global", lead, var]
        direct = lookup["direct", "inferred", "global", lead, var]
        truth = lookup["ar", "true", "global", lead, var]
        print(
            lead,
            var,
            "AR",
            ar,
            "Direct",
            direct,
            "AR vs Direct RMSE improvement",
            100 * (1 - ar / direct),
            "AR true-init vs inferred improvement",
            100 * (1 - truth / ar),
        )
