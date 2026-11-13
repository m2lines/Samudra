# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize matched training runs without conflating throughput and model quality."""

import csv
import datetime
import json
import statistics
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

root = Path(sys.argv[1])
result: dict[str, Any] = {}
for arm in ["local", "remote"]:
    summaries = [
        json.loads((root / arm / f"COMPLETE-rank{rank}.json").read_text())
        for rank in range(8)
    ]
    batches = [
        [
            json.loads(line)
            for line in (root / arm / f"batches-rank{rank}.jsonl")
            .read_text()
            .splitlines()
        ]
        for rank in range(8)
    ]
    measured = [[row for row in rows if row["measured"]] for rows in batches]
    assert len({s["measured_batches"] for s in summaries}) == 1
    assert all(len(rows) == summaries[0]["measured_batches"] for rows in measured)
    waits = [row["train/batch/data_wait_time"] for rows in measured for row in rows]
    walls = [row["wall_seconds"] for rows in measured for row in rows]
    with (root / f"gpu-{arm}.csv").open() as stream:
        gpu_rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    gpu_use = []
    for row in gpu_rows:
        timestamp = (
            datetime.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
            .replace(tzinfo=ZoneInfo("America/New_York"))
            .timestamp()
        )
        if (
            summaries[0]["measured_start_unix"]
            <= timestamp
            <= summaries[0]["measured_end_unix"]
        ):
            gpu_use.append(float(row["utilization.gpu [%]"].split()[0]))
    assert gpu_use, "Missing GPU utilization samples in the measured interval"
    result[arm] = dict(
        summary=summaries[0],
        mean_batch_wall_seconds=statistics.mean(walls),
        median_data_wait_seconds=statistics.median(waits),
        p95_data_wait_seconds=statistics.quantiles(waits, n=100)[94],
        mean_data_wait_seconds=statistics.mean(waits),
        data_wait_fraction=sum(waits) / sum(walls),
        mean_gpu_utilization_percent=statistics.mean(gpu_use),
        gpu_utilization_samples=len(gpu_use),
        first_losses=[row["train/batch/loss"] for row in batches[0][:10]],
    )
local = json.loads((root / "local/COMPLETE-rank0.json").read_text())
remote = json.loads((root / "remote/COMPLETE-rank0.json").read_text())
for rank in range(8):
    a = json.loads((root / "local" / f"COMPLETE-rank{rank}.json").read_text())
    b = json.loads((root / "remote" / f"COMPLETE-rank{rank}.json").read_text())
    assert a["initial_state_sha256"] == b["initial_state_sha256"]
    assert a["sampler_sha256"] == b["sampler_sha256"]
    assert a["host"] == b["host"]
result["remote_over_local_throughput"] = (
    remote["sample_windows_per_second"] / local["sample_windows_per_second"]
)
result["matched_initialization_and_sampler"] = True
local_rows = [
    json.loads(line)
    for line in (root / "local/batches-rank0.jsonl").read_text().splitlines()
]
remote_rows = [
    json.loads(line)
    for line in (root / "remote/batches-rank0.jsonl").read_text().splitlines()
]
prefix = list(zip(local_rows, remote_rows))
result["common_prefix_batches"] = len(prefix)
result["common_prefix_max_absolute_loss_difference"] = max(
    abs(a["train/batch/loss"] - b["train/batch/loss"]) for a, b in prefix
)
(root / "COMPARISON_COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
