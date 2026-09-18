#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Produce a wave report from completed or failed jobs; never submits jobs."""

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(root, jobs):
    root = Path(root)
    lines = [
        "# Surface-initialized ocean prediction: wave 1",
        "",
        "OM4-only, five-day-cadence, 5–30-day hindcasts. These results do not",
        "measure observational skill, daily prediction, or an eight-year rollout.",
        "",
    ]
    runs: dict[str, dict[str, Any]] = {}
    for task in ("initializer", "ar", "direct"):
        directory = root / task
        complete = (directory / "COMPLETE.json").exists()
        lines += [
            f"## {task}",
            "",
            f"Completion marker: **{'present' if complete else 'absent'}**.",
            "",
        ]
        manifest = directory / "manifest.json"
        if manifest.exists():
            info = json.loads(manifest.read_text())
            lines += [
                f"Code: `{info['code_commit']}`; Slurm job: `{info['slurm_job_id']}`.",
                "",
            ]
        progress = directory / "progress.jsonl"
        records = (
            [json.loads(line) for line in progress.read_text().splitlines()]
            if progress.exists()
            else []
        )
        for phase in ("initializer", "pretrain", "joint"):
            progress_scores = [
                r["best_validation_loss"]
                for r in records
                if r.get("phase") == phase and "best_validation_loss" in r
            ]
            if progress_scores:
                lines += [
                    f"{phase} best validation loss: {min(progress_scores):.6g}.",
                    "",
                ]
        peaks = [r.get("peak_gpu_gib", 0) for r in records]
        if peaks:
            lines += [f"Peak allocated GPU memory (rank 0): {max(peaks):.2f} GiB.", ""]
        init_metrics = directory / "initializer_metrics.csv"
        if init_metrics.exists():
            interior = defaultdict(list)
            with init_metrics.open() as file:
                for row in csv.DictReader(file):
                    if row["channel"] in ("thetao_0", "zos"):
                        continue
                    variable = row["channel"].split("_")[0]
                    interior[(row["mode"], variable)].append(
                        float(row["normalized_rmse"]) ** 2
                    )
            lines += [
                "Initializer validation RMSE (surface channels excluded):",
                "",
                "| Mode | T | S | u | v |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for mode in ("inferred", "climatology_with_observed_surface"):
                values = [
                    math.sqrt(sum(interior[(mode, v)]) / len(interior[(mode, v)]))
                    for v in ("thetao", "so", "uo", "vo")
                ]
                lines.append(
                    "| " + mode + " | " + " | ".join(f"{v:.4f}" for v in values) + " |"
                )
            lines += [""]
        metrics = directory / "heldout_metrics.csv"
        if not metrics.exists():
            lines += [
                "No held-out metrics available; no skill conclusion can be drawn.",
                "",
            ]
            runs[task] = {"complete": complete}
            continue
        groups = defaultdict(list)
        velocity: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
        with metrics.open() as file:
            for row in csv.DictReader(file):
                if row["region"] != "global":
                    continue
                variable = (
                    "sst"
                    if row["channel"] == "thetao_0"
                    else row["channel"].split("_")[0]
                )
                key = (row["mode"], int(row["lead_days"]), variable)
                groups[key].append(float(row["normalized_rmse"]) ** 2)
                if variable in ("uo", "vo"):
                    velocity[(row["mode"], int(row["lead_days"]))][0] += float(
                        row["prediction_second_moment"]
                    )
                    velocity[(row["mode"], int(row["lead_days"]))][1] += float(
                        row["target_second_moment"]
                    )
        scores = {
            key: math.sqrt(sum(values) / len(values)) for key, values in groups.items()
        }
        lines += [
            "Global normalized RMSE, equal weight per depth level:",
            "",
            "| Mode | Lead (days) | Subsurface T | S | u | v | SSH | SST |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode in ("inferred", "true", "inferred_persistence", "climatology"):
            for lead in (5, 15, 30):
                values = [
                    scores[(mode, lead, var)]
                    for var in ("thetao", "so", "uo", "vo", "zos", "sst")
                ]
                lines.append(
                    f"| {mode} | {lead} | "
                    + " | ".join(f"{v:.4f}" for v in values)
                    + " |"
                )
        velocity_moments = velocity[("inferred", 30)]
        ratio = (
            velocity_moments[0] / velocity_moments[1]
            if velocity_moments[1]
            else float("nan")
        )
        lines += [
            "",
            f"30-day inferred velocity second-moment ratio to OM4: {ratio:.3f}.",
            "",
            "This amplitude diagnostic does not establish balanced or realistic circulation.",
            "",
        ]
        runs[task] = {
            "complete": complete,
            "scores": {"/".join(map(str, k)): v for k, v in scores.items()},
            "velocity_second_moment_ratio": ratio,
        }
    lines += ["## Accounting", ""]
    accounting = subprocess.run(
        [
            "sacct",
            "-j",
            jobs,
            "-P",
            "--noheader",
            "--allocations",
            "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    total = 0.0
    for account_row in accounting.splitlines():
        job, state, seconds, tres, exit_code = account_row.split("|")[:5]
        gpu = 0
        for item in tres.split(","):
            if item.startswith("gres/gpu="):
                gpu = int(item.split("=")[1])
        hours = gpu * int(seconds) / 3600
        total += hours
        lines.append(
            f"- Job {job}: {state}, exit {exit_code}, {hours:.3f} allocated GPU-hours."
        )
    lines += [
        "",
        f"Total allocated GPU-hours in supplied job IDs: **{total:.3f} / 576**.",
        "",
        "Include every failed/replaced attempt in the job-ID list before approving recovery.",
        "",
        "## Suggested next wave — requires approval",
        "",
    ]
    if not all(
        runs.get(k, {}).get("complete") for k in ("initializer", "ar", "direct")
    ):
        lines += [
            "One or more jobs lack completion markers. Resolve their failure or pending state",
            "and validate checkpoint/output provenance before interpreting model differences.",
        ]
    else:
        ranking = []
        for task in ("ar", "direct"):
            task_scores = runs[task]["scores"]
            joint = (
                sum(task_scores[f"inferred/30/{v}"] ** 2 for v in ("thetao", "so")) / 2
            )
            persistence = (
                sum(
                    task_scores[f"inferred_persistence/30/{v}"] ** 2
                    for v in ("thetao", "so")
                )
                / 2
            )
            ranking.append((joint, task))
            lines.append(
                f"- {task}: 30-day T/S MSE improvement over inferred persistence: {100 * (1 - joint / persistence):.1f}%."
            )
        ranking.sort()
        lines += [
            f"- Prioritize `{ranking[0][1]}` for replication if its advantage persists by depth and region.",
            "- Inspect the true-versus-inferred validation gap to decide whether to spend the next wave on the initializer or evolution.",
            "- Check velocity maps, spectra, and temporal continuity before claiming plausible circulation.",
            "- Once prepared, introduce ERA5 surface-state forcing and observation initialization/interior evaluation, with daily SSH/SST heads.",
        ]
    lines += ["", "No next-wave jobs are submitted by this report.", ""]
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.md").write_text("\n".join(lines))
    (root / "report.json").write_text(
        json.dumps(
            {"runs": runs, "gpu_hours": total, "accounting": accounting}, indent=2
        )
    )
    return root / "report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--jobs", required=True)
    args = parser.parse_args()
    print(summarize(args.root, args.jobs))
