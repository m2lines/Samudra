# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize true-lead latent-autoregressive W&B proxy runs."""

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

import wandb

LEADS = (1, 2, 4)
BOUNDARY_ABLATIONS = ("zero", "batch_shuffle", "time_reverse")
ZERO_DEPTH_KEY = "val/zero_depth_reconstruction/mean/loss"


def lead_key(depth: int) -> str:
    return f"val/physical_lead_{depth}/mean/loss"


def ablation_key(mode: str, depth: int) -> str:
    return f"val/boundary_{mode}/physical_lead_{depth}/mean/loss"


def persistence_key(depth: int) -> str:
    return f"val/physical_lead_{depth}/persistence/mean/loss"


def validate_run_config(config: Mapping[str, Any]) -> None:
    train_config = config.get("config")
    if not isinstance(train_config, Mapping):
        raise ValueError("W&B run does not contain a nested training config")
    if train_config.get("target_time_mode") != "forecast":
        raise ValueError("Latent-AR summary requires forecast targets")
    if train_config.get("train_processor_depths") != list(LEADS):
        raise ValueError(f"Expected train_processor_depths={list(LEADS)}")
    if train_config.get("validation_processor_depths") != list(LEADS):
        raise ValueError(f"Expected validation_processor_depths={list(LEADS)}")


def select_best_row(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the finite row with lowest true one-step validation MSE."""
    key = lead_key(1)
    valid = [
        dict(row)
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
    ]
    if not valid:
        raise ValueError(f"No finite `{key}` found")
    return min(valid, key=lambda row: row[key])


def summarize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "epoch": row.get("epoch"),
        "step": row.get("_step"),
        "zero_depth_reconstruction": row.get(ZERO_DEPTH_KEY),
    }
    for depth in LEADS:
        aligned = row.get(lead_key(depth))
        persistence = row.get(persistence_key(depth))
        summary[f"lead_{depth}"] = aligned
        summary[f"persistence_lead_{depth}"] = persistence
        if (
            isinstance(aligned, (int, float))
            and math.isfinite(aligned)
            and isinstance(persistence, (int, float))
            and math.isfinite(persistence)
            and persistence > 0
        ):
            summary[f"lead_{depth}_persistence_reduction"] = 1.0 - (
                aligned / persistence
            )
        for mode in BOUNDARY_ABLATIONS:
            ablated = row.get(ablation_key(mode, depth))
            summary[f"{mode}_lead_{depth}"] = ablated
            if (
                isinstance(aligned, (int, float))
                and math.isfinite(aligned)
                and aligned > 0
                and isinstance(ablated, (int, float))
                and math.isfinite(ablated)
            ):
                summary[f"{mode}_lead_{depth}_relative_increase"] = (
                    ablated / aligned - 1.0
                )
    return summary


def summarize_run(run: Any) -> dict[str, Any]:
    validate_run_config(run.config)
    keys = [
        "epoch",
        "_step",
        ZERO_DEPTH_KEY,
        *(lead_key(depth) for depth in LEADS),
        *(persistence_key(depth) for depth in LEADS),
        *(ablation_key(mode, depth) for mode in BOUNDARY_ABLATIONS for depth in LEADS),
    ]
    row = select_best_row(run.scan_history(keys=keys, page_size=1000))
    return {
        "path": run.path,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        **summarize_row(row),
    }


def markdown_table(summaries: Iterable[Mapping[str, Any]]) -> str:
    columns = [
        "epoch",
        "zero_depth_reconstruction",
        *(f"lead_{depth}" for depth in LEADS),
        *(f"lead_{depth}_persistence_reduction" for depth in LEADS),
        *(
            f"{mode}_lead_{depth}_relative_increase"
            for mode in BOUNDARY_ABLATIONS
            for depth in LEADS
        ),
    ]
    lines = [
        "| Run | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for summary in summaries:
        values = []
        for column in columns:
            value = summary.get(column)
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append(
            f"| [{summary['name']}]({summary['url']}) | " + " | ".join(values) + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="W&B run paths as entity/project/run_id.",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api()
    summaries = [summarize_run(api.run(path)) for path in args.runs]
    if args.format == "json":
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        print(markdown_table(summaries))


if __name__ == "__main__":
    main()
