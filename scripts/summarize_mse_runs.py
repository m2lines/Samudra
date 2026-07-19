# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize the best epoch from one-step, plain-MSE W&B runs."""

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

import wandb

VAL_LOSS = "val/mean/loss"
METRICS = {
    "all": VAL_LOSS,
    "temperature": "val/loss/variable/thetao_loss",
    "salinity": "val/loss/variable/so_loss",
    "zonal_velocity": "val/loss/variable/uo_loss",
    "meridional_velocity": "val/loss/variable/vo_loss",
    "ssh": "val/loss/variable/zos_loss",
}
HISTORY_KEYS = ["epoch", *METRICS.values(), "train/mean/loss", "_step"]


def validate_run_config(config: Mapping[str, Any]) -> None:
    """Reject runs whose logged losses are not comparable plain one-step MSEs."""
    train_config = config.get("config")
    if not isinstance(train_config, Mapping):
        raise ValueError("W&B run does not contain a nested training config")
    if train_config.get("loss") != "mse":
        raise ValueError("W&B run is not configured with plain MSE loss")
    if train_config.get("steps") != [1]:
        raise ValueError("W&B run is not configured for one-step training")


def select_best_row(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the finite row with the lowest all-channel validation MSE."""
    valid_rows = []
    for row in rows:
        loss = row.get(VAL_LOSS)
        if isinstance(loss, (int, float)) and math.isfinite(loss):
            valid_rows.append(dict(row))
    if not valid_rows:
        raise ValueError(f"No finite `{VAL_LOSS}` history rows found")
    return min(valid_rows, key=lambda row: row[VAL_LOSS])


def summarize_run(run: Any) -> dict[str, Any]:
    """Fetch and normalize the best validation row for one W&B run."""
    validate_run_config(run.config)
    row = select_best_row(run.scan_history(keys=HISTORY_KEYS, page_size=1000))
    return {
        "path": run.path,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "epoch": row.get("epoch"),
        "step": row.get("_step"),
        "train": row.get("train/mean/loss"),
        **{label: row.get(key) for label, key in METRICS.items()},
    }


def markdown_table(summaries: Iterable[Mapping[str, Any]]) -> str:
    """Render summaries as a compact Markdown table."""
    columns = ["epoch", *METRICS, "train"]
    lines = [
        "| Run | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for summary in summaries:
        values = []
        for column in columns:
            value = summary.get(column)
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        name = f"[{summary['name']}]({summary['url']})"
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the best epoch from one-step, plain-MSE W&B runs."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="W&B run paths in entity/project/run-id form.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="W&B API timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=args.timeout)
    summaries = [summarize_run(api.run(path)) for path in args.runs]
    if args.format == "json":
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        print(markdown_table(summaries))


if __name__ == "__main__":
    main()
