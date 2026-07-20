# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize the best epoch from one-step, plain-MSE W&B runs."""

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import wandb


@dataclass(frozen=True)
class MetricFamily:
    """One internally consistent set of validation metrics."""

    name: str
    metrics: Mapping[str, str]


def metric_family(name: str, prefix: str) -> MetricFamily:
    return MetricFamily(
        name=name,
        metrics={
            "all": f"{prefix}/mean/loss",
            "temperature": f"{prefix}/loss/variable/thetao_loss",
            "salinity": f"{prefix}/loss/variable/so_loss",
            "zonal_velocity": f"{prefix}/loss/variable/uo_loss",
            "meridional_velocity": f"{prefix}/loss/variable/vo_loss",
            "ssh": f"{prefix}/loss/variable/zos_loss",
        },
    )


# Prefer the explicitly recomputed, unweighted diagnostics. The resolution-prefixed
# form is emitted by match/multi-scale validation even when only 1 degree is active;
# the unprefixed form is emitted by the standard single-scale schedule. Legacy runs
# predate those diagnostics and fall back to their plain-MSE validation loss.
METRIC_FAMILIES = (
    metric_family("unweighted", "val/unweighted_normalized_mse"),
    metric_family(
        "unweighted_1deg",
        "val/resolution/180x360/unweighted_normalized_mse",
    ),
    metric_family("legacy_plain_mse", "val"),
)
METRICS = METRIC_FAMILIES[0].metrics


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
    row, _ = select_best_row_and_family(rows)
    return row


def select_best_row_and_family(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], MetricFamily]:
    """Select one metric family, then its best finite validation row."""
    materialized = [dict(row) for row in rows]
    for family in METRIC_FAMILIES:
        all_key = family.metrics["all"]
        valid_rows = []
        for row in materialized:
            loss = row.get(all_key)
            if isinstance(loss, (int, float)) and math.isfinite(loss):
                valid_rows.append(row)
        if valid_rows:
            return min(valid_rows, key=lambda row: row[all_key]), family
    expected = ", ".join(f"`{family.metrics['all']}`" for family in METRIC_FAMILIES)
    raise ValueError(f"No finite validation loss found in {expected}")


def scan_best_row_and_family(run: Any) -> tuple[dict[str, Any], MetricFamily]:
    """Scan W&B once per family because absent requested keys suppress rows."""
    for family in METRIC_FAMILIES:
        keys = ["epoch", *family.metrics.values(), "train/mean/loss", "_step"]
        rows = [dict(row) for row in run.scan_history(keys=keys, page_size=1000)]
        all_key = family.metrics["all"]
        valid_rows = [
            row
            for row in rows
            if isinstance(row.get(all_key), (int, float))
            and math.isfinite(row[all_key])
        ]
        if valid_rows:
            return min(valid_rows, key=lambda row: row[all_key]), family
    expected = ", ".join(f"`{family.metrics['all']}`" for family in METRIC_FAMILIES)
    raise ValueError(f"No finite validation loss found in {expected}")


def summarize_run(run: Any) -> dict[str, Any]:
    """Fetch and normalize the best validation row for one W&B run."""
    validate_run_config(run.config)
    row, family = scan_best_row_and_family(run)
    return {
        "path": run.path,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "metric_source": family.name,
        "epoch": row.get("epoch"),
        "step": row.get("_step"),
        "train": row.get("train/mean/loss"),
        **{label: row.get(key) for label, key in family.metrics.items()},
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
