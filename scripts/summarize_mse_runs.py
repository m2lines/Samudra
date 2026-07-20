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


def _same_selected_epoch(
    candidate: Mapping[str, Any], selected: Mapping[str, Any]
) -> bool:
    """Return whether a history fragment belongs to the selected epoch log."""
    selected_step = selected.get("_step")
    candidate_step = candidate.get("_step")
    if selected_step is not None and candidate_step == selected_step:
        return True
    selected_epoch = selected.get("epoch")
    return selected_epoch is not None and candidate.get("epoch") == selected_epoch


def _merge_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge W&B history fragments, retaining non-null values."""
    merged: dict[str, Any] = {}
    for row in rows:
        merged.update({key: value for key, value in row.items() if value is not None})
    return merged


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _media_reference(value: Any) -> dict[str, Any] | None:
    """Return a JSON-safe W&B media reference when one is present."""
    if not isinstance(value, Mapping):
        return None
    media_type = value.get("_type")
    if not isinstance(media_type, str) or "file" not in media_type:
        return None
    return {str(key): item for key, item in value.items() if _is_scalar(item)}


def extract_diagnostics(
    rows: Iterable[Mapping[str, Any]],
    selected: Mapping[str, Any],
    family: MetricFamily,
) -> dict[str, Any]:
    """Extract comparable terminal diagnostics around a selected validation epoch.

    W&B may split metrics logged at one explicit step across history fragments.
    This function merges those fragments, then adds the nearest spatial-image
    epoch at or before the selected checkpoint. Only scalar values are returned;
    maps and spectra are returned as JSON-safe W&B media references.
    """
    materialized = [dict(row) for row in rows]
    selected_epoch_rows = [
        row for row in materialized if _same_selected_epoch(row, selected)
    ]
    selected_values = _merge_rows([selected, *selected_epoch_rows])

    metric_prefix = family.metrics["all"].removesuffix("/mean/loss")
    if "unweighted_normalized_mse" in metric_prefix:
        persistence_prefix = metric_prefix.replace(
            "unweighted_normalized_mse", "persistence_normalized_mse"
        )
    else:
        persistence_prefix = "val/persistence_normalized_mse"
    selected_prefixes = (
        f"{metric_prefix}/",
        f"{persistence_prefix}/",
        "progress/",
        "train/throughput/",
        "train/resources/",
    )
    diagnostics = {
        key: value
        for key, value in selected_values.items()
        if _is_scalar(value)
        and (key.startswith(selected_prefixes) or key.startswith("epoch_"))
    }

    forecast = selected_values.get(f"{metric_prefix}/mean/loss")
    persistence = selected_values.get(f"{persistence_prefix}/mean/loss")
    if (
        isinstance(forecast, (int, float))
        and isinstance(persistence, (int, float))
        and math.isfinite(forecast)
        and math.isfinite(persistence)
        and persistence > 0
    ):
        ratio = forecast / persistence
        diagnostics["derived/forecast_to_persistence_mse_ratio"] = ratio
        diagnostics["derived/persistence_mse_reduction_fraction"] = 1.0 - ratio

    selected_epoch = selected.get("epoch")
    spatial_rows = []
    for row in materialized:
        epoch = row.get("epoch")
        if not isinstance(epoch, (int, float)):
            continue
        if isinstance(selected_epoch, (int, float)) and epoch > selected_epoch:
            continue
        if any("/spatial/" in key for key in row):
            spatial_rows.append(row)
    if spatial_rows:
        spatial_epoch = max(row["epoch"] for row in spatial_rows)
        spatial_values = _merge_rows(
            row for row in spatial_rows if row["epoch"] == spatial_epoch
        )
        diagnostics["spatial/epoch"] = spatial_epoch
        diagnostics.update(
            {
                key: value
                for key, value in spatial_values.items()
                if "/spatial/" in key and _is_scalar(value)
            }
        )
        media_prefixes = ("/mean_map/", "/snapshot/", "/spatial/")
        for key, value in spatial_values.items():
            if not any(prefix in key for prefix in media_prefixes):
                continue
            media = _media_reference(value)
            if media is not None:
                diagnostics[f"media/{key}"] = media

    return dict(sorted(diagnostics.items()))


def summarize_run(run: Any, *, include_diagnostics: bool = False) -> dict[str, Any]:
    """Fetch and normalize the best validation row for one W&B run."""
    validate_run_config(run.config)
    row, family = scan_best_row_and_family(run)
    summary = {
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
    if include_diagnostics:
        history = [dict(item) for item in run.scan_history(page_size=1000)]
        summary["diagnostics"] = extract_diagnostics(history, row, family)
    return summary


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
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Scan full terminal history once for persistence, depth, resource, and spatial diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=args.timeout)
    summaries = [
        summarize_run(api.run(path), include_diagnostics=args.include_diagnostics)
        for path in args.runs
    ]
    if args.format == "json":
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        print(markdown_table(summaries))


if __name__ == "__main__":
    main()
