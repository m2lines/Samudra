# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize fixed-sample SamudraMulti identity diagnostics."""

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MSE_KEY = "identity/mean/mse"
HIGH_WAVENUMBER_PREFIX = "identity/high_wavenumber_power_ratio/channel/"
SEAM_PREFIX = "identity/patch_seam_jump_ratio/channel/"
VARIABLE_KEYS = {
    "temperature": "identity/loss/variable/thetao_loss",
    "salinity": "identity/loss/variable/so_loss",
    "zonal_velocity": "identity/loss/variable/uo_loss",
    "meridional_velocity": "identity/loss/variable/vo_loss",
    "ssh": "identity/loss/variable/zos_loss",
}


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Identity row has no finite `{key}` value")
    return float(value)


def _mean_prefix(row: Mapping[str, Any], prefix: str) -> float:
    values = [
        float(value)
        for key, value in row.items()
        if key.startswith(prefix)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    ]
    if not values:
        raise ValueError(f"Identity row has no finite metrics under `{prefix}`")
    return sum(values) / len(values)


def summarize_trajectory(
    rows: Iterable[Mapping[str, Any]], *, name: str
) -> dict[str, Any]:
    """Summarize the best and final rows of one identity trajectory."""
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("Identity trajectory is empty")
    best = min(normalized, key=lambda row: _finite_float(row, MSE_KEY))
    final = normalized[-1]
    return {
        "name": name,
        "grid": f"{int(_finite_float(final, 'identity/grid_height'))}x"
        f"{int(_finite_float(final, 'identity/grid_width'))}",
        "samples": int(_finite_float(final, "identity/actual_samples")),
        "best_epoch": int(_finite_float(best, "identity/epoch")),
        "best_mse": _finite_float(best, MSE_KEY),
        "final_mse": _finite_float(final, MSE_KEY),
        "high_wavenumber_ratio": _mean_prefix(final, HIGH_WAVENUMBER_PREFIX),
        "patch_seam_ratio": _mean_prefix(final, SEAM_PREFIX),
        **{label: _finite_float(final, key) for label, key in VARIABLE_KEYS.items()},
    }


def summarize_path(path: Path) -> dict[str, Any]:
    """Load an identity trajectory from a run directory or JSON file."""
    is_run_directory = path.is_dir()
    metrics_path = path / "identity_metrics.json" if is_run_directory else path
    with open(metrics_path) as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of metric rows in {metrics_path}")
    name = metrics_path.parent.name if is_run_directory else metrics_path.stem
    return summarize_trajectory(rows, name=name)


def markdown_table(summaries: Iterable[Mapping[str, Any]]) -> str:
    """Render identity summaries as a compact Markdown table."""
    columns = [
        "grid",
        "samples",
        "best_epoch",
        "best_mse",
        "final_mse",
        *VARIABLE_KEYS,
        "high_wavenumber_ratio",
        "patch_seam_ratio",
    ]
    lines = [
        "| Run | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for summary in summaries:
        values = []
        for column in columns:
            value = summary[column]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append(f"| {summary['name']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-sample SamudraMulti identity diagnostics."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directories or identity_metrics.json files.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [summarize_path(path) for path in args.paths]
    if args.format == "json":
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        print(markdown_table(summaries))


if __name__ == "__main__":
    main()
