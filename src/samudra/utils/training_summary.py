# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Durable, machine-readable summaries for completed training epochs."""

import datetime
import json
import math
import numbers
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from samudra.utils.atomic import atomic_path

TRAINING_SUMMARY_NAME = "training_summary.json"
TRAINING_SUMMARY_SCHEMA_VERSION = 1
SEARCH_METRICS_NAME = "search_metrics.parquet"
SEARCH_WORKER_STATUS_NAME = "search_worker_status.json"
SEARCH_WORKER_STATUS_SCHEMA_VERSION = 1
_SECRET_NAME = re.compile(
    r"(?:^|_)(?:SECRET|TOKEN|PASSWORD|CREDENTIALS?|API_KEY|ACCESS_KEY)(?:_|$)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)((?:secret|token|password|credential|api[_-]?key|access[_-]?key)\s*[=:]\s*)([^\s,;]+)"
)


def _redact_secrets(value: Any) -> Any:
    """Remove environment credentials from structured public diagnostics."""
    if isinstance(value, str):
        redacted = value
        for name, secret in os.environ.items():
            if _SECRET_NAME.search(name) and len(secret) >= 4:
                redacted = redacted.replace(secret, "[REDACTED]")
        return _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", redacted)
    if isinstance(value, dict):
        return {
            name: (
                "[REDACTED]"
                if isinstance(name, str) and _SECRET_NAME.search(name)
                else _redact_secrets(item)
            )
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_secrets(item) for item in value)
    return value


def write_search_worker_status(output_dir: Path, stage: str, **details: Any) -> Path:
    """Atomically record durable lifecycle evidence for a search worker."""
    details = {name: _redact_secrets(value) for name, value in details.items()}
    path = output_dir / SEARCH_WORKER_STATUS_NAME
    recorded_at = datetime.datetime.now(datetime.UTC).isoformat()
    history: list[dict[str, Any]] = []
    retained: dict[str, Any] = {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        loaded_history = previous.get("history", [])
        if not isinstance(loaded_history, list):
            raise ValueError(f"Invalid worker status history: {path}")
        history = loaded_history
        retained = {
            key: value
            for key, value in previous.items()
            if key not in {"schema_version", "stage", "updated_at", "history"}
        }
    event = {"stage": stage, "recorded_at": recorded_at, **details}
    history.append(event)
    payload = {
        "schema_version": SEARCH_WORKER_STATUS_SCHEMA_VERSION,
        "stage": stage,
        "updated_at": recorded_at,
        "history": history,
        **retained,
        **details,
    }
    with atomic_path(path) as temporary_path:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return path


def write_training_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Atomically publish the latest completed-epoch training summary."""
    path = output_dir / TRAINING_SUMMARY_NAME
    nonfinite_metrics: list[str] = []
    sanitized: dict[str, Any] = {}
    for name, value in summary.items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                nonfinite_metrics.append(name)
                value = None
        sanitized[name] = value
    payload = {
        "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        **sanitized,
        "nonfinite_metrics": nonfinite_metrics,
    }
    with atomic_path(path) as temporary_path:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return path


def write_search_metrics(output_dir: Path, metrics: dict[str, Any]) -> Path:
    """Atomically append one completed epoch of scalar search diagnostics."""
    row: dict[str, int | float | str | bool | None] = {
        "recorded_at": datetime.datetime.now(datetime.UTC).isoformat()
    }
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.item()
        if isinstance(value, bool) or value is None or isinstance(value, str):
            row[name] = value
        elif isinstance(value, numbers.Integral):
            row[name] = int(value)
        elif isinstance(value, numbers.Real):
            row[name] = float(value)
        # Images and other rich W&B values are intentionally omitted. Their
        # underlying scalar metrics remain in this table; figures are artifacts.

    path = output_dir / SEARCH_METRICS_NAME
    frame = pd.DataFrame([row])
    if path.is_file():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    with atomic_path(path, suffix=".parquet") as temporary_path:
        frame.to_parquet(temporary_path, index=False)
    return path
