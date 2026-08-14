# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Durable, machine-readable summaries for completed training epochs."""

import datetime
import json
import numbers
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

TRAINING_SUMMARY_NAME = "training_summary.json"
TRAINING_SUMMARY_SCHEMA_VERSION = 1
SEARCH_METRICS_NAME = "search_metrics.parquet"
SEARCH_WORKER_STATUS_NAME = "search_worker_status.json"
SEARCH_WORKER_STATUS_SCHEMA_VERSION = 1


def write_search_worker_status(output_dir: Path, stage: str, **details: Any) -> Path:
    """Atomically record durable lifecycle evidence for a search worker."""
    path = output_dir / SEARCH_WORKER_STATUS_NAME
    recorded_at = datetime.datetime.now(datetime.UTC).isoformat()
    history: list[dict[str, Any]] = []
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        loaded_history = previous.get("history", [])
        if not isinstance(loaded_history, list):
            raise ValueError(f"Invalid worker status history: {path}")
        history = loaded_history
    event = {"stage": stage, "recorded_at": recorded_at, **details}
    history.append(event)
    payload = {
        "schema_version": SEARCH_WORKER_STATUS_SCHEMA_VERSION,
        "stage": stage,
        "updated_at": recorded_at,
        "history": history,
        **details,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_dir,
        prefix=f".{SEARCH_WORKER_STATUS_NAME}.",
        delete=False,
        encoding="utf-8",
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    return path


def write_training_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Atomically publish the latest completed-epoch training summary."""
    path = output_dir / TRAINING_SUMMARY_NAME
    payload = {"schema_version": TRAINING_SUMMARY_SCHEMA_VERSION, **summary}
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_dir,
        prefix=f".{TRAINING_SUMMARY_NAME}.",
        delete=False,
        encoding="utf-8",
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
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
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=f".{SEARCH_METRICS_NAME}.", suffix=".parquet"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
