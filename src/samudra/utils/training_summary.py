# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Durable, machine-readable summaries for completed training epochs."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

TRAINING_SUMMARY_NAME = "training_summary.json"
TRAINING_SUMMARY_SCHEMA_VERSION = 1


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
