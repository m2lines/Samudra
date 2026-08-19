# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from samudra.utils.training_summary import (
    TRAINING_SUMMARY_NAME,
    TRAINING_SUMMARY_SCHEMA_VERSION,
    write_search_worker_status,
    write_training_summary,
)


def test_write_training_summary_atomically_replaces_latest_epoch(tmp_path):
    first = {"epoch": 1, "validation_loss": 0.8}
    second = {"epoch": 2, "validation_loss": 0.6}

    path = write_training_summary(tmp_path, first)
    write_training_summary(tmp_path, second)

    assert path == tmp_path / TRAINING_SUMMARY_NAME
    assert json.loads(path.read_text()) == {
        "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        "nonfinite_metrics": [],
        **second,
    }
    assert list(tmp_path.iterdir()) == [path]


def test_write_training_summary_records_nonfinite_values(tmp_path):
    path = write_training_summary(tmp_path, {"validation_loss": float("nan")})

    assert json.loads(path.read_text()) == {
        "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        "validation_loss": None,
        "nonfinite_metrics": ["validation_loss"],
    }


def test_write_search_worker_status_preserves_lifecycle_history(tmp_path):
    path = write_search_worker_status(
        tmp_path, "launched", job_id="123", candidate="control"
    )
    write_search_worker_status(
        tmp_path, "optimizer_step", optimizer_steps=1, batches_seen=32
    )

    status = json.loads(path.read_text())
    assert status["stage"] == "optimizer_step"
    assert status["optimizer_steps"] == 1
    assert status["candidate"] == "control"
    assert status["job_id"] == "123"
    assert [event["stage"] for event in status["history"]] == [
        "launched",
        "optimizer_step",
    ]


def test_write_search_worker_status_redacts_environment_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "super-secret-value")

    path = write_search_worker_status(
        tmp_path,
        "failed",
        error="WANDB_API_KEY=super-secret-value",
    )

    text = path.read_text()
    assert "super-secret-value" not in text
    assert "[REDACTED]" in text


def test_failed_atomic_write_removes_temporary_file(tmp_path):
    with pytest.raises(TypeError):
        write_search_worker_status(tmp_path, "failed", unsupported={"a", "set"})

    assert list(tmp_path.iterdir()) == []
