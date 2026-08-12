# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from samudra.utils.training_summary import (
    TRAINING_SUMMARY_NAME,
    TRAINING_SUMMARY_SCHEMA_VERSION,
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
        **second,
    }
    assert list(tmp_path.iterdir()) == [path]


def test_write_training_summary_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError):
        write_training_summary(tmp_path, {"validation_loss": float("nan")})
