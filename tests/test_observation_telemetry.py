# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fail before training when NVML cannot identify the allocated CUDA devices."""

from uuid import UUID

import pytest

from samudra.observations.qualify_scaling import (
    nvidia_smi_uuid,
    verify_telemetry_devices,
)


def test_cuda_uuid_is_normalized_and_missing_telemetry_is_rejected(monkeypatch):
    value = UUID("e1b80137-1ed0-11a5-6680-0c69f8437c1a")
    expected = f"GPU-{value}"
    assert nvidia_smi_uuid(value) == expected
    assert nvidia_smi_uuid(expected) == expected

    def missing(command, **kwargs):
        assert command[-1] == f"--id={expected}"
        return "No devices were found\n"

    monkeypatch.setattr("subprocess.check_output", missing)
    with pytest.raises(RuntimeError, match="device mismatch"):
        verify_telemetry_devices([expected])
    monkeypatch.setattr(
        "subprocess.check_output", lambda *args, **kwargs: expected + "\n"
    )
    verify_telemetry_devices([expected])
