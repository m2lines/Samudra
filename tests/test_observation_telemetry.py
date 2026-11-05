# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fail before training when NVML cannot identify the allocated CUDA devices."""

from uuid import UUID

import pytest
import torch

from samudra.observations.qualify_scaling import (
    model_digest,
    nvidia_smi_uuid,
    verify_telemetry_devices,
)


def test_model_digest_supports_scalar_parameters_and_buffers():
    model = torch.nn.Module()
    model.register_parameter("scale", torch.nn.Parameter(torch.tensor(1.25)))
    counter = torch.tensor(3)
    model.register_buffer("counter", counter)
    model.register_buffer("bf16", torch.tensor(0.5, dtype=torch.bfloat16))
    model.register_buffer("empty", torch.empty(0))
    original = model_digest(model)
    assert len(original) == 64 and model_digest(model) == original
    counter.add_(1)
    assert model_digest(model) != original


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
