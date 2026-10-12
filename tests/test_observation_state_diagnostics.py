# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_state_diagnostics import (
    StateIntervention,
    diagnostic_data,
    diagnostic_reference,
    interior_diagnostics,
    intervene,
)
from samudra.experiments.observation_training import Samples
from samudra.experiments.quarter_observation_operator import NativeIapOperator


def test_interventions_preserve_observed_channels_and_source_state():
    source = torch.randn(1, 2, 77, 2, 2)
    original = source.clone()
    velocity = intervene(source, "zero-velocity")
    assert torch.count_nonzero(velocity[:, :, :38]) == 0
    torch.testing.assert_close(velocity[:, :, 38:], source[:, :, 38:])
    deep = intervene(source, "mean-deep-ts")
    retained = list(range(52)) + list(range(57, 71)) + [76]
    torch.testing.assert_close(deep[:, :, retained], source[:, :, retained])
    assert torch.count_nonzero(deep[:, :, 52:57]) == 0
    assert torch.count_nonzero(deep[:, :, 71:76]) == 0
    torch.testing.assert_close(source, original)
    with pytest.raises(ValueError):
        intervene(source, "unknown")


def test_unmodified_diagnostic_matches_native_forecast():
    model: Any = SimpleNamespace(
        initialize=lambda *args: torch.ones(1, 2, 77, 2, 2),
        adapt=lambda a: a[:, :, :3],
        evolution=lambda state, forcing, context, mask, steps: (
            state[:, -1] + forcing[:, 0, :1]
        ),
        call=lambda fn, *args: fn(*args),
    )
    args = (
        torch.zeros(1, 19, 2, 2, 2),
        torch.randn(1, 25, 8, 2, 2),
        torch.zeros(1, 25, 5, 2, 2),
        torch.ones(77, 2, 2),
        torch.ones(1, 19, 2, 2, 2),
    )
    expected = ObservationTransfer.forecast(model, *args)
    actual = StateIntervention(model, "none").forecast(*args)
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("native", [False, True])
def test_diagnostic_passes_manifest_geometry_to_samples(monkeypatch, native):
    calls: list[Any] = []

    def samples(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            use_observation_normalization=lambda: calls.append("scaled")
        )

    monkeypatch.setattr(
        "samudra.experiments.observation_state_diagnostics.Samples",
        samples,
    )
    args: dict[str, Any] = {"data": "observations"}
    if native:
        args.update(global_observations=True, iap_grid="iap-grid.npz")
    diagnostic_data(args, "cpu")
    assert calls == [
        (
            ("observations", "cpu", "climatology"),
            {"global_observations": native, "iap_grid": args.get("iap_grid")},
        ),
        "scaled",
    ]


@pytest.mark.parametrize("native", [False, True])
def test_interior_profiles_use_observation_geometry_and_training_support(native):
    data = Samples.__new__(Samples)
    data.dual_grid = native
    data.ts_indices = list(range(38, 52)) + list(range(57, 71))
    data.mean = torch.zeros(1, 1, 77, 1, 1)
    data.std = torch.ones(1, 1, 77, 1, 1) * 2
    lat, lon = [-67.5, -22.5, 22.5, 67.5], np.arange(22.5, 360, 45)
    mask = np.ones((28, 4, 8), dtype=bool)
    if native:
        data.interior_operator = NativeIapOperator(
            lat, lon, [-45, 45], np.arange(0, 360, 90), mask
        )
        data._interior_support = data.interior_operator.supported.clone()
        data._interior_support[0] = False
        data._interior_area = data.interior_operator.area
        shape = (2, 4)
    else:
        data.ts_mask = torch.tensor(mask)
        data.ts_mask[0] = False
        data.area = torch.tensor(np.cos(np.deg2rad(lat))[:, None], dtype=torch.float32)
        shape = (4, 8)
    # Physical scale conversion and spatial projection both affect this field.
    monthly = torch.arange(77 * 32, dtype=torch.float32).reshape(1, 77, 4, 8) / 100
    projected = data.project_interior(monthly)
    target = projected.clone() - 3
    target[:, 0] = 1e6  # Surface temperature slot must not enter interior score.
    target[:, 1, 0, 0] = torch.nan
    sample = {"interior": target}
    prediction, truth, valid, mse, count = interior_diagnostics(data, monthly, sample)
    assert prediction.shape == truth.shape == valid.shape == (28, *shape)
    np.testing.assert_allclose(prediction, projected[0])
    assert not valid[0].any() and not valid[1, 0, 0]
    assert count[0] == 0
    assert count[1] < count[2]
    np.testing.assert_allclose(mse[1:], 9, atol=1e-5)
    if not native:
        # Preserve the old model-grid physical reconstruction values exactly.
        np.testing.assert_array_equal(prediction, monthly[0, data.ts_indices] * 2)
    wrong_shape = {"interior": target[..., :1]}
    with pytest.raises(ValueError, match="shapes differ"):
        interior_diagnostics(data, monthly, wrong_shape)
    bad = monthly.clone()
    bad[:, 39, 0, 1] = torch.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        interior_diagnostics(data, bad, sample)


@pytest.mark.parametrize("external", [False, True])
def test_reference_uses_frozen_protocol_data_and_cohort(tmp_path, external):
    run = tmp_path / "run"
    run.mkdir()
    path = tmp_path / "shared.json" if external else run / "selection-reference.json"
    reference = {
        "protocol": {"selection": "rmse-only"},
        "data_manifest_sha256": "data-hash",
        "control": {"origins": ["2001-01-01"]},
        "spectral_keys": [],
    }
    path.write_text(json.dumps(reference))
    manifest = {
        "arguments": {"selection_reference": str(path)},
        "protocol": reference["protocol"],
        "data_manifest_sha256": "data-hash",
    }
    actual, actual_path = diagnostic_reference(run, manifest, ["2001-01-01"])
    assert actual == reference and actual_path == path
    with pytest.raises(ValueError, match="protocol/data/cohort differs"):
        diagnostic_reference(run, manifest, ["2002-01-01"])
