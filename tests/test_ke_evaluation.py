# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Check evaluation output-slot order, fixed masking, and scalar/map consistency."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr


def test_matched_scores_preserve_both_output_slots_and_area_mask(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "ke_evaluation",
        Path(__file__).parents[1] / "experiments/current-ke-aux/evaluate.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wet = torch.tensor([[[1, 1], [0, 0]], [[1, 0], [1, 1]]]).bool()
    area = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    spatial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class Dataset:
        initial_prognostic = torch.zeros(1)

        def to(self, device):
            return self

        def __len__(self):
            return 36

    class Model:
        def inference(self, dataset, prognostic, *, steps_completed, num_steps, epoch):
            dates = torch.arange(
                2 * steps_completed + 1, 2 * (steps_completed + num_steps) + 1
            )
            errors = dates[:, None, None, None] * spatial[None, None] * wet[None]
            errors = errors.reshape(num_steps, 4, 2, 2)
            return SimpleNamespace(
                prediction=errors,
                target=torch.zeros_like(errors),
                final_prognostic=torch.zeros(1),
                time=xr.DataArray(dates.numpy()),
            )

    evaluator = SimpleNamespace(
        device="cpu",
        N_prog=2,
        source=SimpleNamespace(masks=SimpleNamespace(prognostic=wet)),
        area_weights=area,
        model=Model(),
    )
    errors, times = module.score(evaluator, Dataset(), tmp_path, continuous=False)
    expected = (
        np.asarray(module.LEADS)[:, None, None, None]
        / 5
        * (spatial * wet).numpy()[None]
    )
    np.testing.assert_allclose(errors, expected)
    result = np.load(tmp_path / "scores.npz")
    weights = (area * wet).numpy()
    weights /= weights.sum((-2, -1), keepdims=True)
    expected_mse = (
        np.arange(1, 73)[:, None] ** 2
        * ((spatial.numpy() ** 2) * weights).sum((-2, -1))[None]
    )
    np.testing.assert_allclose(result["mse"], expected_mse)
    np.testing.assert_allclose(
        result["mean_squared_error"],
        np.mean(np.arange(1, 73) ** 2) * (spatial * wet).numpy() ** 2,
    )
    assert len(times) == 72
