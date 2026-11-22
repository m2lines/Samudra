# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact finite-population checks for matched-channel sampling and query layouts."""

import numpy as np
import pytest
import torch

from samudra.skyline.coverage import proposal_probabilities
from samudra.skyline.model import ModelConfig, PointTransformer
from samudra.skyline.multitask import (
    ARMS,
    ComparisonSource,
    comparison_losses,
    full_grid_weights,
    learning_rate_factor,
    query_channels,
)


def population():
    area = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    wet = np.zeros((19, 2, 3), dtype=bool)
    for z in range(19):
        wet[z].flat[: max(1, 5 - z // 4)] = True
    return wet, area


@pytest.mark.parametrize("arm", ARMS)
def test_exact_full_grid_objective_and_gradient(arm):
    wet, area = population()
    # Enumerate every occurrence of the latent-depth base distribution.
    z, y, x = np.where(wet)
    channels = query_channels(arm, z)
    weights, valid = full_grid_weights(arm, z, y, x, channels, wet, area)
    prob = area[y, x] / (19 * (wet * area[None]).sum((1, 2))[z])
    np.testing.assert_allclose(prob.sum(), 1)
    channel_wet = np.concatenate([wet] * 4 + [wet[:1]])
    values = torch.arange(77 * 6, dtype=torch.float64).reshape(77, 2, 3) / 100
    values.requires_grad_(True)
    selected = values[channels, y[:, None], x[:, None]]
    sampled = (
        selected.square() * torch.tensor(weights) * torch.tensor(prob[:, None])
    ).sum()
    idx = slice(38, 57) if arm == "temperature" else slice(None)
    exact = (values[idx].square() * torch.tensor(channel_wet[idx])).mean()
    torch.testing.assert_close(sampled, exact, rtol=1e-6, atol=1e-8)
    (grad_sample,) = torch.autograd.grad(sampled, values, retain_graph=True)
    (grad_exact,) = torch.autograd.grad(exact, values)
    torch.testing.assert_close(grad_sample, grad_exact, rtol=1e-6, atol=1e-8)
    if arm == "all_fields":
        assert np.array_equal(valid[:, -1], z == 0)


def test_coverage_correction_composes_with_grid_weights():
    # Pool contains unequal weights and duplicated occurrences.
    losses = np.array([3.0, 3.0, 7.0, 18.0, 1.0])
    q = proposal_probabilities(np.array([0, 0, 0, 1, 2]), 0.5)
    np.testing.assert_allclose(np.sum(q * losses / (len(q) * q)), losses.mean())


def synthetic_source(arm):
    source = ComparisonSource.__new__(ComparisonSource)
    source.arm = arm
    source.wet, source.area = population()
    source.wet_area = (source.wet * source.area[None]).sum((1, 2))
    source.height, source.width = source.area.shape
    source.xyz = np.ones((2, 3, 3), dtype=np.float32)
    source.locations, source.location_cdfs = [], []
    for wet in source.wet:
        locations = np.flatnonzero(wet)
        weights = source.area.ravel()[locations]
        source.locations.append(locations)
        source.location_cdfs.append(np.cumsum(weights / weights.sum()))
    frame = dict(
        input=np.random.default_rng(3).normal(size=(2, 80, 2, 3)).astype(np.float32),
        target=np.random.default_rng(4).normal(size=(77, 2, 3)).astype(np.float32),
        origin=1,
    )
    return source, frame


@pytest.mark.parametrize("arm,queries", zip(ARMS, (1, 5, 77), strict=True))
def test_crop_query_loss_and_backward(arm, queries):
    source, frame = synthetic_source(arm)
    batch = source.sample(frame, np.random.default_rng(1), 2, 3, np.array([0, 18]))
    torch.manual_seed(12)
    model = PointTransformer(
        ModelConfig(
            width=32, layers=1, heads=4, objective="regression", field_queries=True
        )
    )
    output = model(batch)
    assert output.shape == (2, queries)
    loss = comparison_losses(output, batch).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )
    # Masked query tokens cannot influence the active predictions.
    invalid = ~batch["query_valid"]
    if invalid.any():
        original = output.detach()
        batch["query_current"][invalid] += 1000
        changed = model(batch).detach()
        torch.testing.assert_close(original[~invalid], changed[~invalid])


def test_matched_initialization_and_paired_crops():
    models, batches = [], []
    for arm in ARMS:
        torch.manual_seed(42)
        models.append(
            PointTransformer(
                ModelConfig(
                    width=32,
                    layers=1,
                    heads=4,
                    objective="regression",
                    field_queries=True,
                )
            )
        )
        source, frame = synthetic_source(arm)
        batches.append(source.sample(frame, np.random.default_rng(15), 3, 3))
    for model, batch in zip(models[1:], batches[1:], strict=True):
        for key, value in models[0].state_dict().items():
            assert torch.equal(value, model.state_dict()[key])
        for key in ("values", "wet", "geometry", "row", "col", "depth"):
            assert torch.equal(batch[key], batches[0][key])


def test_update_schedule():
    assert learning_rate_factor(1, 200, 25000) == 1 / 200
    assert learning_rate_factor(200, 200, 25000) == 1
    assert learning_rate_factor(25000, 200, 25000) == pytest.approx(0.1)


def test_validation_uses_identical_cases_and_weights_across_layouts():
    from samudra.skyline.comparison_eval import evaluate_comparison

    class FixedError(torch.nn.Module):
        def forward(self, batch):
            return batch["target"] + (batch["query_channel"] + 1).float() / 77

    results = []
    for arm in ARMS:
        source, frame = synthetic_source(arm)
        source.origins = {"validation": np.arange(8)}
        source.std = np.ones(80)
        source.read = lambda origin: frame
        config = dict(
            validation_dates=8,
            validation_seed=123,
            validation_points_per_depth=2,
            microbatch=4,
            crop=3,
        )
        results.append(
            evaluate_comparison(FixedError(), source, config, torch.device("cpu"), 0)
        )
    for a, b in zip(results[1]["per_channel"], results[2]["per_channel"], strict=True):
        assert a == b
    assert results[0]["per_channel"] == results[1]["per_channel"][38:57]
