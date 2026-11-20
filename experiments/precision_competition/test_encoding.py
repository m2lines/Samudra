# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Encoding contracts and refinement geometry."""

import pytest
import torch

from .benchmark import Config
from .encoding import (
    EXTENSION_KINDS,
    KINDS,
    SUPPORT,
    EncodedModel,
    Source,
    advance,
    geometry,
    labels,
    quantize,
    twohot,
)


def test_all_words_roundtrip():
    words = torch.arange(65536)
    values = words.double() / 4096 - 8
    decoded, encoded = quantize(values)
    assert torch.equal(encoded, words)
    assert torch.equal(decoded, values)


def test_twohot_reconstructs_scalar():
    centers = torch.tensor([[-4.0, -1.0, 0.0, 2.0, 8.0]], dtype=torch.float64).expand(
        100, -1
    )
    values = torch.linspace(-4, 8, 100, dtype=torch.float64)
    weights = twohot(values, centers)
    torch.testing.assert_close((weights * centers).sum(-1), values)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(values))


def test_overlap_covers_parent_and_can_correct_neighbor_error():
    c, r = torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    centers, edges = geometry(c, r, "ar_overlap")
    assert (centers.diff().max() / 2) < r.item() / 8
    target = torch.tensor([0.0], dtype=c.dtype)
    index = labels(target, centers, edges)
    nc, nr = advance(c, r, centers, edges, index)
    assert abs(nc.item()) < nr.item()
    second, edges = geometry(nc, nr, "ar_overlap")
    final, _ = advance(nc, nr, second, edges, labels(target, second, edges))
    assert final.abs().item() < nc.abs().item()


@pytest.mark.parametrize("kind", KINDS + EXTENSION_KINDS)
def test_prediction_does_not_read_target_and_gradients_flow(kind):
    torch.set_num_threads(2)
    source = Source(Config(width=16, depth=1, heads=2, tokens=64), "cpu")
    model = EncodedModel(source.c, source.positions, kind).float()
    batch = source.draw(8, source.generator(123))
    pred, _, loss, detail = model(batch)
    altered = {**batch, "main": batch["main"] + 0.31}
    other = model(altered)[0]
    torch.testing.assert_close(pred, other, rtol=0, atol=0)
    loss.backward()
    params = dict(model.named_parameters())
    for name in ("decoder.0.weight", "backbone.query.0.weight"):
        grad = params[name].grad
        assert grad is not None and grad.norm() > 0
    assert torch.isfinite(pred).all()
    assert detail["stages"].shape[0] == 8


def test_nested_oracle_has_four_levels_and_uniform_resolution():
    source = Source(Config(width=16, depth=1), "cpu")
    model = EncodedModel(source.c, source.positions, "ar_uniform").double()
    values = torch.linspace(-SUPPORT, SUPPORT - 1 / 4096, 1000, dtype=torch.float64)
    prediction, _, detail = model.decode(
        torch.zeros(1000, 16, dtype=torch.float64), values, oracle=True
    )
    assert detail["stages"].shape == (1000, 4)
    assert (prediction - values).abs().max() <= 1 / 8192 + 1e-12


def test_training_resume_is_identical(tmp_path):
    from copy import deepcopy

    from .encoding import train

    torch.set_num_threads(2)
    source = Source(
        Config(
            width=16,
            depth=1,
            heads=2,
            tokens=64,
            batch=8,
            validation_size=8,
            test_size=16,
        ),
        "cpu",
    )
    torch.manual_seed(11)
    model = EncodedModel(source.c, source.positions, "ar_twohot_soft").float()
    initial = deepcopy(model.state_dict())
    train(model, source, tmp_path / "full", 2, 1e-4, 1.0, "joint", 1.0, 0)
    expected = deepcopy(model.state_dict())
    model.load_state_dict(initial)
    train(model, source, tmp_path / "resume", 1, 1e-4, 1.0, "joint", 1.0, 0)
    train(model, source, tmp_path / "resume", 2, 1e-4, 1.0, "joint", 1.0, 0)
    for key, value in expected.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("base", ["ar_overlap", "ar_twohot_hard", "ar_twohot_soft"])
def test_scaled_ce_changes_only_loss(base):
    import math

    source = Source(Config(width=16, depth=1), "cpu")
    plain = EncodedModel(source.c, source.positions, base).double()
    scaled = EncodedModel(source.c, source.positions, base + "_scaled").double()
    with torch.no_grad():
        for parameter in plain.decoder.parameters():
            parameter.zero_()
    scaled.load_state_dict(plain.state_dict())
    h = torch.zeros(4, 16, dtype=torch.float64)
    target = torch.tensor([-0.5, 0.0, 0.1, 0.8], dtype=torch.float64)
    p1, l1, _ = plain.decode(h, target)
    p2, l2, _ = scaled.decode(h, target)
    torch.testing.assert_close(p1, p2, rtol=0, atol=0)
    assert l1.item() == pytest.approx(math.log(16))
    assert l2.item() == pytest.approx(
        math.log(16) * sum(64.0**-k for k in range(4)) / 4
    )


def test_continuous_loss_scaling_and_radius():
    source = Source(Config(width=16, depth=1), "cpu")
    physical = EncodedModel(source.c, source.positions, "cont_shrink_mse").double()
    normalized = EncodedModel(
        source.c, source.positions, "cont_shrink_normalized"
    ).double()
    fixed = EncodedModel(source.c, source.positions, "cont_fixed_mse").double()
    with torch.no_grad():
        for parameter in physical.decoder.parameters():
            parameter.zero_()
    normalized.load_state_dict(physical.state_dict())
    fixed.load_state_dict(physical.state_dict())
    h = torch.zeros(4, 16, dtype=torch.float64)
    target = torch.ones(4, dtype=torch.float64) * 0.5
    p1, l1, d1 = physical.decode(h, target)
    p2, l2, _ = normalized.decode(h, target)
    _, l3, d3 = fixed.decode(h, target)
    torch.testing.assert_close(p1, p2, rtol=0, atol=0)
    torch.testing.assert_close(
        d1["radius"][0], torch.tensor([8.0, 1.0, 0.125, 0.015625], dtype=torch.float64)
    )
    assert torch.all(d3["radius"] == 8.0)
    assert l1.item() == pytest.approx(0.25)
    assert l2.item() == pytest.approx(
        0.25 * sum(1 / r**2 for r in [8.0, 1.0, 0.125, 0.015625]) / 4
    )
    assert l3.item() == pytest.approx(0.25)
