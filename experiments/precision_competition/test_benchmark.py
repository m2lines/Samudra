# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Scientific invariants and restart checks for the standalone experiment."""

import json
from dataclasses import replace

import pytest
import torch

from .benchmark import (
    Config,
    Fields,
    Model,
    bin_labels,
    centers,
    losses,
    train,
    update_probe,
)
from .suite import make_suite


def test_pair_has_identical_inputs_and_main_target():
    c = Config(main="forecast", auxiliary="evolution", tokens=64)
    a, b = Fields(c, "cpu"), Fields(replace(c, relation="dependent"), "cpu")
    x, y = a.draw(12, a.generator(5)), b.draw(12, b.generator(5))
    for key in ("values", "index", "main", "current", "main_field"):
        torch.testing.assert_close(x[key], y[key], atol=0, rtol=0)
    assert not torch.equal(x["auxiliary"], y["auxiliary"])


def test_high_frequency_interaction_survives_only_product_first():
    after = Fields(Config(tokens=64, filter="after"), "cpu")
    before = Fields(Config(tokens=64, filter="before"), "cpu")
    x = 2 * torch.pi * torch.arange(64, dtype=torch.float64) / 64
    t, u = (12 * x).cos()[None], (13 * x).cos()[None]
    # cos(12x)cos(13x)=.5cos(x)+.5cos(25x), so -P d/dx /4=.125sin(x).
    torch.testing.assert_close(after.flux(t, u), x.sin()[None] / 8, atol=2e-14, rtol=0)
    assert before.flux(t, u).abs().max() < 1e-14


def test_source_scale_and_bounded_targets():
    source = Fields(Config(tokens=64, main="forecast", alpha=0.1), "cpu")
    batch = source.draw(2048, source.generator(5))
    observed = batch["values"][:, :, 0].square().mean().sqrt().item()
    assert abs(observed / source.scale - 1) < 0.03
    assert batch["main_field"].abs().max() < 1.1
    assert batch["values"].abs().max() < 1
    assert (
        source.lowpass(batch["auxiliary_field"])
        .sub(batch["auxiliary_field"])
        .abs()
        .max()
        < 1e-14
    )


def test_zero_auxiliary_weight_preserves_main_update():
    torch.set_num_threads(2)
    config = Config(width=16, heads=2, depth=1, tokens=64)
    source = Fields(config, "cpu")
    models = []
    for relation in ("independent", "dependent"):
        c = replace(config, relation=relation)
        torch.manual_seed(17)
        model = Model(c, source.positions).float()
        batch = Fields(c, "cpu").draw(4, source.generator(8))
        pred, aux = model(batch)
        lm, la = losses(pred, aux, batch, c)
        (lm + 0 * la).backward()
        models.append(model)
    for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_quantization_bound_and_tolerance():
    c = Config(bins=16)
    y = torch.linspace(-c.support, c.support, 1001, dtype=torch.float64)
    quantized = centers(c, "cpu", torch.float64)[bin_labels(y, c)]
    assert (quantized - y).abs().max() <= c.support / c.bins + 1e-15
    lm, _ = losses(
        y + c.support / c.bins * 0.99,
        y,
        {"main": y, "auxiliary": y},
        replace(c, objective="tolerance"),
    )
    assert lm == 0
    with pytest.raises(ValueError, match="outside"):
        bin_labels(torch.tensor([3.0]), c)


def test_bin_count_preserves_backbone_initialization():
    c = Config(width=16, heads=2, depth=1, tokens=64, objective="ce")
    source = Fields(c, "cpu")
    torch.manual_seed(19)
    a = Model(c, source.positions)
    torch.manual_seed(19)
    b = Model(replace(c, bins=4096), source.positions)
    for name, value in a.state_dict().items():
        if not name.startswith("head."):
            torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)


def test_resume_is_identical_to_uninterrupted(tmp_path):
    c = Config(
        width=16,
        heads=2,
        depth=1,
        tokens=64,
        batch=4,
        steps=4,
        eval_every=2,
        validation_size=8,
        test_size=8,
        train_seconds=600,
    )
    a, b = tmp_path / "continuous", tmp_path / "resumed"
    train(c, a, "cpu")
    interrupted = train(c, b, "cpu", stop_after=2)
    assert interrupted["interrupted"]
    train(c, b, "cpu")
    left = torch.load(a / "latest.pt", weights_only=False)
    right = torch.load(b / "latest.pt", weights_only=False)
    for key in left["model"]:
        torch.testing.assert_close(
            left["model"][key], right["model"][key], atol=0, rtol=0
        )
    assert (
        json.loads((a / "result.json").read_text())["test"]
        == json.loads((b / "result.json").read_text())["test"]
    )


def test_manifest_has_solo_counterpart_for_every_joint_arm():
    entries = make_suite()
    ids = {entry["id"] for entry in entries}
    assert len(ids) == len(entries)
    for entry in entries:
        c = Config(**entry["config"])
        if c.auxiliary_weight and c.main_weight:
            solo = replace(
                c, auxiliary_weight=0.0, relation="independent", auxiliary_start=0
            )
            assert solo.run_id in ids


def test_wire_exact(tmp_path):
    c = Config(
        width=16,
        heads=2,
        depth=0,
        tokens=64,
        path="wire",
        steps=0,
        precision="fp64",
        batch=4,
        validation_size=8,
        test_size=8,
    )
    result = train(c, tmp_path, "cpu")
    assert result["test"]["rmse"] == 0


def test_counterfactual_update_restores_state():
    config = Config(width=16, heads=2, depth=1, tokens=64)
    source = Fields(config, "cpu")
    model = Model(config, source.positions).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    result = update_probe(
        model,
        optimizer,
        source.draw(4, source.generator(1)),
        source.draw(4, source.generator(2)),
        config,
        0.0,
    )
    assert result["auxiliary_added_mse_change"] == 0
    assert not optimizer.state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
