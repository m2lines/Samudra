# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import copy

import torch

from .benchmark import Config, Fields, Model
from .noise import draw, optimizer, predictions, step, target
from .noise_suite import make


def setup():
    torch.set_num_threads(2)
    c = Config(width=16, depth=1, heads=2, tokens=64)
    source = Fields(c, "cpu")
    torch.manual_seed(13)
    return source, Model(c, source.positions).float()


def test_constant_offset_does_not_change_update():
    source, a = setup()
    b = copy.deepcopy(a)
    for model, offset in ((a, 0.0), (b, 1.0)):
        step(
            model,
            optimizer(model, 3e-4),
            source,
            64,
            source.generator(1),
            source.generator(2),
            0.0,
            "aux",
            offset,
        )
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_noise_is_hidden_and_data_stream_does_not_depend_on_sigma():
    source, model = setup()
    data1, data2 = source.generator(1), source.generator(1)
    for sigma, generator in ((0.0, data1), (1.0, data2)):
        clone = copy.deepcopy(model)
        step(
            clone,
            optimizer(clone, 1e-5),
            source,
            64,
            generator,
            source.generator(2),
            sigma,
            "aux",
        )
    torch.testing.assert_close(data1.get_state(), data2.get_state(), atol=0, rtol=0)
    batch = source.draw(64, data1)
    assert target(batch).shape == (64,)


def test_aux_head_change_does_not_change_copy():
    source, model = setup()
    batch = draw(source, 64, 1)
    before = predictions(model, batch)[0]
    with torch.no_grad():
        model.aux_head.weight.add_(10.0)
    torch.testing.assert_close(before, predictions(model, batch)[0], atol=0, rtol=0)


def test_sgd_gradient_accumulation_averages_microbatches():
    source, a = setup()
    b = copy.deepcopy(a)
    step(
        a,
        optimizer(a, 1e-5, "sgd"),
        source,
        128,
        source.generator(1),
        source.generator(2),
        0.3,
        "aux",
    )
    opt = optimizer(b, 1e-5, "sgd")
    opt.zero_grad()
    data_rng, noise_rng = source.generator(1), source.generator(2)
    for _ in range(2):
        batch = source.draw(64, data_rng)
        epsilon = torch.randn(64, generator=noise_rng)
        _, aux = b(batch)
        (
            ((aux - target(batch).float()) / source.scale - 0.3 * epsilon)
            .square()
            .mean()
            / 2
        ).backward()
    opt.step()
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_manifest_counts(tmp_path):
    entries = make(tmp_path)
    assert [len(entries[k]) for k in ("prepare", "perturb", "joint")] == [6, 36, 18]


def test_trajectory_resume_preserves_raw_ema_and_rng(tmp_path):
    from dataclasses import asdict

    from .noise import trajectory

    source, model = setup()
    saved = {
        "base_config": asdict(source.c),
        "model": model.state_dict(),
        "learned_head": model.aux_head.state_dict(),
        "seed": 11,
    }
    args = (saved, source)
    a = trajectory(*args, tmp_path / "a", 1e-5, 64, "learned", 0.3, "aux", 8, 0)
    trajectory(
        *args, tmp_path / "b", 1e-5, 64, "learned", 0.3, "aux", 8, 0, stop_after=4
    )
    b = trajectory(*args, tmp_path / "b", 1e-5, 64, "learned", 0.3, "aux", 8, 0)
    assert a["test"] == b["test"] and a["ema_test"] == b["ema_test"]
    assert a["history"] == b["history"]
