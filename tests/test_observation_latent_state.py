# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from samudra.experiments import initializer_models, surface_state
from samudra.experiments.observation_joint import om4_objective
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_training import Samples


def tiny_model(monkeypatch, latent=10):
    def net(i, o, w):
        return nn.Sequential(nn.Conv2d(i, 8, 1), nn.GELU(), nn.Conv2d(8, o, 1))

    monkeypatch.setattr(initializer_models, "make_unet", net)
    monkeypatch.setattr(surface_state, "make_unet", net)
    names = [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)] + [
        "zos"
    ]
    torch.manual_seed(2)
    model = ObservationTransfer(
        names, "instance", "d", "unet", "observed-only", "input-adapters", latent
    )
    model.activation_checkpointing = False
    return model, names


def test_latent_memory_is_initialized_carried_and_trained_by_future_physical_loss(
    monkeypatch,
):
    model, _ = tiny_model(monkeypatch)
    mask = torch.ones(77, 4, 4)
    mask[:, 0, 0] = 0
    surface = torch.randn(1, 22, 2, 4, 4) * mask[:2]
    atmosphere = torch.randn(1, 22, 8, 4, 4)
    contexts = torch.randn(1, 22, 5, 4, 4)
    states, outputs = [], []
    model.evolution.register_forward_pre_hook(
        lambda module, args: states.append(args[0])
    )

    def keep_output(module, args, output):
        output.retain_grad()
        outputs.append(output)

    model.evolution.register_forward_hook(keep_output)
    prediction, initial = model.forecast(
        surface, atmosphere, contexts, mask, torch.ones_like(surface)
    )
    assert initial.shape == (1, 2, 87, 4, 4)
    assert prediction.shape == (1, 3, 87, 4, 4)
    assert initial[:, :, 77:].count_nonzero() > 0
    assert initial[:, :, :, 0, 0].count_nonzero() == 0
    assert prediction[:, :, :, 0, 0].count_nonzero() == 0
    torch.testing.assert_close(states[0], initial)
    torch.testing.assert_close(states[1][:, -1], outputs[0])
    torch.testing.assert_close(states[2][:, -1], outputs[1])
    prediction[:, -1, :77].square().mean().backward()
    for module, rows in [
        (model.initializer, list(range(77, 87)) + list(range(164, 174))),
        (model.evolution, list(range(77, 87))),
    ]:
        grad = module.net[-1].weight.grad
        assert grad is not None and grad[rows].count_nonzero() > 0
    assert (
        outputs[-1].grad is not None and outputs[-1].grad[:, 77:].count_nonzero() == 0
    )
    assert outputs[0].grad is not None and outputs[0].grad[:, 77:].count_nonzero() > 0
    data = Samples.__new__(Samples)
    data.std = torch.ones(1, 1, 77, 1, 1)
    data.mean = torch.zeros_like(data.std)
    torch.testing.assert_close(data.physical(prediction), prediction[:, :, :77])


def test_om4_joint_loss_accepts_unlabelled_memory_and_backpropagates(monkeypatch):
    model, names = tiny_model(monkeypatch)
    mask = torch.ones(77, 4, 4)
    surface = torch.randn(1, 19, 2, 4, 4)
    past = torch.randn(1, 19, 3, 4, 4)
    context = torch.randn(1, 5, 4, 4)
    truth = torch.randn(1, 2, 77, 4, 4)
    forcing = torch.randn(1, 6, 3, 4, 4)
    labels = torch.randn(1, 6, 77, 4, 4)
    data = SimpleNamespace(
        mask=mask,
        names=names,
        weights=mask,
        trainset=None,
        model_sample=lambda *a: (surface, past, context, truth, forcing, labels),
    )
    loss = om4_objective(model, data, [0], 0.1)
    assert torch.isfinite(loss)
    loss.backward()
    grad = model.evolution.net[-1].weight.grad
    assert grad is not None and grad[77:].count_nonzero() > 0


def test_zero_latent_default_preserves_checkpoint_shapes(monkeypatch):
    implicit, names = tiny_model(monkeypatch, 0)
    explicit = ObservationTransfer.from_arguments(
        names,
        dict(
            normalization="instance",
            initializer_architecture="unet",
            surface_policy="observed-only",
            task_conditioning="input-adapters",
        ),
    )
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    assert explicit.latent_channels == 0
    assert explicit.initializer.channels == 77
