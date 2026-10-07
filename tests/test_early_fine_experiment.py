# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from samudra.experiments import extent_models, initializer_models
from samudra.experiments.early_fine_data import WetCoarsener
from samudra.experiments.early_fine_models import FineEncoder
from samudra.experiments.early_fine_training import fine_objective
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.surface_state import geographic_features


def bounds(h, w):
    x, y = np.meshgrid(np.linspace(0, 360, w + 1), np.linspace(-90, 90, h + 1))
    return {"lat_b": y, "lon_b": x}


def test_coarsener_preserves_wet_constant_and_does_not_count_land():
    remap = WetCoarsener(bounds(16, 32), bounds(4, 8))
    wet = torch.ones(3, 16, 32, dtype=torch.bool)
    wet[:, :, :2] = False
    field = torch.full((2, 3, 16, 32), 7.0)
    field[:, :, :, :2] = float("nan")
    torch.testing.assert_close(remap(field, wet), torch.full((2, 3, 4, 8), 7.0))
    torch.testing.assert_close(
        remap.integrate(torch.ones(16, 32)).sum(),
        torch.tensor(4 * np.pi, dtype=torch.float32),
    )


@pytest.mark.parametrize("latent", [0, 10])
def test_fine_task_keeps_recurrent_state_and_trains_io_without_gold_initialization(
    monkeypatch, latent
):
    def tiny(i, o, w):
        return nn.Sequential(
            nn.Sequential(nn.Conv2d(i, 8, 1), nn.GELU()), nn.Conv2d(8, o, 1)
        )

    monkeypatch.setattr(initializer_models, "make_unet", tiny)
    monkeypatch.setattr(extent_models, "make_unet", tiny)
    names = [f"{v}_{i}" for v in ["uo", "vo", "thetao", "so"] for i in range(19)] + [
        "zos"
    ]

    def model(kind):
        torch.manual_seed(1729)
        return ObservationTransfer(
            names, "instance", kind, "unet", "observed-only", "input-adapters", latent
        )

    net = model("extent-fine")
    coarse = model("extent-unet")
    for name, p in coarse.state_dict().items():
        torch.testing.assert_close(p, net.state_dict()[name], rtol=0, atol=0)
    mask = torch.ones(77, 4, 8, dtype=torch.bool)
    fine_mask = torch.ones(77, 16, 32, dtype=torch.bool)
    lat = torch.linspace(-70, 70, 4)
    lon = torch.arange(8).float() * 45
    geo = geographic_features(lat, lon)
    fine_geo = geographic_features(
        torch.linspace(-85, 85, 16), torch.arange(32).float() * 11.25
    )
    source = SimpleNamespace(
        surface_ids=[38, 76],
        coarsen=WetCoarsener(bounds(16, 32), bounds(4, 8)),
        mask=fine_mask,
        weights=fine_mask.float(),
        names=names,
        lat=torch.linspace(-85, 85, 16),
    )
    global_data = SimpleNamespace(mask=mask, weights=mask.float(), geo=geo)
    sample = dict(
        surface=torch.randn(1, 19, 2, 16, 32),
        past=torch.randn(1, 19, 3, 16, 32),
        forcing=torch.randn(1, 6, 3, 16, 32),
        truth=torch.randn(1, 2, 77, 16, 32),
        labels=torch.randn(1, 6, 77, 16, 32),
        context=torch.cat((fine_geo, torch.zeros(2, 16, 32)), 0)[None],
    )
    net.activation_checkpointing = True
    states = []
    net.evolution.register_forward_pre_hook(
        lambda _, a: states.append(a[0].detach().clone())
    )
    outputs = []

    def hook(_, a, out):
        out.retain_grad()
        outputs.append(out)

    handle = net.evolution.register_forward_hook(hook)
    loss = fine_objective(
        net,
        source,
        sample,
        global_data,
        torch.ones(1, 19, 2, 4, 8, dtype=torch.bool),
        17,
        0.1,
        0.1,
    )
    initial = states[0].clone()
    assert states[0].shape == (1, 2, 77 + latent, 4, 8)
    torch.testing.assert_close(states[1][:, -1], outputs[0].detach())
    handle.remove()
    loss.backward()
    assert all(
        any(p.grad is not None and p.grad.count_nonzero() for p in m.parameters())
        for m in [
            net.initializer.fine_encoder,
            net.evolution.fine_decoder,
            net.initializer.net,
            net.evolution.net,
        ]
    )
    if latent:
        assert outputs[0].grad[:, 77:].count_nonzero()
        assert net.initializer.fine_encoder.net[-1].weight.grad[77:87].count_nonzero()
    # Changing all gold state targets changes loss, never the initialized input state.
    net.eval()
    states.clear()
    sample["truth"] = sample["truth"] + 100
    sample["labels"] = sample["labels"] - 100
    with torch.no_grad():
        fine_objective(
            net,
            source,
            sample,
            global_data,
            torch.ones(1, 19, 2, 4, 8, dtype=torch.bool),
            17,
            0.1,
            0.1,
        )
    torch.testing.assert_close(states[0], initial, rtol=0, atol=0)


def test_encoder_reduces_spatial_dimensions_exactly_fourfold():
    encoder = FineEncoder(87)
    value = encoder(
        torch.zeros(1, 19, 2, 24, 40),
        torch.zeros(1, 19, 3, 24, 40),
        torch.ones(1, 19, 2, 24, 40, dtype=torch.bool),
        torch.zeros(1, 5, 24, 40),
    )
    assert value.shape == (1, 2, 87, 6, 10)
