# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import io

import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_direct import (
    DirectLatentProcessor,
    install_direct_processor,
    load_latent_checkpoint,
)
from samudra.experiments.diffusion_latent import PersistentLatentProcessor


def small_model():
    model = nn.Module()
    model.processor = PersistentLatentProcessor(
        width=8, depth=1, latent_shape=(4, 6), conditioning_shape=(8, 12)
    )
    model.processor.input_adapters = nn.ModuleDict(
        {"om4": nn.Identity(), "obs": nn.Conv2d(24, 24, 1)}
    )
    return model


def inputs():
    return (
        torch.randn(1, 2, 8, 4, 6),
        torch.randn(1, 3, 8, 12),
        torch.randn(1, 5, 8, 12),
    )


def test_zero_head_direct_is_zero_and_residual_is_persistence():
    torch.set_num_threads(1)
    model = small_model()
    nn.init.zeros_(model.processor.head.weight)
    nn.init.zeros_(model.processor.head.bias)
    latent, forcing, context = inputs()
    residual = model.processor(latent, forcing, context)
    assert torch.equal(residual[:, 0], latent[:, -1])
    assert torch.equal(residual[:, 1], latent[:, -1])
    install_direct_processor(model)
    direct = model.processor(latent, forcing, context)
    assert torch.equal(direct[:, 0], latent[:, -1])
    assert torch.equal(direct[:, 1], torch.zeros_like(latent[:, -1]))
    direct[:, 1].sum().backward()
    assert model.processor.head.weight.grad.abs().sum() > 0


def test_conversion_preserves_parameters_adapters_mode_and_rng():
    model = small_model().eval()
    parameters = list(model.parameters())
    rng = torch.get_rng_state().clone()
    install_direct_processor(model)
    assert isinstance(model.processor, DirectLatentProcessor)
    assert not model.processor.training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert set(model.processor.input_adapters) == {"om4", "obs"}
    assert model.processor.direct_prediction_version.item() == 1


@pytest.mark.parametrize("mode", [None, "residual", "direct"])
def test_checkpoint_roundtrip_preserves_predictions_and_history(mode):
    torch.set_num_threads(1)
    model = small_model().eval()
    if mode == "direct":
        install_direct_processor(model)
    saved = {"model": model.state_dict()}
    if mode is not None:
        saved["processor_prediction"] = mode
    stream = io.BytesIO()
    torch.save(saved, stream)
    stream.seek(0)
    restored = load_latent_checkpoint(
        small_model().eval(), torch.load(stream, weights_only=True)
    )
    latent, forcing, context = inputs()
    expected = model.processor(latent, forcing, context, "obs")
    actual = restored.processor(latent, forcing, context, "obs")
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, 0], latent[:, -1])
    assert isinstance(restored.processor, DirectLatentProcessor) == (mode == "direct")


@pytest.mark.parametrize(
    ("mode", "marker"),
    [
        ("unknown", None),
        ("direct", None),
        ("residual", torch.tensor(1)),
        (None, torch.tensor(1)),
        ("direct", torch.tensor(2)),
        ("direct", torch.tensor(1.0)),
        ("direct", torch.tensor([1])),
        ("direct", 1),
    ],
)
def test_inconsistent_or_unknown_checkpoint_mode_is_rejected(mode, marker):
    saved = {"model": small_model().state_dict()}
    if mode is not None:
        saved["processor_prediction"] = mode
    if marker is not None:
        saved["model"]["processor.direct_prediction_version"] = marker
    with pytest.raises(ValueError):
        load_latent_checkpoint(small_model(), saved)


def test_loading_remains_strict():
    model = install_direct_processor(small_model())
    saved = {"model": model.state_dict(), "processor_prediction": "direct"}
    del saved["model"]["processor.head.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_latent_checkpoint(small_model(), saved)
