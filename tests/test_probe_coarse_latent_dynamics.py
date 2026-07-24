# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from scripts.probe_coarse_latent import make_resolution
from scripts.probe_coarse_latent_dynamics import (
    DynamicsConfig,
    DynamicsProbe,
    counterfactual_pair,
    patch_mean,
)


def config(encoder: str = "resolved") -> DynamicsConfig:
    return DynamicsConfig(
        encoder=encoder,  # type: ignore[arg-type]
        decoder="anchored",
        coarse_height=4,
        coarse_width=4,
        patch_height=3,
        patch_width=5,
        channels=2,
        latent_channels=16,
        batch_size=2,
        eval_samples=2,
        steps=1,
        learning_rate=1e-3,
        reconstruction_weight=1.0,
        anomaly_amplitude=1.0,
        anomaly_modes=4,
        moment_count=4,
        seed=0,
        device="cpu",
    )


def test_dynamics_probe_shapes_and_gradients() -> None:
    settings = config()
    model = DynamicsProbe(settings)
    resolution = make_resolution(12, 20)
    source = torch.randn(2, 2, 12, 20)
    reconstruction, forecast, latent = model(source, resolution)

    assert reconstruction.shape == source.shape
    assert forecast.shape == source.shape
    assert latent.shape == (2, 16, 4, 4)
    (reconstruction.square().mean() + forecast.square().mean()).backward()
    assert model.processor.scale.grad is not None


def test_counterfactual_states_have_identical_patch_means() -> None:
    settings = config()
    state_a, state_b = counterfactual_pair(settings, samples=3, seed=4)
    mean_a = patch_mean(state_a, settings.coarse_height, settings.coarse_width)
    mean_b = patch_mean(state_b, settings.coarse_height, settings.coarse_width)

    torch.testing.assert_close(mean_a, mean_b, atol=1e-6, rtol=1e-6)
