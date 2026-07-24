# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from scripts.probe_coarse_latent import (
    FOURIER_MODES,
    CoarseLatentProbe,
    ProbeConfig,
    analytic_fields,
    coarse_resolution,
    make_resolution,
)


def config(encoder: str, decoder: str) -> ProbeConfig:
    return ProbeConfig(
        encoder=encoder,  # type: ignore[arg-type]
        decoder=decoder,  # type: ignore[arg-type]
        coarse_height=4,
        coarse_width=4,
        input_channels=2,
        latent_channels=16,
        batch_size=2,
        eval_samples=2,
        steps=1,
        learning_rate=1e-3,
        attention_heads=2,
        attention_dim_head=4,
        moment_count=4,
        neighborhood_radius=1,
        position_bias_strength=8.0,
        spectral_decay=1.0,
        seed=0,
        device="cpu",
    )


@pytest.mark.parametrize("encoder", ["mean", "attention", "resolved", "moments"])
@pytest.mark.parametrize("decoder", ["bilinear", "coordinate", "anchored", "hybrid"])
def test_coarse_latent_probe_shapes_and_gradients(encoder: str, decoder: str) -> None:
    probe = CoarseLatentProbe(config(encoder, decoder))
    source_resolution = make_resolution(12, 20)
    output_resolution = make_resolution(24, 40)
    coefficients = torch.randn(2, 2, len(FOURIER_MODES))
    source = analytic_fields(coefficients, source_resolution)
    output, latent = probe(source, source_resolution, output_resolution)

    assert latent.shape == (2, 16, 4, 4)
    assert output.shape == (2, 2, 24, 40)
    output.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in probe.parameters()
        if parameter.requires_grad
    )


def test_coarse_resolution_matches_patch_centers_across_input_scales() -> None:
    coarse_from_3x5 = coarse_resolution(make_resolution(12, 20), 4, 4)
    coarse_from_6x10 = coarse_resolution(make_resolution(24, 40), 4, 4)

    torch.testing.assert_close(coarse_from_3x5[0], coarse_from_6x10[0])
    torch.testing.assert_close(coarse_from_3x5[1], coarse_from_6x10[1])
