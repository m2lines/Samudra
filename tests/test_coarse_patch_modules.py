# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.config import DecoderConfig, EncoderConfig
from samudra.models.modules import (
    ContinuousCoordinateAttentionCorrection,
    ContinuousResampleAttentionResidualDecoder,
    PatchMomentEncoder,
)


def resolution(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(-90 + 90 / height, 90 - 90 / height, height),
        (torch.arange(width) + 0.5) * 360 / width,
    )


def test_patch_moment_encoder_keeps_fixed_coarse_grid_across_resolutions() -> None:
    encoder = PatchMomentEncoder(
        in_channels=4,
        out_channels=16,
        patch_extent=(45.0, 90.0),
        moment_count=4,
        mean_channels=4,
        geometry_mode="none",
    )
    coarse = encoder(torch.randn(2, 4, 12, 20), resolution(12, 20))
    fine_input = torch.randn(2, 4, 24, 40, requires_grad=True)
    fine = encoder(fine_input, resolution(24, 40))

    assert coarse.shape == fine.shape == (2, 16, 4, 4)
    torch.testing.assert_close(
        encoder.output_resolution(resolution(12, 20))[0],
        encoder.output_resolution(resolution(24, 40))[0],
    )
    torch.testing.assert_close(
        encoder.output_resolution(resolution(12, 20))[1],
        encoder.output_resolution(resolution(24, 40))[1],
    )
    fine.square().mean().backward()
    assert fine_input.grad is not None


def test_continuous_attention_is_zero_initialized_and_query_dependent() -> None:
    correction = ContinuousCoordinateAttentionCorrection(
        in_channels=16,
        out_channels=3,
        hidden_dim=16,
        heads=2,
        dim_head=4,
        neighborhood_radius=1,
        position_bias_strength=8.0,
        query_chunk_size=17,
        zero_initialize_output=True,
    )
    latent = torch.randn(2, 16, 4, 4, requires_grad=True)
    output = correction(
        latent,
        resolution(4, 4),
        resolution(12, 20),
    )

    assert output.shape == (2, 3, 12, 20)
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    assert correction.output_projection.weight.grad is not None


def test_continuous_hybrid_starts_as_coordinate_resampling_base() -> None:
    decoder = DecoderConfig(
        continuous_resample_attention_residual=True,
        residual_hidden_dim=16,
        residual_heads=2,
        residual_dim_head=4,
        residual_query_chunk_size=17,
    ).build(
        in_channels=16,
        out_channels=3,
        patch_extent=(45.0, 90.0),
        implementation="naive",
    )
    assert isinstance(decoder, ContinuousResampleAttentionResidualDecoder)
    latent = torch.randn(2, 16, 4, 4)
    output_resolution = resolution(12, 20)
    output = decoder(
        latent,
        output_resolution,
        source_resolution=resolution(4, 4),
    )
    expected = decoder.base(
        latent,
        output_resolution,
        source_resolution=resolution(4, 4),
    )

    torch.testing.assert_close(output, expected)


def test_continuous_hybrid_ignores_a_physical_grid_mask_for_latent_attention() -> None:
    decoder = DecoderConfig(
        continuous_resample_attention_residual=True,
        residual_hidden_dim=16,
        residual_heads=2,
        residual_dim_head=4,
        residual_query_chunk_size=17,
    ).build(
        in_channels=16,
        out_channels=3,
        patch_extent=(45.0, 90.0),
        implementation="naive",
    )
    latent = torch.randn(2, 16, 4, 4)
    output = decoder(
        latent,
        resolution(12, 20),
        source_resolution=resolution(4, 4),
        valid_mask=torch.ones(3, 12, 20, dtype=torch.bool),
    )

    assert output.shape == (2, 3, 12, 20)


def test_config_builds_patch_moment_encoder() -> None:
    encoder = EncoderConfig(
        geometry_mode="none",
        patch_moment_count=4,
        patch_mean_channels=4,
    ).build(
        in_channels=4,
        out_channels=16,
        patch_extent=(45.0, 90.0),
        max_lat_size=24,
        max_lon_size=40,
        implementation="naive",
    )

    assert isinstance(encoder, PatchMomentEncoder)
