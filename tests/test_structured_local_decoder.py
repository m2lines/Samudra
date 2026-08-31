# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.models.modules import StructuredLocalDecoder


def _resolution(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(-90 + 90 / height, 90 - 90 / height, height),
        (torch.arange(width) + 0.5) * 360 / width,
    )


def test_structured_local_decoder_starts_as_smooth_base() -> None:
    decoder = StructuredLocalDecoder(
        in_channels=8,
        out_channels=3,
        hidden_dim=16,
        heads=2,
        dim_head=4,
        neighborhood_radius=1,
        query_chunk_size=17,
    )
    latent = torch.randn(2, 8, 4, 4, requires_grad=True)
    output = decoder(
        latent,
        _resolution(12, 20),
        source_resolution=_resolution(4, 4),
    )
    expected = decoder.base_projection(
        torch.nn.functional.interpolate(
            torch.nn.functional.pad(latent, (1, 1, 0, 0), mode="circular"),
            size=(12, 30),
            mode="bilinear",
            align_corners=False,
        )[..., 5:25]
    )

    assert output.shape == (2, 3, 12, 20)
    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert decoder.output_projection.weight.grad is not None
