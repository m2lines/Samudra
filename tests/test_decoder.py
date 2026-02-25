import torch
from test_encoder import make_perceiver, make_resolution  # type: ignore

from ocean_emulators.models.modules import PerceiverDecoder, PerceiverEncoder


def test_roundtrip():
    x = torch.randn(3, 10, 4, 8)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(180, 180),
        perceiver=make_perceiver(10, 4),
    )

    patches = patch_embed(x, make_resolution(x))

    decode = PerceiverDecoder(
        in_channels=4,
        out_channels=10,
        patch_extent=(180, 180),
        perceiver=make_perceiver(4, 10, num_latents=6, input_axis=1),
    )

    y_hat = decode(patches, make_resolution(x))

    assert y_hat.shape[-2:] == patches.shape[-2:], (
        "The decoder preservers the input grid."
    )
    assert y_hat.shape[1] == 10, (
        "The decoder outputs the desired number of output channels."
    )


def test_decode():
    x = torch.randn(3, 12, 4, 8)

    # Resolution represents the original (pre-encoder) physical grid, which is
    # larger than the decoder's latent input — mirroring the real pipeline where
    # the encoder reduces spatial dimensions.
    resolution = (
        torch.linspace(-90, 90, steps=8),
        torch.linspace(0, 360, steps=16),
    )

    decode = PerceiverDecoder(
        in_channels=12,
        out_channels=24,
        patch_extent=(90, 90),
        perceiver=make_perceiver(12, 24, num_latents=6, input_axis=1),
    )

    y_hat = decode(x, resolution)

    assert y_hat.shape[-2:] == x.shape[-2:], "The decoder preservers the input grid."
    assert y_hat.shape[1] == 24, (
        "The decoder outputs the desired number of output channels."
    )
