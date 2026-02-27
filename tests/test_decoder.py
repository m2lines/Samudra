import torch
from test_encoder import make_perceiver, make_resolution  # type: ignore

from ocean_emulators.models.modules import PerceiverDecoder, PerceiverEncoder
from ocean_emulators.models.modules.encoder import patch_from

LATENT_DIM = 3


def test_roundtrip():
    H, W = 4, 8
    x = torch.randn(3, 10, H, W)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(180, 180),
        perceiver=make_perceiver(10, 4),
    )

    patches = patch_embed(x, make_resolution(x))

    resolution = make_resolution(x)
    _, _, nh, nw = patches.shape
    patch_h, patch_w = H // nh, W // nw
    num_latents = patch_h * patch_w

    # Perceiver input_channels = in_channels + 2 (latent + 2D pixel query)
    decode = PerceiverDecoder(
        in_channels=4,
        out_channels=10,
        patch_extent=(180, 180),
        latent_dim=LATENT_DIM,
        perceiver=make_perceiver(4 + 2, 10, num_latents=num_latents),
    )

    y_hat = decode(patches, resolution)

    assert y_hat.shape == (3, 10, H, W), (
        f"Decoder should produce full-resolution output, got {y_hat.shape}."
    )


def test_decode():
    H, W = 8, 16
    # Resolution represents the original (pre-encoder) physical grid, which is
    # larger than the decoder's latent input — mirroring the real pipeline where
    # the encoder reduces spatial dimensions.
    resolution = (
        torch.linspace(-90, 90, steps=H),
        torch.linspace(0, 360, steps=W),
    )

    # patch_from((90, 90), 8, 16) -> patch_h=4, patch_w=4
    # So nh=8/4=2, nw=16/4=4.  Input x has shape (3, 12, 2, 4).
    x = torch.randn(3, 12, 2, 4)
    patch_h, patch_w = patch_from((90, 90), H, W)
    num_latents = patch_h * patch_w

    # Perceiver input_channels = in_channels + 2 (latent + 2D pixel query)
    decode = PerceiverDecoder(
        in_channels=12,
        out_channels=24,
        patch_extent=(90, 90),
        latent_dim=LATENT_DIM,
        perceiver=make_perceiver(12 + 2, 24, num_latents=num_latents),
    )

    y_hat = decode(x, resolution)

    assert y_hat.shape == (3, 24, H, W), (
        f"Decoder should produce full-resolution output, got {y_hat.shape}."
    )
