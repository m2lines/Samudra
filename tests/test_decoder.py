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
        latent_dim=12,
        patch_extent=(180, 180),
        perceiver=make_perceiver(4, 12, num_latents=6, input_axis=1),
    )

    y_hat = decode(patches, make_resolution(x))

    assert y_hat.shape == x.shape
