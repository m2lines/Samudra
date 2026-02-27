import torch
from perceiver_pytorch.perceiver_io import PerceiverIO
from test_encoder import make_resolution  # type: ignore

from ocean_emulators.models.modules import PerceiverDecoder, PerceiverEncoder

# Small values for fast tests.
LATENT_DIM = 8
QUERIES_DIM = 16
NUM_LATENTS = 4


def make_perceiver_encoder(in_channels, out_channels, *, num_latents=2):
    """Build a regular Perceiver for the encoder (uses mean-pooling)."""
    from perceiver_pytorch import Perceiver

    return Perceiver(
        num_freq_bands=4,
        max_freq=1.0,
        depth=2,
        input_axis=2,
        input_channels=in_channels,
        latent_dim=3,
        num_latents=num_latents,
        num_classes=out_channels,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )


def make_decoder_perceiver_io(in_channels, out_channels):
    """Build a PerceiverIO for the decoder."""
    return PerceiverIO(
        depth=2,
        dim=in_channels,
        queries_dim=QUERIES_DIM,
        logits_dim=out_channels,
        num_latents=NUM_LATENTS,
        latent_dim=LATENT_DIM,
        weight_tie_layers=True,
        decoder_ff=True,
    )


def test_roundtrip():
    H, W = 4, 8
    x = torch.randn(3, 10, H, W)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(180, 180),
        perceiver=make_perceiver_encoder(10, 4),
    )

    patches = patch_embed(x, make_resolution(x))
    resolution = make_resolution(x)

    decode = PerceiverDecoder(
        in_channels=4,
        out_channels=10,
        patch_extent=(180, 180),
        queries_dim=QUERIES_DIM,
        perceiver_io=make_decoder_perceiver_io(4, 10),
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

    decode = PerceiverDecoder(
        in_channels=12,
        out_channels=24,
        patch_extent=(90, 90),
        queries_dim=QUERIES_DIM,
        perceiver_io=make_decoder_perceiver_io(12, 24),
    )

    y_hat = decode(x, resolution)

    assert y_hat.shape == (3, 24, H, W), (
        f"Decoder should produce full-resolution output, got {y_hat.shape}."
    )


def test_windowed_decode():
    """At high resolution, windowing splits queries into fixed-size chunks."""
    H, W = 8, 16
    resolution = (
        torch.linspace(-90, 90, steps=H),
        torch.linspace(0, 360, steps=W),
    )

    # H*W = 128 pixels total, window_size=32 -> 4 PerceiverIO calls.
    x = torch.randn(2, 12, 2, 4)

    decode = PerceiverDecoder(
        in_channels=12,
        out_channels=24,
        patch_extent=(90, 90),
        queries_dim=QUERIES_DIM,
        perceiver_io=make_decoder_perceiver_io(12, 24),
        window_size=32,
    )

    y_hat = decode(x, resolution)

    assert y_hat.shape == (2, 24, H, W), (
        f"Windowed decoder should produce full-resolution output, got {y_hat.shape}."
    )


def test_windowed_matches_non_windowed():
    """Windowed and non-windowed decoding should produce identical results."""
    H, W = 4, 8
    resolution = (
        torch.linspace(-90, 90, steps=H),
        torch.linspace(0, 360, steps=W),
    )

    x = torch.randn(2, 12, 2, 4)
    pio = make_decoder_perceiver_io(12, 24)

    kwargs = dict(
        in_channels=12,
        out_channels=24,
        patch_extent=(90, 90),
        queries_dim=QUERIES_DIM,
        perceiver_io=pio,
    )

    full = PerceiverDecoder(**kwargs)
    windowed = PerceiverDecoder(**kwargs, window_size=8)

    full.eval()
    windowed.eval()

    # Share the same parameters (same pio and same query_embed/pos/scale).
    windowed.load_state_dict(full.state_dict())

    with torch.no_grad():
        y_full = full(x, resolution)
        y_windowed = windowed(x, resolution)

    assert torch.allclose(y_full, y_windowed, atol=1e-5), (
        "Windowed and non-windowed results should match."
    )
