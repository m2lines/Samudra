import pytest
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
        window_patches=None,
        context_patches=None,
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
        window_patches=None,
        context_patches=None,
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

    # nh=2, nw=4 latent grid, window_patches=1 -> 2*4=8 PerceiverIO calls.
    x = torch.randn(2, 12, 2, 4)

    decode = PerceiverDecoder(
        in_channels=12,
        out_channels=24,
        patch_extent=(90, 90),
        queries_dim=QUERIES_DIM,
        perceiver_io=make_decoder_perceiver_io(12, 24),
        window_patches=1,  # 1 patch per window side → 1x1 blocks of patches
        context_patches=None,
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

    full = PerceiverDecoder(**kwargs, window_patches=None, context_patches=None)
    # window_patches=4 covers the full 2x4 latent grid in one call,
    # so the windowed path should produce identical results to global.
    windowed = PerceiverDecoder(**kwargs, window_patches=4, context_patches=0)

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


def test_full_context_matches_non_windowed():
    """context_patches=None (full context) with windowed queries matches global."""
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

    full = PerceiverDecoder(**kwargs, window_patches=None, context_patches=None)
    # window_patches=1 with context_patches=None: windowed queries but every
    # window sees the full latent grid as data.
    windowed_full_ctx = PerceiverDecoder(
        **kwargs, window_patches=1, context_patches=None
    )

    full.eval()
    windowed_full_ctx.eval()
    windowed_full_ctx.load_state_dict(full.state_dict())

    with torch.no_grad():
        y_full = full(x, resolution)
        y_windowed = windowed_full_ctx(x, resolution)

    assert torch.allclose(y_full, y_windowed, atol=1e-5), (
        "Full-context windowed and non-windowed results should match."
    )


def test_context_patches_affects_output():
    """Different context_patches values should produce different outputs."""
    H, W = 8, 16
    resolution = (
        torch.linspace(-90, 90, steps=H),
        torch.linspace(0, 360, steps=W),
    )

    # nh=2, nw=4 latent grid with window_patches=1.
    x = torch.randn(2, 12, 2, 4)
    pio = make_decoder_perceiver_io(12, 24)

    kwargs = dict(
        in_channels=12,
        out_channels=24,
        patch_extent=(90, 90),
        queries_dim=QUERIES_DIM,
        perceiver_io=pio,
    )

    cp0 = PerceiverDecoder(**kwargs, window_patches=1, context_patches=0)
    cp1 = PerceiverDecoder(**kwargs, window_patches=1, context_patches=1)

    cp0.eval()
    cp1.eval()
    cp1.load_state_dict(cp0.state_dict())

    with torch.no_grad():
        y_cp0 = cp0(x, resolution)
        y_cp1 = cp1(x, resolution)

    # context_patches=0 sees only the local 1x1 patch per window,
    # context_patches=1 sees a 3x3 neighborhood — different data means
    # different cross-attention, so outputs must differ.
    assert not torch.allclose(y_cp0, y_cp1, atol=1e-5), (
        "context_patches=0 and context_patches=1 should produce different outputs."
    )


def test_more_context_closer_to_global():
    """Increasing context_patches should converge toward the global result."""
    H, W = 8, 16
    resolution = (
        torch.linspace(-90, 90, steps=H),
        torch.linspace(0, 360, steps=W),
    )

    # nh=2, nw=4 latent grid with window_patches=1.
    x = torch.randn(2, 12, 2, 4)
    pio = make_decoder_perceiver_io(12, 24)

    kwargs = dict(
        in_channels=12,
        out_channels=24,
        patch_extent=(90, 90),
        queries_dim=QUERIES_DIM,
        perceiver_io=pio,
    )

    global_dec = PerceiverDecoder(**kwargs, window_patches=None, context_patches=None)
    cp0 = PerceiverDecoder(**kwargs, window_patches=1, context_patches=0)
    cp1 = PerceiverDecoder(**kwargs, window_patches=1, context_patches=1)

    global_dec.eval()
    cp0.eval()
    cp1.eval()
    cp0.load_state_dict(global_dec.state_dict())
    cp1.load_state_dict(global_dec.state_dict())

    with torch.no_grad():
        y_global = global_dec(x, resolution)
        y_cp0 = cp0(x, resolution)
        y_cp1 = cp1(x, resolution)

    err_cp0 = (y_global - y_cp0).abs().mean().item()
    err_cp1 = (y_global - y_cp1).abs().mean().item()

    # More context should bring the windowed result closer to global.
    assert err_cp1 < err_cp0, (
        f"context_patches=1 (err={err_cp1:.6f}) should be closer to global "
        f"than context_patches=0 (err={err_cp0:.6f})."
    )


def test_context_patches_without_window_patches_raises():
    """Setting context_patches without window_patches should raise."""

    with pytest.raises(ValueError, match="window_patches must be set"):
        PerceiverDecoder(
            in_channels=12,
            out_channels=24,
            patch_extent=(90, 90),
            queries_dim=QUERIES_DIM,
            perceiver_io=make_decoder_perceiver_io(12, 24),
            window_patches=None,
            context_patches=1,
        )
