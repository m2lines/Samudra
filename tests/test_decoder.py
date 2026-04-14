import pytest
import torch
from perceiver_pytorch.perceiver_io import PerceiverIO
from test_encoder import make_resolution  # type: ignore

from ocean_emulators.models.modules import PerceiverDecoder, PerceiverEncoder

# Small values for fast tests.
LATENT_DIM = 8
QUERIES_DIM = 16
NUM_LATENTS = 4

IN_CHANNELS = 12
OUT_CHANNELS = 24
PATCH_EXTENT = (90.0, 90.0)
BATCH = 2

# With patch_extent=(90, 90) and H=8, W=16:
#   patch_h=4, patch_w=4  →  nh=2, nw=4
NH, NW = 2, 4
H, W = 8, 16


ENCODER_LATENT_DIM = 4


def make_perceiver_encoder(prog_channels, *, num_latents=2, max_freq=10.0):
    """Build a 2-D Perceiver for the encoder's prognostic stream."""
    from perceiver_pytorch import Perceiver

    return Perceiver(
        num_freq_bands=4,
        max_freq=max_freq,
        depth=2,
        input_axis=2,
        input_channels=prog_channels,
        latent_dim=ENCODER_LATENT_DIM,
        num_latents=num_latents,
        num_classes=ENCODER_LATENT_DIM,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )


def make_decoder_perceiver_io(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS):
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


@pytest.fixture()
def resolution():
    """Standard (H, W) = (8, 16) resolution grid."""
    return (
        torch.linspace(-90, 90, steps=H),
        torch.linspace(0, 360, steps=W),
    )


@pytest.fixture()
def latent_input():
    """Latent grid tensor with shape (BATCH, IN_CHANNELS, NH, NW)."""
    return torch.randn(BATCH, IN_CHANNELS, NH, NW)


@pytest.fixture()
def decoder_kwargs():
    """Common kwargs for building a PerceiverDecoder (without windowing args).

    Uses a single shared PerceiverIO so weight-sharing tests can rely on
    identical parameters across decoders built from the same fixture.
    """
    return dict(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_extent=PATCH_EXTENT,
        queries_dim=QUERIES_DIM,
        perceiver_io=make_decoder_perceiver_io(),
    )


def make_decoder_with_shared_weights(
    reference: PerceiverDecoder,
    **overrides,
) -> PerceiverDecoder:
    """Clone a decoder with different windowing params but shared weights.

    Builds a new PerceiverDecoder from *reference*'s config, applies
    *overrides* (e.g. ``window_patches``, ``context_patches``), copies
    weights, and sets eval mode on both.
    """
    kwargs = dict(
        in_channels=reference.in_channels,
        out_channels=reference.out_channels,
        patch_extent=reference.patch_extent,
        queries_dim=reference.query_embed.out_features,
        perceiver_io=reference.perceiver_io,
        window_patches=reference.window_patches,
        context_patches=reference.context_patches,
    )
    kwargs.update(overrides)
    other = PerceiverDecoder(**kwargs)  # type: ignore

    reference.eval()
    other.eval()
    other.load_state_dict(reference.state_dict())
    return other


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_roundtrip():
    H_rt, W_rt = 4, 8
    embed_dim = 4
    prog = torch.randn(3, 7, H_rt, W_rt)
    boundary = torch.randn(3, 3, H_rt, W_rt)

    patch_embed = PerceiverEncoder(
        prog_channels=7,
        boundary_channels=3,
        out_channels=embed_dim,
        prog_latent_dim=ENCODER_LATENT_DIM,
        boundary_latent_dim=ENCODER_LATENT_DIM,
        patch_extent=(180, 180),
        perceiver=make_perceiver_encoder(7),
        boundary_perceiver=make_perceiver_encoder(3),
    )

    res = make_resolution(prog)
    patches = patch_embed(prog, boundary, res)

    decode = PerceiverDecoder(
        in_channels=embed_dim,
        out_channels=10,
        patch_extent=(180, 180),
        queries_dim=QUERIES_DIM,
        perceiver_io=make_decoder_perceiver_io(embed_dim, 10),
        window_patches=None,
        context_patches=None,
    )

    y_hat = decode(patches, res)

    assert y_hat.shape == (3, 10, H_rt, W_rt), (
        f"Decoder should produce full-resolution output, got {y_hat.shape}."
    )


def test_decode(resolution, latent_input, decoder_kwargs):
    decode = PerceiverDecoder(
        **decoder_kwargs, window_patches=None, context_patches=None
    )
    y_hat = decode(latent_input, resolution)

    assert y_hat.shape == (BATCH, OUT_CHANNELS, H, W), (
        f"Decoder should produce full-resolution output, got {y_hat.shape}."
    )


def test_windowed_decode(resolution, latent_input, decoder_kwargs):
    """At high resolution, windowing splits queries into fixed-size chunks."""
    decode = PerceiverDecoder(**decoder_kwargs, window_patches=1, context_patches=None)
    y_hat = decode(latent_input, resolution)

    assert y_hat.shape == (BATCH, OUT_CHANNELS, H, W), (
        f"Windowed decoder should produce full-resolution output, got {y_hat.shape}."
    )


def test_windowed_matches_non_windowed(resolution, latent_input, decoder_kwargs):
    """Windowed with full context should match non-windowed decoding."""
    full = PerceiverDecoder(**decoder_kwargs, window_patches=None, context_patches=None)
    # window_patches=2 divides nh=2 and nw=4; context_patches=None gives
    # every window the full latent grid as data, so the only difference
    # from global is that queries are tiled into blocks.
    windowed = make_decoder_with_shared_weights(
        full, window_patches=2, context_patches=None
    )

    with torch.no_grad():
        y_full = full(latent_input, resolution)
        y_windowed = windowed(latent_input, resolution)

    assert torch.allclose(y_full, y_windowed, atol=1e-5), (
        "Windowed and non-windowed results should match."
    )


def test_full_context_matches_non_windowed(resolution, latent_input, decoder_kwargs):
    """context_patches=None (full context) with windowed queries matches global."""
    full = PerceiverDecoder(**decoder_kwargs, window_patches=None, context_patches=None)
    # window_patches=1 with context_patches=None: windowed queries but every
    # window sees the full latent grid as data.
    windowed_full_ctx = make_decoder_with_shared_weights(
        full, window_patches=1, context_patches=None
    )

    with torch.no_grad():
        y_full = full(latent_input, resolution)
        y_windowed = windowed_full_ctx(latent_input, resolution)

    assert torch.allclose(y_full, y_windowed, atol=1e-5), (
        "Full-context windowed and non-windowed results should match."
    )


def test_context_patches_affects_output(resolution, latent_input, decoder_kwargs):
    """Different context_patches values should produce different outputs."""
    cp0 = PerceiverDecoder(**decoder_kwargs, window_patches=1, context_patches=0)
    cp1 = make_decoder_with_shared_weights(cp0, context_patches=1)

    with torch.no_grad():
        y_cp0 = cp0(latent_input, resolution)
        y_cp1 = cp1(latent_input, resolution)

    # context_patches=0 sees only the local 1x1 patch per window,
    # context_patches=1 sees a 3x3 neighborhood — different data means
    # different cross-attention, so outputs must differ.
    assert not torch.allclose(y_cp0, y_cp1, atol=1e-5), (
        "context_patches=0 and context_patches=1 should produce different outputs."
    )


def test_more_context_closer_to_global(resolution, latent_input, decoder_kwargs):
    """Increasing context_patches should converge toward the global result."""
    global_dec = PerceiverDecoder(
        **decoder_kwargs, window_patches=None, context_patches=None
    )
    cp0 = make_decoder_with_shared_weights(
        global_dec, window_patches=1, context_patches=0
    )
    cp1 = make_decoder_with_shared_weights(
        global_dec, window_patches=1, context_patches=1
    )

    with torch.no_grad():
        y_global = global_dec(latent_input, resolution)
        y_cp0 = cp0(latent_input, resolution)
        y_cp1 = cp1(latent_input, resolution)

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
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            patch_extent=PATCH_EXTENT,
            queries_dim=QUERIES_DIM,
            perceiver_io=make_decoder_perceiver_io(),
            window_patches=None,
            context_patches=1,
        )
