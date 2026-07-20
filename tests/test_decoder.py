# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from perceiver_pytorch.perceiver_io import PerceiverIO
from test_encoder import make_resolution  # type: ignore

from samudra.models.modules import (
    DirectPatchDecoder,
    PerceiverDecoder,
    PerceiverEncoder,
)

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
        window_batch_size=reference.window_batch_size,
        fine_scale_in_channels=(
            None
            if reference.fine_scale_query_embed is None
            else reference.fine_scale_query_embed.in_channels
        ),
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
    x = torch.randn(3, 10, H_rt, W_rt)

    patch_embed = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=(180, 180),
        perceiver=make_perceiver_encoder(10, 4),
    )

    patches = patch_embed(x, make_resolution(x))
    res = make_resolution(x)

    decode = PerceiverDecoder(
        in_channels=4,
        out_channels=10,
        patch_extent=(180, 180),
        queries_dim=QUERIES_DIM,
        perceiver_io=make_decoder_perceiver_io(4, 10),
        window_patches=None,
        context_patches=None,
    )

    y_hat = decode(patches, res)

    assert y_hat.shape == (3, 10, H_rt, W_rt), (
        f"Decoder should produce full-resolution output, got {y_hat.shape}."
    )


def test_one_pixel_perceiver_roundtrip():
    height, width = 4, 8
    x = torch.randn(2, 10, height, width)
    patch_extent = (180 / height, 360 / width)
    resolution = make_resolution(x)
    encoder = PerceiverEncoder(
        in_channels=10,
        out_channels=4,
        patch_extent=patch_extent,
        perceiver=make_perceiver_encoder(10, 4),
    )
    decoder = PerceiverDecoder(
        in_channels=4,
        out_channels=10,
        patch_extent=patch_extent,
        queries_dim=QUERIES_DIM,
        perceiver_io=make_decoder_perceiver_io(4, 10),
        window_patches=1,
        context_patches=0,
    )

    output = decoder(encoder(x, resolution), resolution)

    assert output.shape == x.shape


def test_decode(resolution, latent_input, decoder_kwargs):
    decode = PerceiverDecoder(
        **decoder_kwargs, window_patches=None, context_patches=None
    )
    y_hat = decode(latent_input, resolution)

    assert y_hat.shape == (BATCH, OUT_CHANNELS, H, W), (
        f"Decoder should produce full-resolution output, got {y_hat.shape}."
    )


def test_direct_decoder_preserves_one_pixel_grid(resolution):
    decoder = DirectPatchDecoder(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_extent=(180 / H, 360 / W),
    )
    x = torch.randn(BATCH, IN_CHANNELS, H, W, requires_grad=True)

    output = decoder(x, resolution)

    assert output.shape == (BATCH, OUT_CHANNELS, H, W)
    output.sum().backward()
    assert decoder.projection.weight.grad is not None


def test_direct_decoder_rejects_mismatched_grid(resolution):
    decoder = DirectPatchDecoder(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_extent=(180 / H, 360 / W),
    )

    with pytest.raises(ValueError, match="processor and output grids to match"):
        decoder(torch.randn(BATCH, IN_CHANNELS, H // 2, W), resolution)


def test_windowed_decode(resolution, latent_input, decoder_kwargs):
    """At high resolution, windowing splits queries into fixed-size chunks."""
    decode = PerceiverDecoder(**decoder_kwargs, window_patches=1, context_patches=None)
    y_hat = decode(latent_input, resolution)

    assert y_hat.shape == (BATCH, OUT_CHANNELS, H, W), (
        f"Windowed decoder should produce full-resolution output, got {y_hat.shape}."
    )


@pytest.mark.parametrize("window_patches", [None, 1])
def test_fine_scale_queries_preserve_baseline_at_initialization(
    resolution, latent_input, decoder_kwargs, window_patches
):
    baseline = PerceiverDecoder(
        **decoder_kwargs,
        window_patches=window_patches,
        context_patches=None,
    )
    fine_scale = PerceiverDecoder(
        **decoder_kwargs,
        window_patches=window_patches,
        context_patches=None,
        fine_scale_in_channels=5,
    )
    fine_scale.load_state_dict(baseline.state_dict(), strict=False)
    baseline.eval()
    fine_scale.eval()
    inputs = torch.randn(BATCH, 5, H, W)

    with torch.no_grad():
        expected = baseline(latent_input, resolution)
        actual = fine_scale(latent_input, resolution, inputs)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_fine_scale_queries_learn_from_zero_initialization(
    resolution, latent_input, decoder_kwargs
):
    decoder = PerceiverDecoder(
        **decoder_kwargs,
        window_patches=1,
        context_patches=None,
        fine_scale_in_channels=5,
    )
    assert decoder.fine_scale_query_embed is not None
    inputs = torch.randn(BATCH, 5, H, W)

    initial_output = decoder(latent_input, resolution, inputs)
    initial_output.square().mean().backward()

    gradient = decoder.fine_scale_query_embed.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0

    with torch.no_grad():
        decoder.fine_scale_query_embed.weight.add_(gradient, alpha=-0.01)
        updated_output = decoder(latent_input, resolution, inputs)
    assert not torch.allclose(updated_output, initial_output)


def test_fine_scale_queries_require_matching_features(
    resolution, latent_input, decoder_kwargs
):
    decoder = PerceiverDecoder(
        **decoder_kwargs,
        window_patches=None,
        context_patches=None,
        fine_scale_in_channels=5,
    )

    with pytest.raises(ValueError, match="fine_scale_features are required"):
        decoder(latent_input, resolution)
    with pytest.raises(
        ValueError, match="must match the decoder batch and output grid"
    ):
        decoder(latent_input, resolution, torch.randn(BATCH, 5, H // 2, W))


@pytest.mark.parametrize("context_patches", [None, 0, 1])
def test_vectorized_windows_match_sequential_windows(
    resolution, latent_input, decoder_kwargs, context_patches
):
    sequential = PerceiverDecoder(
        **decoder_kwargs,
        window_patches=1,
        context_patches=context_patches,
        window_batch_size=1,
    )
    vectorized = make_decoder_with_shared_weights(sequential, window_batch_size=None)

    with torch.no_grad():
        expected = sequential(latent_input, resolution)
        actual = vectorized(latent_input, resolution)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


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
