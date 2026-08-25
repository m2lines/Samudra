# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from test_encoder import make_resolution  # type: ignore

from samudra.config import DecoderConfig
from samudra.models.modules import (
    DirectCrossAttentionIO,
    Perceiver,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
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


def make_direct_cross_attention_io(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS):
    return DirectCrossAttentionIO(
        input_dim=in_channels,
        queries_dim=QUERIES_DIM,
        output_dim=out_channels,
        heads=2,
        dim_head=8,
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
        output_overlap_patches=reference.output_overlap_patches,
        processor_conditioning=reference.processor_conditioner is not None,
    )
    kwargs.update(overrides)
    other = PerceiverDecoder(**kwargs)  # type: ignore

    reference.eval()
    other.eval()
    other.load_state_dict(reference.state_dict())
    return other


def serial_overlapping_decode(
    decoder: PerceiverDecoder,
    data_grid: torch.Tensor,
    queries_grid: torch.Tensor,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    """Reference implementation retained to test vectorized numerical parity."""
    batch, nh, nw, _ = data_grid.shape
    height, width, _ = queries_grid.shape
    wp = decoder.window_patches
    cp = decoder.context_patches
    op = decoder.output_overlap_patches
    assert wp is not None and op > 0

    weighted = data_grid.new_zeros(batch, height, width, decoder.decoded_channels)
    weight_sum = data_grid.new_zeros(height, width, 1)
    n_blocks_h = nh // wp
    n_blocks_w = nw // wp
    halo_h = op * patch_h
    halo_w = op * patch_w

    if cp == 0:
        data = rearrange(data_grid, "b nh nw c -> b c nh nw")
        data_windows = data.unfold(2, wp, wp).unfold(3, wp, wp)
    else:
        assert cp is not None
        data = rearrange(data_grid, "b nh nw c -> b c nh nw")
        data = F.pad(data, (cp, cp, 0, 0), mode="circular")
        data = F.pad(data, (0, 0, cp, cp), mode="constant", value=0)
        window_size = wp + 2 * cp
        data_windows = data.unfold(2, window_size, wp).unfold(3, window_size, wp)

    for bi in range(n_blocks_h):
        query_i0 = max(0, (bi * wp - op) * patch_h)
        query_i1 = min(height, ((bi + 1) * wp + op) * patch_h)
        lat_indices = torch.arange(query_i0, query_i1, device=data_grid.device)
        lat_weights = decoder._overlap_weights(
            len(lat_indices),
            halo_h,
            fade_start=bi > 0,
            fade_end=bi + 1 < n_blocks_h,
            device=data_grid.device,
            dtype=data_grid.dtype,
        )
        for bj in range(n_blocks_w):
            lon_indices = torch.arange(
                (bj * wp - op) * patch_w,
                ((bj + 1) * wp + op) * patch_w,
                device=data_grid.device,
            ).remainder(width)
            local_data = rearrange(data_windows[:, :, bi, bj], "b c h w -> b (h w) c")
            local_queries = queries_grid.index_select(0, lat_indices)
            local_queries = local_queries.index_select(1, lon_indices)
            local_out = decoder._decode_queries(
                local_data, rearrange(local_queries, "h w d -> (h w) d")
            )
            local_out = rearrange(
                local_out,
                "b (h w) c -> b h w c",
                h=len(lat_indices),
                w=len(lon_indices),
            )
            lon_weights = decoder._overlap_weights(
                len(lon_indices),
                halo_w,
                fade_start=True,
                fade_end=True,
                device=data_grid.device,
                dtype=data_grid.dtype,
            )
            weights = lat_weights[:, None] * lon_weights[None, :]
            weighted[:, lat_indices[:, None], lon_indices[None, :], :] += (
                local_out * weights[None, :, :, None]
            )
            weight_sum[lat_indices[:, None], lon_indices[None, :], :] += weights[
                :, :, None
            ]

    return rearrange(weighted / weight_sum, "b h w c -> b c h w")


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


def test_overlapping_direct_decoder_supports_backward(
    resolution, latent_input, decoder_kwargs
):
    decoder_kwargs["perceiver_io"] = make_direct_cross_attention_io()
    decoder = PerceiverDecoder(
        **decoder_kwargs,
        window_patches=2,
        context_patches=0,
        output_overlap_patches=1,
        processor_conditioning=True,
    )
    latent_input.requires_grad_()

    output = decoder(latent_input, resolution)
    output.square().mean().backward()

    assert output.shape == (BATCH, OUT_CHANNELS, H, W)
    assert latent_input.grad is not None
    assert torch.isfinite(latent_input.grad).all()
    assert decoder.conditioning_strength is not None
    assert decoder.conditioning_strength.grad is not None


@pytest.mark.parametrize("context_patches", [0, 1])
@pytest.mark.parametrize(
    ("max_window_batch_size", "expected_decode_calls"),
    [(128, 2), (4, 5)],
)
def test_vectorized_overlap_matches_serial_forward_and_backward(
    context_patches,
    max_window_batch_size,
    expected_decode_calls,
):
    """Window batching preserves outputs and gradients, including polar halos."""
    torch.manual_seed(0)
    kwargs = dict(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_extent=(30.0, 60.0),
        queries_dim=QUERIES_DIM,
        window_patches=2,
        context_patches=context_patches,
        output_overlap_patches=1,
        processor_conditioning=False,
    )
    vectorized = PerceiverDecoder(
        **kwargs, perceiver_io=make_direct_cross_attention_io()
    )
    vectorized._MAX_WINDOW_BATCH_SIZE = max_window_batch_size
    serial = PerceiverDecoder(**kwargs, perceiver_io=make_direct_cross_attention_io())
    serial.load_state_dict(vectorized.state_dict())

    vectorized_data = torch.randn(BATCH, 6, 6, IN_CHANNELS, requires_grad=True)
    serial_data = vectorized_data.detach().clone().requires_grad_()
    vectorized_queries = torch.randn(12, 24, QUERIES_DIM, requires_grad=True)
    serial_queries = vectorized_queries.detach().clone().requires_grad_()

    with patch.object(
        vectorized, "_decode_queries", wraps=vectorized._decode_queries
    ) as decode_queries:
        actual = vectorized._decode_overlapping(
            vectorized_data, vectorized_queries, patch_h=2, patch_w=4
        )
    assert decode_queries.call_count == expected_decode_calls
    expected = serial_overlapping_decode(
        serial, serial_data, serial_queries, patch_h=2, patch_w=4
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    output_gradient = torch.randn_like(actual)
    actual.backward(output_gradient)
    expected.backward(output_gradient)
    torch.testing.assert_close(
        vectorized_data.grad, serial_data.grad, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        vectorized_queries.grad, serial_queries.grad, atol=2e-5, rtol=2e-5
    )
    for actual_parameter, expected_parameter in zip(
        vectorized.parameters(), serial.parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=2e-5,
            rtol=2e-5,
        )


def test_overlap_requires_window_and_fits_inside_core():
    with pytest.raises(ValueError, match="window_patches must be set"):
        PerceiverDecoder(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            patch_extent=PATCH_EXTENT,
            queries_dim=QUERIES_DIM,
            perceiver_io=make_decoder_perceiver_io(),
            window_patches=None,
            context_patches=None,
            output_overlap_patches=1,
        )

    with pytest.raises(ValueError, match="must not exceed half"):
        PerceiverDecoder(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            patch_extent=PATCH_EXTENT,
            queries_dim=QUERIES_DIM,
            perceiver_io=make_decoder_perceiver_io(),
            window_patches=2,
            context_patches=0,
            output_overlap_patches=2,
        )


def test_decoder_config_builds_direct_cross_attention():
    config = DecoderConfig.model_validate(
        {
            "architecture": "direct_cross_attention",
            "queries_dim": QUERIES_DIM,
            "perceiver": {"cross_heads": 2, "cross_dim_head": 8},
        }
    )

    decoder = config.build(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_extent=(1.0, 1.0),
        implementation="naive",
    )

    assert isinstance(decoder.perceiver_io, DirectCrossAttentionIO)


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
