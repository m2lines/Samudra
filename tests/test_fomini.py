# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from perceiver_pytorch.perceiver_io import PerceiverIO
from test_encoder import make_resolution  # type: ignore

from ocean_emulators.models.fomini import FOMini
from ocean_emulators.utils.ctx import GridContext


def make_perceiver_io(
    in_channels: int, out_channels: int, queries_dim: int
) -> PerceiverIO:
    return PerceiverIO(
        depth=2,
        dim=in_channels,
        queries_dim=queries_dim,
        logits_dim=out_channels,
        num_latents=8,
        latent_dim=8,
        weight_tie_layers=True,
        decoder_ff=True,
    )


def make_ctx(out_channels: int, H: int, W: int) -> GridContext:
    mask = torch.ones(out_channels, H, W, dtype=torch.bool)
    dummy = torch.randn(1, 1, H, W)
    res = make_resolution(dummy)
    return GridContext(mask, res, res)


def make_model(query_chunk_size: int | None) -> FOMini:
    return FOMini(
        in_channels=10,
        out_channels=6,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        input_embedding_dim=12,
        coordinate_embedding_dim=8,
        queries_dim=10,
        query_chunk_size=query_chunk_size,
        perceiver_io=make_perceiver_io(12, 6, 10),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
    )


def _split_prog_boundary(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # FOMini expects in_channels=10; split arbitrarily into 6 prog + 4 boundary.
    return x[:, :6], x[:, 6:]


def test_forward_shape():
    model = make_model(query_chunk_size=None)
    x = torch.randn(2, 10, 4, 8)
    prog, boundary = _split_prog_boundary(x)
    out = model.forward_once(prog, boundary, make_ctx(out_channels=6, H=4, W=8))
    assert out.shape == (2, 6, 4, 8)


def test_chunked_queries_match_full_decode():
    x = torch.randn(2, 10, 4, 8)
    prog, boundary = _split_prog_boundary(x)
    ctx = make_ctx(out_channels=6, H=4, W=8)

    full = make_model(query_chunk_size=None)
    chunked = make_model(query_chunk_size=5)
    chunked.load_state_dict(full.state_dict())

    full.eval()
    chunked.eval()

    with torch.no_grad():
        y_full = full.forward_once(prog, boundary, ctx)
        y_chunked = chunked.forward_once(prog, boundary, ctx)
    assert torch.allclose(y_full, y_chunked, atol=1e-5)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError, match="query_chunk_size must be positive"):
        _ = make_model(query_chunk_size=0)
