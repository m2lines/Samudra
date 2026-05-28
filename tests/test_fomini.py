import pytest
import torch
from perceiver_pytorch.perceiver_io import PerceiverIO
from test_encoder import make_resolution  # type: ignore

from ocean_emulators.models.fomini import FOMini
from ocean_emulators.models.modules.augment_input import fourier_features_2d_dim
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


def make_metadata(
    channels: int, variables: int, times: int, depths: int
) -> list[tuple[int, int, int]]:
    return [(i % variables, i % times, i % depths) for i in range(channels)]


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
        input_num_freq_bands=4,
        input_max_freq=None,
        coordinate_embedding_dim=8,
        queries_dim=10,
        query_chunk_size=query_chunk_size,
        output_head_hidden_dim=14,
        output_channel_chunk_size=3,
        input_channel_metadata=make_metadata(10, 5, 2, 3),
        output_channel_metadata=make_metadata(6, 5, 2, 3),
        num_variables=5,
        num_times=2,
        num_depths=3,
        perceiver_io=make_perceiver_io(12, 10, 10),
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


def test_input_fourier_projection_shape():
    model = make_model(query_chunk_size=None)

    assert model.input_fourier_embed.in_features == fourier_features_2d_dim(4)
    assert model.input_fourier_embed.out_features == 12


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


def test_mask_information_affects_zero_input():
    model = make_model(query_chunk_size=None)
    x = torch.zeros(1, 10, 4, 8)
    prog, boundary = _split_prog_boundary(x)
    wet_ctx = make_ctx(out_channels=6, H=4, W=8)
    land_ctx = make_ctx(out_channels=6, H=4, W=8)
    land_ctx = GridContext(
        torch.zeros_like(land_ctx.label_mask),
        land_ctx.input_resolution_cpu,
        land_ctx.output_resolution_cpu,
    )

    with torch.no_grad():
        wet_out = model.forward_once(prog, boundary, wet_ctx)
        land_out = model.forward_once(prog, boundary, land_ctx)

    assert not torch.allclose(wet_out, land_out)
