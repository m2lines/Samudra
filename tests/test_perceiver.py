# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import cast

import pytest
import torch

from samudra.config import PerceiverConfig
from samudra.models.modules import Attention, Perceiver, PerceiverIO


def test_attention_matches_explicit_reference_with_mask():
    attention = Attention(
        query_dim=12,
        context_dim=10,
        heads=3,
        dim_head=4,
        backend="math",
    ).eval()
    queries = torch.randn(2, 5, 12)
    context = torch.randn(2, 7, 10)
    mask = torch.tensor([[True, True, False, True, False, True, True], [True] * 7])

    actual = attention(queries, context=context, mask=mask)

    query = attention._split_heads(attention.to_q(queries))
    key, value = attention.to_kv(context).chunk(2, dim=-1)
    key = attention._split_heads(key)
    value = attention._split_heads(value)
    scores = query @ key.transpose(-2, -1) / math.sqrt(attention.dim_head)
    scores = scores.masked_fill(~mask[:, None, None], float("-inf"))
    expected = scores.softmax(dim=-1) @ value
    expected = expected.transpose(1, 2).contiguous().flatten(2)
    expected = attention.to_out(expected)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_attention_preserves_additive_mask_semantics():
    attention = Attention(
        query_dim=12,
        context_dim=10,
        heads=3,
        dim_head=4,
        backend="math",
    ).eval()
    queries = torch.randn(2, 5, 12)
    context = torch.randn(2, 7, 10)
    boolean_mask = torch.tensor(
        [[True, True, False, True, False, True, True], [True] * 7]
    )
    additive_mask = torch.where(
        boolean_mask,
        torch.tensor(0.0),
        torch.tensor(float("-inf")),
    )

    with torch.no_grad():
        expected = attention(queries, context=context, mask=boolean_mask)
        actual = attention(queries, context=context, mask=additive_mask)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("additive", [False, True])
def test_attention_fully_masked_sample_stays_finite(additive: bool):
    attention = Attention(
        query_dim=8,
        context_dim=8,
        heads=2,
        dim_head=4,
        backend="math",
    )
    queries = torch.randn(2, 3, 8, requires_grad=True)
    context = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[False] * 5, [True] * 5])
    if additive:
        mask = torch.where(mask, torch.tensor(0.0), torch.tensor(float("-inf")))

    output = attention(queries, context=context, mask=mask)
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, 6, dtype=torch.bool),
        torch.ones(2, 1, 7, dtype=torch.bool),
    ],
)
def test_attention_rejects_mask_with_wrong_shape(mask: torch.Tensor):
    attention = Attention(query_dim=8, context_dim=8)

    with pytest.raises(ValueError, match="mask must have shape"):
        attention(
            torch.randn(2, 3, 8),
            context=torch.randn(2, 7, 8),
            mask=mask,
        )


def test_attention_rejects_integer_mask():
    attention = Attention(query_dim=8, context_dim=8)

    with pytest.raises(TypeError, match="Boolean or floating-point"):
        attention(
            torch.randn(2, 3, 8),
            context=torch.randn(2, 7, 8),
            mask=torch.ones(2, 7, dtype=torch.int64),
        )


def test_attention_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported attention backend"):
        Attention(query_dim=8, backend="unknown")  # type: ignore[arg-type]


def test_perceiver_encoder_shapes_and_weight_tying():
    model = Perceiver(
        num_freq_bands=2,
        depth=3,
        max_freq=4,
        input_channels=5,
        input_axis=2,
        num_latents=4,
        latent_dim=8,
        num_classes=6,
        weight_tie_layers=True,
        self_per_cross_attn=2,
    )

    output = model(torch.randn(2, 3, 5, 5))
    embeddings = model(torch.randn(2, 3, 5, 5), return_embeddings=True)

    assert output.shape == (2, 6)
    assert embeddings.shape == (2, 4, 8)
    layers = [cast(torch.nn.ModuleList, layer) for layer in model.layers]
    assert layers[0][0] is not layers[1][0]
    assert layers[1][0] is layers[2][0]


def test_perceiver_flattens_mask_over_input_axes():
    model = Perceiver(
        num_freq_bands=2,
        depth=1,
        max_freq=4,
        input_channels=5,
        input_axis=2,
        num_latents=4,
        latent_dim=8,
        num_classes=6,
    )
    data = torch.randn(2, 3, 5, 5)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    mask[:, 1, 2] = False

    output = model(data, mask=mask)

    assert output.shape == (2, 6)


def test_perceiver_io_supports_shared_and_batched_queries():
    model = PerceiverIO(
        depth=2,
        dim=7,
        queries_dim=6,
        logits_dim=5,
        num_latents=4,
        latent_dim=8,
        weight_tie_layers=True,
        decoder_ff=True,
    ).eval()
    data = torch.randn(2, 9, 7)
    shared_queries = torch.randn(11, 6)
    batched_queries = shared_queries.unsqueeze(0).expand(2, -1, -1)

    with torch.no_grad():
        shared_output = model(data, queries=shared_queries)
        batched_output = model(data, queries=batched_queries)

    assert shared_output.shape == (2, 11, 5)
    assert torch.allclose(shared_output, batched_output)


def test_perceiver_io_preserves_legacy_decoder_without_query_residual():
    model = PerceiverIO(
        depth=1,
        dim=7,
        queries_dim=6,
        num_latents=4,
        latent_dim=8,
        decoder_ff=False,
    ).eval()
    for parameter in model.decoder_cross_attn.parameters():
        torch.nn.init.zeros_(parameter)

    data = torch.randn(2, 9, 7)
    queries = torch.randn(2, 11, 6)

    with torch.no_grad():
        output = model(data, queries=queries)

    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize(
    ("implementation", "backend"),
    [("auto", "auto"), ("naive", "math"), ("flash", "flash")],
)
def test_config_maps_implementation_to_sdpa_backend(implementation, backend):
    config = PerceiverConfig(depth=1, latent_dim=8, num_latents=4)

    encoder = config.build(
        in_channels=5,
        out_channels=6,
        max_patch_size=(3, 5),
        implementation=implementation,
    )
    decoder = config.build_io(
        in_channels=7,
        queries_dim=6,
        out_channels=5,
        implementation=implementation,
    )

    assert isinstance(encoder, torch.nn.Sequential)
    assert isinstance(encoder[-1], Perceiver)
    assert isinstance(decoder, PerceiverIO)
    encoder_attention = next(
        module for module in encoder.modules() if isinstance(module, Attention)
    )
    decoder_attention = next(
        module for module in decoder.modules() if isinstance(module, Attention)
    )
    assert encoder_attention.backend == backend
    assert decoder_attention.backend == backend


@pytest.mark.cuda
def test_attention_runs_forced_flash_forward_and_backward():
    if not torch.backends.cuda.is_flash_attention_available():
        pytest.skip("PyTorch was built without FlashAttention support")

    attention = (
        Attention(
            query_dim=64,
            heads=1,
            dim_head=64,
            backend="flash",
        )
        .cuda()
        .bfloat16()
    )
    inputs = torch.randn(
        2,
        32,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output = attention(inputs)
    output.float().square().mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
