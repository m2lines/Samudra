# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import cast

import torch

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
