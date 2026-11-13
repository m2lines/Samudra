# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch.nn import functional as F

from samudra.skyline.channel_vocabulary import ChannelVocabulary
from samudra.skyline.model import ModelConfig, PointTransformer, smooth_linear_head
from samudra.skyline.optimization import accumulate_gradients
from tests.test_skyline_comparison import synthetic_source


def test_two_hot_physical_brackets_centers_and_tails():
    vocab = ChannelVocabulary([16.0] * 77, 32)
    centers = vocab.centers[0]
    values = torch.stack(
        (
            centers[5],
            0.75 * centers[10] + 0.25 * centers[11],
            torch.tensor(-1e10),
            torch.tensor(1e10),
        )
    )[None]
    channels = torch.zeros_like(values, dtype=torch.long)
    logits = torch.randn(1, 4, 32, requires_grad=True)
    soft = vocab.two_hot_loss(logits, values, channels)
    logp = logits.log_softmax(-1)
    expected = torch.stack(
        (
            -logp[0, 0, 5],
            -0.75 * logp[0, 1, 10] - 0.25 * logp[0, 1, 11],
            -logp[0, 2, 0],
            -logp[0, 3, -1],
        )
    )[None]
    torch.testing.assert_close(soft, expected)
    torch.testing.assert_close(
        vocab.nll(logits, values, channels),
        F.cross_entropy(
            logits.flatten(0, 1),
            vocab.encode(values, channels).flatten(),
            reduction="none",
        )[None],
    )
    soft.sum().backward()
    assert logits.grad is not None and logits.grad.isfinite().all()
    all_centers = centers[None]
    all_channels = torch.zeros_like(all_centers, dtype=torch.long)
    all_logits = torch.randn(1, 32, 32)
    torch.testing.assert_close(
        vocab.two_hot_loss(all_logits, all_centers, all_channels),
        vocab.nll(all_logits, all_centers, all_channels),
    )


def test_two_hot_accumulation_and_masked_ssh():
    source, frame = synthetic_source("all_fields")
    batch = source.sample(frame, np.random.default_rng(1), 2, 3, np.array([0, 18]))
    batch["importance"] = torch.tensor([0.5, 1.5])
    model = PointTransformer(
        ModelConfig(width=32, layers=1, heads=4, bins=32, field_queries=True)
    )
    vocab = ChannelVocabulary([16.0] * 77, 32)
    logits = model(batch)
    manual = (
        (
            vocab.two_hot_loss(logits, batch["target"], batch["query_channel"])
            * batch["loss_weight"]
        ).sum(-1)
        * batch["importance"]
    ).mean()
    manual.backward()
    grads = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        grads[name] = parameter.grad.clone()
    model.zero_grad(set_to_none=True)
    stats = accumulate_gradients(
        model, lambda: batch, "comparison_two_hot", vocab, torch.device("cpu"), 2
    )
    torch.testing.assert_close(stats[0], manual.detach())
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[name])
    changed = batch["target"].clone()
    changed[1, -1] = 1e8
    changed_loss = (
        (
            vocab.two_hot_loss(logits, changed, batch["query_channel"])
            * batch["loss_weight"]
        ).sum(-1)
        * batch["importance"]
    ).mean()
    torch.testing.assert_close(changed_loss, manual)


def test_smoothing_preserves_backbone_rng_and_head_rms():
    configs = [
        ModelConfig(
            width=32,
            layers=1,
            heads=4,
            bins=1024,
            field_queries=True,
            head_init_sigma=sigma,
        )
        for sigma in (0.0, 4.0)
    ]
    torch.manual_seed(9)
    iid = PointTransformer(configs[0])
    iid_rng = torch.get_rng_state()
    torch.manual_seed(9)
    smooth = PointTransformer(configs[1])
    smooth_rng = torch.get_rng_state()
    assert torch.equal(iid_rng, smooth_rng)
    for name, value in iid.state_dict().items():
        if not name.startswith("head."):
            assert torch.equal(value, smooth.state_dict()[name]), name
    for key in ("weight_rms", "bias_rms"):
        np.testing.assert_allclose(
            iid.initial_head_statistics["used"][key],
            smooth.initial_head_statistics["used"][key],
            rtol=1e-6,
        )
    assert abs(iid.initial_head_statistics["used"]["neighbor_weight_cosine"]) < 0.05
    assert smooth.initial_head_statistics["used"]["neighbor_weight_cosine"] > 0.95
    assert all(p.requires_grad for p in smooth.head.parameters())


def test_boundary_normalization_preserves_row_variance():
    torch.manual_seed(12)
    head = torch.nn.Linear(8192, 64, bias=False)
    original = head.weight.square().mean().item()
    smooth_linear_head(head, 4)
    row_variances = head.weight.square().mean(-1)
    # Many independent features estimate variance at each boundary and interior row.
    assert torch.all(
        (row_variances / original > 0.90) & (row_variances / original < 1.10)
    )
    torch.testing.assert_close(head.weight.square().mean(), torch.tensor(original))


def test_all_readouts_remain_exact_centers_after_smooth_init():
    source, frame = synthetic_source("all_fields")
    batch = source.sample(frame, np.random.default_rng(3), 3, 3, np.array([0, 7, 13]))
    model = PointTransformer(
        ModelConfig(
            width=32, layers=1, heads=4, bins=64, field_queries=True, head_init_sigma=4
        )
    )
    vocab = ChannelVocabulary([16.0] * 77, 64)
    logits = model(batch)
    mode, _ = vocab.predictions(logits, batch["query_channel"])
    torch.testing.assert_close(
        mode,
        vocab.centers[batch["query_channel"]]
        .gather(-1, logits.argmax(-1)[..., None])
        .squeeze(-1),
        rtol=0,
        atol=0,
    )


def test_final_discrete_cases_reconcile_depth_metric():
    from samudra.skyline.comparison_eval import evaluate_comparison

    source, frame = synthetic_source("all_fields")
    source.origins = {"validation": np.arange(2)}
    source.std = np.ones(80)
    source.read = lambda origin: frame
    model = PointTransformer(
        ModelConfig(
            width=16, layers=1, heads=4, bins=32, field_queries=True, head_init_sigma=4
        )
    )
    vocab = ChannelVocabulary([16.0] * 77, 32)
    config = dict(
        validation_dates=2,
        validation_seed=12,
        validation_points_per_depth=1,
        microbatch=19,
        crop=3,
        discrete_readout_only=True,
        model=dict(coarse_bins=0),
        record_depth13_cases=True,
        max_updates=2,
    )
    result = evaluate_comparison(
        model, source, config, torch.device("cpu"), 2, vocab=vocab
    )
    cases = result["depth13_cases"]
    assert len(cases) == 2
    metric = next(r for r in result["per_channel"] if r["channel_index"] == 51)
    np.testing.assert_allclose(
        np.mean([r["weighted_squared_error"] for r in cases]), metric["full_grid_nmse"]
    )
    for case in cases:
        assert case["prediction_normalized"] == float(
            vocab.centers[51, case["predicted_bin"]]
        )


def test_cpu_smoothing_matches_convolution_at_boundaries_and_short_vocabularies():
    """Compare the CPU implementation with the original padded-convolution formula."""
    import copy

    for bins, width in ((2, 1), (17, 7), (64, 32), (8192, 8)):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(19)
            actual = torch.nn.Linear(width, bins, dtype=dtype)
            expected = copy.deepcopy(actual)
            offsets = torch.arange(-16, 17, dtype=dtype)
            kernel = torch.exp(-0.5 * (offsets / 4).square())[None, None]
            norm = F.conv1d(
                torch.ones(1, 1, bins, dtype=dtype), kernel.square(), padding=16
            ).sqrt()
            with torch.no_grad():
                for parameter in (expected.weight, expected.bias):
                    rms = parameter.square().mean().sqrt()
                    values = parameter.reshape(bins, -1).T[:, None]
                    smoothed = (F.conv1d(values, kernel, padding=16) / norm)[
                        :, 0
                    ].T.reshape_as(parameter)
                    smoothed *= rms / smoothed.square().mean().sqrt()
                    parameter.copy_(smoothed)
            rng = torch.get_rng_state()
            smooth_linear_head(actual, 4)
            assert torch.equal(rng, torch.get_rng_state())
            tolerance = 2e-7 if dtype == torch.float32 else 1e-14
            torch.testing.assert_close(
                actual.weight, expected.weight, atol=tolerance, rtol=tolerance
            )
            torch.testing.assert_close(
                actual.bias, expected.bias, atol=tolerance, rtol=tolerance
            )
