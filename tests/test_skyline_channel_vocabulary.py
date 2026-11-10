# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.skyline.channel_vocabulary import ChannelVocabulary
from samudra.skyline.model import ModelConfig, PointTransformer
from samudra.skyline.optimization import accumulate_gradients
from tests.test_skyline_comparison import synthetic_source


def test_centers_roundtrip_and_open_tails():
    vocab = ChannelVocabulary([16.0] * 77, 2048)
    channels = torch.arange(77)[:, None].expand(77, 2048)
    ids = vocab.encode(vocab.centers, channels)
    assert torch.equal(ids, torch.arange(2048)[None].expand(77, 2048))
    values = torch.tensor([[-1e10, 1e10]])
    c = torch.tensor([[0, 76]])
    assert torch.equal(vocab.encode(values, c), torch.tensor([[0, 2047]]))
    assert vocab.tails(values, c).all()
    torch.testing.assert_close(
        vocab.quantized(values, c), torch.tensor([[-16.0, 16.0]])
    )
    logits = torch.zeros(1, 2, 2048, requires_grad=True)
    loss = vocab.nll(logits, values, c)
    torch.testing.assert_close(
        loss, torch.full((1, 2), torch.log(torch.tensor(2048.0)).item())
    )
    loss.sum().backward()
    assert logits.grad is not None and logits.grad.isfinite().all()


def test_backbone_and_field_initialization_match_regression():
    torch.manual_seed(20261106)
    regression = PointTransformer(
        ModelConfig(
            width=32, layers=2, heads=4, field_queries=True, objective="regression"
        )
    )
    torch.manual_seed(20261106)
    classifier = PointTransformer(
        ModelConfig(
            width=32, layers=2, heads=4, field_queries=True, objective="absolute_ce"
        )
    )
    for name, value in regression.state_dict().items():
        if not name.startswith("head."):
            assert torch.equal(value, classifier.state_dict()[name]), name


def test_classifier_gradients_and_accumulation_preserve_masked_channel_objective():
    source, frame = synthetic_source("all_fields")
    import numpy as np

    batch = source.sample(frame, np.random.default_rng(1), 2, 3, np.array([0, 18]))
    batch["importance"] = torch.tensor([0.5, 1.5])
    model = PointTransformer(
        ModelConfig(
            width=32,
            layers=1,
            heads=4,
            field_queries=True,
            objective="absolute_ce",
            bins=32,
        )
    )
    vocab = ChannelVocabulary([16.0] * 77, 32)
    logits = model(batch)
    assert logits.shape == (2, 5, 32)
    manual = (
        (
            vocab.nll(logits, batch["target"], batch["query_channel"])
            * batch["loss_weight"]
        ).sum(-1)
        * batch["importance"]
    ).mean()
    manual.backward()
    expected = {}
    for k, p in model.named_parameters():
        assert p.grad is not None
        expected[k] = p.grad.clone()
    model.zero_grad(set_to_none=True)
    stats = accumulate_gradients(
        model, lambda: batch, "comparison_ce", vocab, torch.device("cpu"), 2
    )
    torch.testing.assert_close(stats[0], manual.detach())
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, expected[name])
    # Deep SSH has neither loss nor attention influence.
    changed = dict(batch, target=batch["target"].clone())
    changed["target"][1, -1] = 1e6
    loss_changed = (
        (
            vocab.nll(logits, changed["target"], batch["query_channel"])
            * batch["loss_weight"]
        ).sum(-1)
        * batch["importance"]
    ).mean()
    torch.testing.assert_close(loss_changed, manual)


def test_classification_validation_uniform_likelihood_and_quantization():
    import numpy as np

    from samudra.skyline.comparison_eval import evaluate_comparison

    source, frame = synthetic_source("all_fields")
    source.area[:] = 1
    source.wet_area = source.wet.sum((1, 2))
    source.location_cdfs = [np.arange(1, len(p) + 1) / len(p) for p in source.locations]
    source.origins = {"validation": np.arange(4)}
    source.std = np.ones(80)
    source.read = lambda origin: frame
    vocab = ChannelVocabulary([16.0] * 77, 32)

    class Uniform(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros(*batch["target"].shape, 32)

    config = dict(
        validation_dates=4,
        validation_seed=123,
        validation_points_per_depth=2,
        microbatch=4,
        crop=3,
    )
    result = evaluate_comparison(
        Uniform(), source, config, torch.device("cpu"), 0, vocab=vocab
    )
    assert len(result["per_channel"]) == 77
    for row in result["per_channel"]:
        np.testing.assert_allclose(row["wet_cell_nll"], np.log(32), rtol=1e-6)
        assert row["quantization_full_grid_nmse"] <= row["full_grid_nmse"]
        assert row["probability_mean_full_grid_nmse"] >= 0
