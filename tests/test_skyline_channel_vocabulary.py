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


def test_coarse_fine_joint_normalization_loss_and_discrete_decode():
    from samudra.skyline.model import CoarseFineHead

    torch.manual_seed(7)
    head = CoarseFineHead(16, 32, 4)
    x = torch.randn(2, 5, 16)
    joint = head(x)
    torch.testing.assert_close(joint.exp().sum(-1), torch.ones(2, 5))
    ids = torch.randint(32, (2, 5))
    coarse = head.coarse(x).log_softmax(-1)
    fine = head.fine(
        torch.nn.functional.silu(head.context(x).unsqueeze(-2) + head.token.weight)
    ).log_softmax(-1)
    expected = -coarse.gather(-1, (ids // 8).unsqueeze(-1)).squeeze(-1)
    selected = fine.gather(-2, (ids // 8)[..., None, None].expand(2, 5, 1, 8)).squeeze(
        -2
    )
    expected -= selected.gather(-1, (ids % 8).unsqueeze(-1)).squeeze(-1)
    loss = torch.nn.functional.cross_entropy(
        joint.flatten(0, 1), ids.flatten(), reduction="none"
    ).view(2, 5)
    torch.testing.assert_close(loss, expected)
    loss.mean().backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in head.parameters()
    )
    vocab = ChannelVocabulary([16.0] * 77, 32)
    channels = torch.zeros(2, 5, dtype=torch.long)
    mode, _ = vocab.predictions(joint, channels)
    torch.testing.assert_close(mode, vocab.centers[0, joint.argmax(-1)])


def test_coarse_fine_and_flat_share_backbone_initialization():
    models = []
    for bins, coarse in ((32, 0), (64, 0), (64, 8)):
        torch.manual_seed(71)
        models.append(
            PointTransformer(
                ModelConfig(
                    width=32,
                    layers=1,
                    heads=4,
                    bins=bins,
                    coarse_bins=coarse,
                    field_queries=True,
                )
            )
        )
    for name, value in models[0].state_dict().items():
        if not name.startswith("head."):
            for model in models[1:]:
                assert torch.equal(value, model.state_dict()[name]), name


def test_discrete_validation_has_no_continuous_readout():
    import numpy as np

    from samudra.skyline.comparison_eval import evaluate_comparison

    source, frame = synthetic_source("all_fields")
    source.area[:] = 1
    source.wet_area = source.wet.sum((1, 2))
    source.location_cdfs = [np.arange(1, len(p) + 1) / len(p) for p in source.locations]
    source.origins = {"validation": np.arange(2)}
    source.std = np.ones(80)
    source.read = lambda origin: frame
    model = PointTransformer(
        ModelConfig(
            width=16, layers=1, heads=4, bins=32, coarse_bins=4, field_queries=True
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
        model=dict(coarse_bins=4),
    )
    result = evaluate_comparison(
        model, source, config, torch.device("cpu"), 0, vocab=vocab
    )
    assert "probability_mean_full_grid_nmse" not in result
    for row in result["per_channel"]:
        assert not any("probability_mean" in key for key in row)
        assert row["greedy_full_grid_nmse"] >= 0
        for key in (
            "wet_cell_coarse_error_rate",
            "wet_cell_teacher_coarse_fine_error_rate",
            "wet_cell_leaf_error_rate",
        ):
            assert 0 <= row[key] <= 1.000001
