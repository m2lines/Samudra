# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.monthly_batching import batched_loss, groups, prepare_model
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_data import model_loss


def test_preparation_preserves_original_draw_order_and_single_loss():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    model = MonthlyLatentOcean(8, 8, (16, 32), history=19, encoder_width=8).eval()
    generator = torch.Generator().manual_seed(42)
    batch = dict(
        fields=torch.randn(1, 21, 77, 16, 32),
        forcing=torch.randn(1, 21, 3, 16, 32),
        contexts=torch.randn(1, 21, 5, 16, 32),
        mask=torch.rand(77, 16, 32) > 0.2,
        weights=torch.tensor([0.4, 0.6]),
        duration=10.0,
    )
    state = generator.get_state()
    expected = model_loss(model, batch, "surface", generator, torch.ones(16, 32))
    expected_rng = generator.get_state()
    generator.set_state(state)
    item = prepare_model(batch, generator)
    assert torch.equal(expected_rng, generator.get_state())
    actual, _, _ = batched_loss(model, [item], "om4", torch.ones(16, 32))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    expected.backward()
    expected_grad = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    actual, _, _ = batched_loss(model, [item], "om4", torch.ones(16, 32))
    actual.backward()
    for name, parameter in model.named_parameters():
        if name in expected_grad:
            torch.testing.assert_close(
                parameter.grad, expected_grad[name], rtol=1e-5, atol=1e-7
            )


def test_grouping_preserves_all_examples_without_cross_update_resampling():
    items = []
    for index, length in enumerate((7, 6, 7, 6, 8, 7, 6, 8)):
        items.append(
            dict(
                index=index,
                fields=torch.empty(1, 19 + length, 1),
                weights=torch.ones(length),
            )
        )
    grouped = list(groups(items, 2))
    assert sorted(x["index"] for group in grouped for x in group) == list(range(8))
    assert all(len(group) <= 2 for group in grouped)
    assert all(len({len(x["weights"]) for x in group}) == 1 for group in grouped)
    assert [len(g) for g in grouped] == [2, 1, 2, 1, 2]


def test_observation_preparation_preserves_rng_and_masked_objective():
    from types import SimpleNamespace

    from samudra.experiments.monthly_batching import prepare_observation
    from samudra.experiments.monthly_pilot_data import observation_loss

    torch.set_num_threads(1)
    torch.manual_seed(15)
    model = MonthlyLatentOcean(8, 8, (16, 32), history=19, encoder_width=8).eval()
    data = SimpleNamespace(
        mean=torch.zeros(1, 1, 77, 1, 1),
        std=torch.ones(1, 1, 77, 1, 1),
        ts_indices=list(range(38)),
        ts_mask=torch.ones(38, 16, 32, dtype=torch.bool),
        area=torch.ones(16, 32),
    )
    sample = dict(
        surface=torch.randn(1, 21, 2, 16, 32),
        validity=torch.rand(1, 21, 2, 16, 32) > 0.2,
        interior=torch.randn(1, 38, 16, 32),
        atmosphere=torch.randn(1, 21, 8, 16, 32),
        contexts=torch.randn(1, 21, 5, 16, 32),
        month_weights=torch.tensor([0.4, 0.6]),
        name="2001-02",
    )
    generator = torch.Generator().manual_seed(17)
    state = generator.get_state()
    expected, _ = observation_loss(model, data, sample, "surface", generator)
    expected_rng = generator.get_state()
    generator.set_state(state)
    item = prepare_observation(data, sample, generator)
    assert torch.equal(expected_rng, generator.get_state())
    actual, _, _ = batched_loss(model, [item], "obs", data.area)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_ddp_local_weighting_is_equal_two_examples_with_unequal_lengths():
    from samudra.experiments.monthly_distributed import UpdateLoss

    torch.set_num_threads(1)
    torch.manual_seed(33)
    model = MonthlyLatentOcean(8, 8, (16, 32), history=19, encoder_width=8).eval()
    items = []
    for length in (2, 3):
        batch = dict(
            fields=torch.randn(1, 19 + length, 77, 16, 32),
            forcing=torch.randn(1, 19 + length, 3, 16, 32),
            contexts=torch.randn(1, 19 + length, 5, 16, 32),
            mask=torch.ones(77, 16, 32, dtype=torch.bool),
            weights=torch.ones(length) / length,
            duration=float(5 * length),
        )
        items.append(prepare_model(batch, torch.Generator().manual_seed(length)))
    area = torch.ones(16, 32)
    expected = sum(batched_loss(model, [item], "om4", area)[0] for item in items) / 2
    actual, _, _ = UpdateLoss(model)(items, "om4", area)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_training_compilation_hook_cannot_change_eval_denoising():
    from samudra.experiments.monthly_latent import WindowDecoder

    torch.set_num_threads(1)
    decoder = WindowDecoder(8, 8)
    noisy = torch.randn(1, 77, 16, 32)
    sigma = torch.ones(1)
    latent = torch.randn(1, 2, 8, 4, 8)
    support = torch.ones_like(noisy, dtype=torch.bool)
    condition = decoder.condition(latent, support, 5.0, 0.0)
    decoder._compiled_training = lambda *args: torch.zeros_like(noisy)
    decoder.train()
    assert not decoder.training_denoise(noisy, sigma, condition, support).any()
    decoder.eval()
    expected = decoder.core(noisy, sigma, condition, support)
    torch.testing.assert_close(
        decoder.training_denoise(noisy, sigma, condition, support),
        expected,
        rtol=0,
        atol=0,
    )
