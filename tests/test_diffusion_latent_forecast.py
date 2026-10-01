# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_latent_forecast import LatentOceanForecast


def model_for_test(stochastic):
    initializer = SimpleNamespace(
        channels=4, history=3, surface=[0, 3], net=nn.Conv2d(26, 8, 1)
    )
    return LatentOceanForecast(
        initializer,
        nn.Conv2d(8, 3, 1),
        stochastic=stochastic,
        latent_width=8,
        latent_shape=(4, 6),
        conditioning_shape=(8, 12),
        processor_depth=2,
        decoder_width=8,
        sampling_steps=3,
    )


@pytest.mark.parametrize("stochastic", [False, True])
def test_interior_supervision_trains_latent_path_without_teacher_forcing(stochastic):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = model_for_test(stochastic)
    surface, past = torch.randn(1, 3, 2, 8, 12), torch.randn(1, 3, 3, 8, 12)
    context, truth = torch.randn(1, 5, 8, 12), torch.randn(1, 2, 4, 8, 12)
    truth[:, :, [0, 3]] = surface[:, -2:]
    forcing, contexts = torch.randn(1, 3, 3, 8, 12), torch.randn(1, 3, 5, 8, 12)
    targets, mask = torch.randn(1, 3, 4, 8, 12), torch.ones(4, 8, 12)
    states = []
    hook = model.processor.register_forward_hook(
        lambda module, inputs, result: states.append(result.detach().clone())
    )
    loss = model.pretraining_loss(
        surface,
        past,
        context,
        truth,
        forcing,
        contexts,
        targets,
        mask,
        mask,
        torch.Generator().manual_seed(41),
    )
    original_states = states[:3]
    loss.backward()
    for module in (
        model.encoder.encoder,
        model.encoder.projection,
        model.processor.stem,
        model.processor.head,
        model.decoder.head,
    ):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    assert (
        model.adapter.weight.grad is None
    )  # Native OM4 forcings bypass ERA5 adaptation.
    states.clear()
    changed_truth = truth.clone()
    changed_truth[:, :, 1:3] += 100
    with torch.no_grad():
        changed = model.pretraining_loss(
            surface,
            past,
            context,
            changed_truth,
            forcing,
            contexts,
            targets + 100,
            mask,
            mask,
            torch.Generator().manual_seed(41),
        )
    assert not torch.equal(changed, loss)
    for original, altered in zip(original_states, states, strict=True):
        torch.testing.assert_close(original, altered, rtol=0, atol=0)
    hook.remove()


@pytest.mark.parametrize("stochastic", [False, True])
def test_forecast_keeps_decoding_noise_and_future_observations_out_of_recurrence(
    stochastic,
):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = model_for_test(stochastic).eval()
    surface = torch.randn(1, 5, 2, 8, 12)
    atmosphere, contexts = torch.randn(1, 5, 8, 8, 12), torch.randn(1, 5, 5, 8, 12)
    mask, valid = torch.ones(4, 8, 12), torch.ones_like(surface)
    valid[:, 2, 0, 1, 1] = 0
    states = []
    hook = model.processor.register_forward_hook(
        lambda module, inputs, result: states.append(result.detach().clone())
    )
    members = 2 if stochastic else 1

    def run(seed):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            return model.forecast(
                surface,
                atmosphere,
                contexts,
                mask,
                valid,
                generator=torch.Generator().manual_seed(seed),
                members=members,
            )

    forecast, initial = run(12)
    assert forecast.shape == (members, 1, 2, 4, 8, 12)
    assert initial.shape == (members, 1, 2, 4, 8, 12)
    reference_states = states.copy()
    assert len(reference_states) == 2  # One latent path, independent of member count.
    known = valid[:, 1:3].bool().expand(members, -1, -1, -1, -1, -1)
    torch.testing.assert_close(
        initial[:, :, :, [0, 3]][known],
        surface[:, 1:3].expand(members, -1, -1, -1, -1, -1)[known],
        rtol=0,
        atol=0,
    )
    surface[:, 3:] = 1000
    states.clear()
    same, _ = run(12)
    torch.testing.assert_close(forecast, same, rtol=0, atol=0)
    states.clear()
    other, _ = run(13)
    for original, altered in zip(reference_states, states, strict=True):
        torch.testing.assert_close(original, altered, rtol=0, atol=0)
    if stochastic:
        assert not torch.equal(forecast, other)
    hook.remove()


def test_checkpointed_denoising_preserves_explicit_rng_and_gradients():
    """Recomputing the decoder must not draw a second corruption from its generator."""
    from copy import deepcopy

    from samudra.experiments.joint_diffusion import JointInteriorDecoder, denoising_loss

    torch.set_num_threads(1)
    torch.manual_seed(42)
    ordinary = JointInteriorDecoder(4, 4, width=8)
    recomputed = deepcopy(ordinary)
    target = torch.randn(1, 4, 8, 12)
    features = torch.randn_like(target)
    mask = torch.ones(4, 8, 12)
    outcomes = []
    for model, enabled in ((ordinary, False), (recomputed, True)):
        latent = features.clone().requires_grad_()
        generator = torch.Generator().manual_seed(123)
        loss = denoising_loss(
            model,
            latent,
            target,
            mask,
            mask,
            generator,
            checkpoint_denoiser=enabled,
        )
        loss.backward()
        outcomes.append((loss.detach(), latent.grad, generator.get_state()))
    for expected, actual in zip(outcomes[0], outcomes[1], strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for expected, actual in zip(
        ordinary.parameters(), recomputed.parameters(), strict=True
    ):
        torch.testing.assert_close(expected.grad, actual.grad, rtol=0, atol=0)


@pytest.mark.parametrize("stochastic", [False, True])
def test_omitting_unused_initial_readout_preserves_forecast_gradients_and_rng(
    stochastic,
):
    from copy import deepcopy

    torch.set_num_threads(1)
    torch.manual_seed(99)
    original = model_for_test(stochastic).train()
    optimized = deepcopy(original)
    surface = torch.randn(1, 5, 2, 8, 12)
    atmosphere = torch.randn(1, 5, 8, 8, 12)
    contexts = torch.randn(1, 5, 5, 8, 12)
    mask, valid = torch.ones(4, 8, 12), torch.ones_like(surface)
    results = []
    for model, include in ((original, True), (optimized, False)):
        rng = torch.Generator().manual_seed(13)
        forecast, initial = model.forecast(
            surface,
            atmosphere,
            contexts,
            mask,
            valid,
            generator=rng,
            members=2 if stochastic else 1,
            decode_initial=include,
        )
        assert (initial is None) == (not include)
        forecast.square().mean().backward()
        results.append((forecast.detach(), rng.get_state()))
    for a, b in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(original.parameters(), optimized.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_compiled_decoder_keeps_portable_checkpoint_keys(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(73)
    model = model_for_test(True).eval()
    original_keys = set(model.state_dict())
    inputs = (
        torch.randn(1, 8, 8, 12),
        torch.ones(1),
        torch.randn(1, 16, 4, 6),
        torch.ones(8, 8, 12),
    )
    with torch.no_grad():
        expected = model.decoder(*inputs)
    # CPU eager backend exercises the module compilation wrapper and serialization;
    # Inductor numerical differences are measured by the real GPU benchmarks.
    model.decoder.compile(backend="eager")
    with torch.no_grad():
        actual = model.decoder(*inputs)
    assert set(model.state_dict()) == original_keys
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    restored = model_for_test(True).eval()
    restored.load_state_dict(
        torch.load(tmp_path / "weights.pt", weights_only=True), strict=True
    )
    with torch.no_grad():
        reloaded = restored.decoder(*inputs)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(expected, reloaded, rtol=0, atol=0)


def test_selected_readouts_preserve_full_forecast_members_and_rng():
    from samudra.experiments.diffusion_endpoint_maps import selected_readouts

    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = model_for_test(True).eval()
    surface = torch.randn(1, 3, 2, 8, 12)
    atmosphere = torch.randn(1, 7, 8, 8, 12)
    contexts = torch.randn(1, 7, 5, 8, 12)
    mask = torch.ones(4, 8, 12)
    validity = torch.ones_like(surface, dtype=torch.bool)
    full_rng = torch.Generator().manual_seed(23)
    selected_rng = torch.Generator().manual_seed(23)
    with torch.no_grad():
        full, _ = model.forecast(
            surface, atmosphere, contexts, mask, validity, generator=full_rng, members=3
        )
        adapted = model.adapt(atmosphere)
        initial, known, anchor = model.encode_native(
            surface, adapted[:, :3], contexts[:, 2], mask, validity
        )
        states = model.processor.rollout(initial, adapted[:, 3:], contexts[:, 3:])
        result = selected_readouts(
            model, states, mask, known, anchor, selected_rng, members=3, leads=(2, 4)
        )
    torch.testing.assert_close(result, full[:, :, [1, 3]], rtol=0, atol=0)
    assert torch.equal(full_rng.get_state(), selected_rng.get_state())


def test_native_endpoint_chunks_preserve_history_and_order_without_future_inputs():
    from samudra.experiments.diffusion_endpoint_maps import native_sequence

    class Wave:
        def model_sample(self, dataset, ids):
            start = ids[0]
            history = tuple(torch.tensor(start + j) for j in range(4))
            forcing = torch.arange(start, start + dataset.steps)[None]
            labels = forcing + 100
            return (*history, forcing, labels)

    result = native_sequence(Wave(), SimpleNamespace(steps=6), 17, steps=73)
    assert [x.item() for x in result[:4]] == [17, 18, 19, 20]
    torch.testing.assert_close(result[4], torch.arange(17, 90)[None])
    torch.testing.assert_close(result[5], torch.arange(117, 190)[None])
