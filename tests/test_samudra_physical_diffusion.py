# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from samudra.experiments.samudra_physical_diffusion import SamudraPhysicalDiffusion


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = SamudraPhysicalDiffusion(
        SimpleNamespace(channels=4, history=3, surface=[0, 3]),
        widths=(8, 16, 32),
        sampling_steps=2,
    )
    surface = torch.randn(1, 5, 2, 33, 41)
    atmosphere = torch.randn(1, 5, 8, 33, 41)
    contexts = torch.randn(1, 5, 5, 33, 41)
    mask = torch.ones(4, 33, 41)
    mask[:, 0] = 0
    validity = torch.ones_like(surface)
    validity[:, 1, 0, 4, 4] = 0
    return model, (surface, atmosphere, contexts, mask, validity)


def rng(seed=12):
    return torch.Generator().manual_seed(seed)


def test_physical_members_anchors_mask_and_future_observation_exclusion():
    model, args = setup()
    model.eval()
    with torch.no_grad():
        prediction, initial = model.forecast(*args, generator=rng(), members=2)
        surface, atmosphere, contexts, mask, validity = args
        changed = surface.clone()
        changed[:, 3:] = 999
        repeated, _ = model.forecast(
            changed, atmosphere, contexts, mask, validity, generator=rng(), members=2
        )
    assert prediction.shape == (2, 1, 2, 4, 33, 41)
    assert initial.shape == (2, 1, 2, 4, 33, 41)
    torch.testing.assert_close(prediction, repeated, rtol=0, atol=0)
    assert not torch.equal(prediction[0], prediction[1])
    assert torch.count_nonzero(prediction[..., 0, :]) == 0
    known = surface[:, 1:3]
    valid = validity[:, 1:3].bool() & mask[[0, 3]].bool()
    for member in initial:
        torch.testing.assert_close(member[:, :, [0, 3]][valid], known[valid])


def test_history_shift_and_rng_resume():
    model, (_, _, contexts, mask, _) = setup()
    model.eval()
    states = torch.randn(1, 2, 4, 33, 41) * mask
    forcing = torch.randn(1, 8, 33, 41)
    generator = rng()
    with torch.no_grad():
        next_state = model.advance(states, forcing, contexts[:, 0], mask, generator)
        saved = generator.get_state()
        expected = model.advance(next_state, forcing, contexts[:, 1], mask, generator)
        restored = rng(999)
        restored.set_state(saved)
        actual = model.advance(next_state, forcing, contexts[:, 1], mask, restored)
    torch.testing.assert_close(next_state[:, 0], states[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_observation_sampler_backpropagates_both_samudra_diffusers():
    model, args = setup()
    predictions, _ = model.forecast(*args, generator=rng(), members=1)
    predictions.square().mean().backward()
    for module in (model.initializer.net, model.processor.net):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_om4_single_noise_level_losses_and_invalid_surface_conditioning():
    model, (surface, atmosphere, contexts, mask, validity) = setup()
    past = atmosphere[:, :3]
    conditioning, _, anchor = model.encode_native(
        surface[:, :3], past, contexts[:, 2], mask, validity[:, :3]
    )
    changed = surface[:, :3].clone()
    changed[~validity[:, :3].bool()] = 999
    repeat, _, _ = model.encode_native(
        changed, past, contexts[:, 2], mask, validity[:, :3]
    )
    torch.testing.assert_close(conditioning, repeat)
    assert not anchor.reshape(1, 2, 4, 33, 41)[0, 0, 0, 4, 4]
    truth = torch.randn(1, 2, 4, 33, 41) * mask
    loss = model.pretraining_loss(
        surface[:, :3], past, contexts[:, 2], truth, mask, mask, rng()
    )
    loss = loss + model.transition_loss(
        truth, past[:, -1], contexts[:, 2], truth[:, -1], mask, mask, rng()
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_diffusers_use_instance_normalization_without_batch_statistics():
    model, _ = setup()
    for diffuser in (model.initializer, model.processor):
        assert any(isinstance(m, nn.InstanceNorm2d) for m in diffuser.modules())
        assert not any(isinstance(m, nn.BatchNorm2d) for m in diffuser.modules())


def test_all_eight_era5_channels_preserved_without_projection():
    model, (surface, atmosphere, contexts, mask, validity) = setup()
    past = atmosphere[:, :3]
    condition, _, _ = model.encode_native(
        surface[:, :3], past, contexts[:, 2], mask, validity[:, :3]
    )
    torch.testing.assert_close(condition[:, -24:], past.flatten(1, 2))
    states = torch.randn(1, 2, 4, 33, 41)
    condition = model.processor_condition(states, atmosphere[:, 3], contexts[:, 3])
    torch.testing.assert_close(condition[:, 8:16], atmosphere[:, 3])
    assert not hasattr(model, "adapter")


def test_vectorized_members_match_serial_with_identical_per_example_noise(monkeypatch):
    from samudra.experiments import joint_diffusion
    from samudra.experiments.diffusion_noise import diffusion_noise

    model, args = setup()
    model.eval()
    surface, atmosphere, contexts, mask, validity = args
    # Distinct examples catch accidentally mixing the member and batch axes.
    surface = torch.cat((surface, surface + 0.4), 0)
    atmosphere = torch.cat((atmosphere, atmosphere - 0.3), 0)
    contexts = torch.cat((contexts, contexts + 0.2), 0)
    validity = validity.expand(2, *validity.shape[1:])
    args = surface, atmosphere, contexts, mask, validity
    members, batch = 3, 2
    noise = []
    for stage in range(3):
        fields = 8 if stage == 0 else 4
        wet = mask.repeat(2, 1, 1) if stage == 0 else mask
        noise.append(
            diffusion_noise(
                (members * batch, fields, 33, 41), wet, rng(80 + stage), correlation=0.5
            )
        )
    assert not torch.equal(noise[0][:batch], noise[0][batch : 2 * batch])

    def run(vectorized):
        model.vectorize_members = vectorized
        calls: list = []

        def common_noise(shape, wet, generator, correlation, paired):
            assert correlation == 0.5 and not paired
            index = len(calls)
            calls.append(shape)
            if vectorized:
                value = noise[index]
            else:
                member, stage = divmod(index, 3)
                value = noise[stage][member * batch : (member + 1) * batch]
            assert tuple(value.shape) == tuple(shape)
            return value.clone()

        monkeypatch.setattr(joint_diffusion, "diffusion_noise", common_noise)
        with torch.no_grad():
            result = model.forecast(*args, generator=rng(), members=members)
        assert len(calls) == (3 if vectorized else 9)
        return result

    vectorized = run(True)
    serial = run(False)
    for actual, expected in zip(vectorized, serial, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_checkpoint_toggle_preserves_sample_and_parameter_gradients():
    model, args = setup()
    outcomes = []
    for enabled in (True, False):
        model.zero_grad(set_to_none=True)
        model.checkpoint_denoiser = enabled
        prediction, _ = model.forecast(*args, generator=rng(), members=2)
        prediction.square().mean().backward()
        outcomes.append(
            (prediction.detach(), [p.grad.clone() for p in model.parameters()])
        )
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0], rtol=0, atol=0)
    for actual, expected in zip(outcomes[0][1], outcomes[1][1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_compile_cache_is_training_only_and_safe_for_ema_and_state_dict(monkeypatch):
    import copy

    model, args = setup()
    original_keys = tuple(model.state_dict())
    calls = []

    def fake_compile(function, **kwargs):
        def compiled(value):
            calls.append(kwargs)
            return function(value)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    model.enable_compile()
    assert tuple(model.state_dict()) == original_keys
    copied = copy.deepcopy(model)
    assert not hasattr(copied.initializer, "_compiled_net")
    assert not hasattr(copied.processor, "_compiled_net")
    copied.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        model.forecast(*args, generator=rng())
    assert calls
    calls.clear()
    model.eval()
    with torch.no_grad():
        prediction = model.forecast(*args, generator=rng())[0]
        expected = copied.eval().forecast(*args, generator=rng())[0]
    assert not calls
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
