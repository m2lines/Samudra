# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Boundary and gradient safeguards for the continuation pilot."""

import torch

from samudra.experiments.diffusion_spatial import (
    directional_pairs,
    multiscale_mse,
    multiscale_pairs,
    spatial_crps,
)


def test_multiscale_ignores_constant_bias_but_detects_diagonal_artifact():
    target = torch.randn(1, 2, 20, 24)
    weights = torch.ones_like(target)
    torch.testing.assert_close(
        multiscale_mse(target + 3, target, weights), torch.zeros(1), atol=1e-10, rtol=0
    )
    y, x = torch.meshgrid(torch.arange(20), torch.arange(24), indexing="ij")
    prediction = (target + torch.sin((x + y) * 2.0)).requires_grad_()
    loss = multiscale_mse(prediction, target, weights).sum()
    assert loss > 0
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


def test_lag_cannot_bridge_missing_cells_and_latitude_never_wraps():
    valid = torch.ones(1, 2, 20, 24, dtype=torch.bool)
    valid[..., 5] = False
    entries = {(dy, dx): support for dy, dx, _, support in multiscale_pairs(valid)}
    assert not entries[(0, 4)][..., 2].any()  # Endpoints wet, land in between.
    assert entries[(0, 4)][..., 23].all()  # Longitude wraps.
    assert entries[(8, 0)].shape[-2] == 12  # Latitude does not wrap.
    a, b = directional_pairs(torch.arange(24).expand(20, 24), 1, -1)
    assert a[0, 0] == 0 and b[0, 0] == 23


def test_multiscale_crps_missing_values_have_finite_gradients():
    target = torch.randn(1, 2, 20, 24)
    target[..., 6:10] = float("nan")
    members = torch.randn(2, *target.shape, requires_grad=True)
    value, present = spatial_crps(
        members,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.ones(20, 1),
        1.0,
        multiscale=0.5,
    )
    value[present].mean().backward()
    assert members.grad is not None
    assert torch.isfinite(value).all() and torch.isfinite(members.grad).all()
    assert members.grad[..., 6:10].count_nonzero() == 0


def test_replay_targets_follow_carried_latent_and_reuse_detaches(monkeypatch):
    from types import SimpleNamespace

    import samudra.experiments.diffusion_interventions as pilot

    class Processor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rate = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, state, forcing, context, task):
            return state + self.rate

        def rollout(self, state, forcing, contexts, task):
            values = [state]
            for _ in range(forcing.shape[1]):
                values.append(self(values[-1], None, None, task))
            return values

    class Wave:
        source = object()
        mask = torch.ones(1, 1, 1)
        weights = mask
        device = "cpu"
        trainset = SimpleNamespace(steps=6)

        def dataset(self, source, steps):
            class Dataset:
                steps: int

                def __len__(self):
                    return 100

            dataset = Dataset()
            dataset.steps = steps
            return dataset

        def model_sample(self, dataset, ids):
            origin = ids[0]
            surface = torch.ones(1, 19, 1, 1, 1)
            truth = torch.tensor([origin - 1, origin], dtype=torch.float32).reshape(
                1, 2, 1, 1, 1
            )
            labels = (
                torch.arange(origin + 1, origin + dataset.steps + 1)
                .reshape(1, -1, 1, 1, 1)
                .float()
            )
            return (
                surface,
                surface,
                torch.zeros(1, 5, 1, 1),
                truth,
                torch.zeros(1, dataset.steps, 3, 1, 1),
                labels,
            )

    wave = Wave()
    model = SimpleNamespace(
        surface=[0],
        processor=Processor(),
        decoder=None,
        training=True,
        encode_native=lambda *a, **k: (torch.zeros(1, 2, 1, 1, 1), None, None),
    )
    seen = []

    def denoise(decoder, latent, target, *a, **k):
        seen.append(target.detach().clone())
        return latent.square().mean()

    monkeypatch.setattr(pilot, "denoising_loss", denoise)
    monkeypatch.setattr(pilot, "structured_visibility", lambda valid, seed: valid)
    replay = pilot.LatentReplay(wave)
    coverage = torch.ones(1, 19, 1, 1, 1, dtype=torch.bool)
    replay.loss(model, 0, 0, coverage, 99).backward()
    entry = replay.slots[0]
    assert entry["lead"] == 12 and entry["latent"].grad_fn is None
    origin = entry["index"]
    assert seen[0].flatten().tolist() == [origin + 6, origin + 7]
    assert seen[-1].flatten().tolist() == [origin + 11, origin + 12]
    replay.loss(model, 0, 1, coverage, 100).backward()
    assert replay.slots[0]["index"] == origin and replay.slots[0]["lead"] == 18
    assert not replay.last_info["refreshed"]
    assert seen[6].flatten().tolist() == [origin + 12, origin + 13]
    assert model.processor.rate.grad > 0


def test_alternative_features_preserve_bias_and_detect_diagonal_curvature():
    from samudra.experiments.diffusion_structure_losses import structure_mse

    y, x = torch.meshgrid(torch.arange(20), torch.arange(24), indexing="ij")
    truth = torch.zeros(1, 2, 20, 24)
    weights = torch.ones_like(truth)
    assert structure_mse(truth + 2, truth, weights, "curvature") == 0
    assert structure_mse(truth + 2, truth, weights, "block") > 0
    noisy = (truth + torch.sin((x + y) * 2.0)).requires_grad_()
    loss = structure_mse(noisy, truth, weights, "curvature").sum()
    loss.backward()
    assert loss > 0 and noisy.grad is not None and torch.isfinite(noisy.grad).all()


def test_alternative_feature_crps_masks_nan_and_poles():
    from samudra.experiments.diffusion_structure_losses import features, structure_crps

    for kind in ("curvature", "block"):
        target = torch.randn(1, 2, 20, 24)
        target[..., 5] = float("nan")
        members = torch.randn(2, *target.shape, requires_grad=True)
        values = structure_crps(
            members,
            target,
            torch.ones_like(target, dtype=torch.bool),
            torch.ones(20, 1),
            1.0,
            kind,
        )
        values.sum().backward()
        assert torch.isfinite(values).all() and members.grad is not None
        assert torch.isfinite(members.grad).all()
        assert members.grad[..., 5].count_nonzero() == 0
        if kind == "block":
            _, _, weights = next(
                features(members, target, torch.ones_like(target), kind)
            )
            assert not weights[..., 0, :].any() and not weights[..., -1, :].any()
            assert not weights[..., 4:7].any()


def test_latent_bound_penalizes_growth_without_shrinking_normal_states():
    from samudra.experiments.diffusion_structure_losses import excess_latent_loss

    reference = torch.ones(1, 2, 3, 4, 5, requires_grad=True)
    assert excess_latent_loss([reference * 1.5], reference) == 0
    state = torch.full_like(reference, 3, requires_grad=True)
    excess_latent_loss([state], reference).backward()
    assert reference.grad is None
    assert state.grad is not None and (state.grad > 0).all()


def test_structure_mse_all_unsupported_features_are_zero():
    from samudra.experiments.diffusion_structure_losses import structure_mse

    value = torch.randn(1, 2, 20, 24, requires_grad=True)
    weights = torch.zeros_like(value)
    weights[..., 5, 5] = 1
    for kind in ("curvature", "block"):
        loss = structure_mse(value, value.detach(), weights, kind)
        assert loss == 0
        loss.sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
