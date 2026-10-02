# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.diffusion_spatial import spatial_crps, spatial_mse
from samudra.experiments.task_schedule import TaskSchedule, sample_indices


def test_matched_exposure_and_partitioned_samples():
    schedule = TaskSchedule(8000, 8000, "mixed")
    assert schedule.counts(6949) == {"om4": 4949, "observation": 2000}
    assert schedule.counts(16000) == {"om4": 8000, "observation": 8000}
    for size in (243, 2800):
        ids = sample_indices(size, 1729, 456, 8)
        sharded = {i: ids[i] for rank in range(4) for i in range(rank, 8, 4)}
        assert [sharded[i] for i in range(8)] == ids


def test_spatial_loss_has_no_latitude_wrap_and_excludes_dry_edges():
    x = torch.zeros(1, 1, 3, 4)
    x[..., -1, :] = 1
    weight = torch.ones_like(x)
    # Four real meridional differences out of eight; no polar closing edge.
    assert spatial_mse(x, torch.zeros_like(x), weight).item() == 0.5
    weight[..., -1, :] = 0
    assert spatial_mse(x, torch.zeros_like(x), weight).item() == 0
    x.zero_()
    x[..., 0] = 1
    weight.fill_(1)
    # Both longitude edges contribute, including the periodic closing edge.
    assert spatial_mse(x, torch.zeros_like(x), weight).item() == 0.5


def test_increment_crps_distinguishes_equal_pixel_marginals():
    truth = torch.zeros(1, 1, 3, 4)
    coherent = torch.stack([torch.ones_like(truth), -torch.ones_like(truth)])
    grain = coherent.clone()
    grain[..., ::2] *= -1
    mask = torch.ones_like(truth)
    pixel_a, _ = spatial_crps(coherent, truth, mask, mask, 1, coefficient=0)
    pixel_b, _ = spatial_crps(grain, truth, mask, mask, 1, coefficient=0)
    torch.testing.assert_close(pixel_a, pixel_b)
    # With two perfectly antithetic members fair CRPS is zero for both here;
    # use four draws with identical per-cell empirical marginal distributions.
    coherent = torch.cat([coherent, coherent])
    grain = torch.cat([grain, grain])
    a, _ = spatial_crps(coherent, truth, mask, mask, 1)
    b, _ = spatial_crps(grain, truth, mask, mask, 1)
    assert bool((b > a).all())


def test_missing_targets_do_not_contaminate_spatial_gradients():
    target = torch.randn(1, 2, 4, 6)
    target[..., 0, 0] = float("nan")
    members = torch.randn(3, *target.shape, requires_grad=True)
    value, valid = spatial_crps(
        members, target, torch.ones_like(target), torch.ones(4, 1), 1
    )
    value[valid].mean().backward()
    assert members.grad is not None
    assert torch.isfinite(members.grad).all()
    assert not bool(members.grad[..., 0, 0].count_nonzero())


def test_accumulation_matches_partitioned_optimizer_step_after_resume():
    # CPU analogue of the synchronous rank reduction, including AdamW resume.
    import copy

    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    x, y = torch.randn(8, 3), torch.randn(8, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    other = torch.optim.AdamW(reference.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        ((model(x) - y).square().mean()).backward()
        optimizer.step()
        other.zero_grad(set_to_none=True)
        summed = [torch.zeros_like(p) for p in reference.parameters()]
        for rank in range(4):
            reference.zero_grad(set_to_none=True)
            for micro in range(rank, 8, 4):
                (
                    (reference(x[micro : micro + 1]) - y[micro : micro + 1])
                    .square()
                    .mean()
                    / 8
                ).backward()
            for total, parameter in zip(summed, reference.parameters(), strict=True):
                assert parameter.grad is not None
                total.add_(parameter.grad)
        for total, parameter in zip(summed, reference.parameters(), strict=True):
            parameter.grad = total
        other.step()
        state = copy.deepcopy(other.state_dict())
        other = torch.optim.AdamW(reference.parameters(), lr=1e-4)
        other.load_state_dict(state)
        for a, b in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
