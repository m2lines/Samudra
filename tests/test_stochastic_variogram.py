# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for finite-ensemble spatial dependence supervision."""

import numpy as np
import pytest
import torch

from samudra.experiments.stochastic_variogram import variogram_score


def _oracle(members, target, mask, area, lags):
    """Independent scalar-loop oracle, using the off-diagonal member product."""
    members, target, mask, area = (
        tensor.detach().numpy() for tensor in (members, target, mask, area)
    )
    count, batch, channels, height, width = members.shape
    lag_results = []
    for lag in lags:
        directions = []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            bc_results = []
            for b in range(batch):
                for c in range(channels):
                    pairs, weights = [], []
                    for y in range(height - dy * lag):
                        for x in range(width):
                            path = [
                                (y + dy * k, (x + dx * k) % width)
                                for k in range(lag + 1)
                            ]
                            if not all(
                                mask[c, iy, ix] and np.isfinite(target[b, c, iy, ix])
                                for iy, ix in path
                            ):
                                continue
                            iy, ix = path[-1]
                            t = np.abs(
                                members[:, b, c, y, x] - members[:, b, c, iy, ix]
                            ).astype(float)
                            a = float(abs(target[b, c, y, x] - target[b, c, iy, ix]))
                            plugin = (t.mean() - a) ** 2
                            fair = (
                                a**2
                                - 2 * a * t.mean()
                                + sum(
                                    t[i] * t[j]
                                    for i in range(count)
                                    for j in range(count)
                                    if i != j
                                )
                                / (count * (count - 1))
                                if count > 1
                                else plugin
                            )
                            pairs.append((fair, plugin, plugin - fair))
                            weights.append(0.5 * (area[y, x] + area[iy, ix]))
                    if pairs:
                        bc_results.append(np.average(pairs, axis=0, weights=weights))
            if bc_results:
                directions.append(np.mean(bc_results, axis=0))
        lag_results.append(np.mean(directions, axis=0))
    return np.mean(lag_results, axis=0), np.array(lag_results)


def _inputs(count=3, batch=2, channels=2, height=4, width=5):
    generator = torch.Generator().manual_seed(281)
    members = torch.randn(count, batch, channels, height, width, generator=generator)
    target = torch.randn(batch, channels, height, width, generator=generator)
    mask = torch.ones(channels, height, width, dtype=torch.bool)
    area = torch.linspace(0.1, 1.0, height)[:, None].expand(height, width)
    return members, target, mask, area


def test_matches_explicit_pair_product_oracle_and_area_aggregation():
    members, target, mask, area = _inputs()
    mask[0, 1, 2] = False
    mask[1, 2, 3] = False
    target[0, 1, 0, 0] = float("nan")
    actual = variogram_score(members, target, mask, area, lags=(1, 2))
    expected, expected_lags = _oracle(members, target, mask, area, (1, 2))
    for index, key in enumerate(("fair", "plugin", "correction")):
        np.testing.assert_allclose(actual[key], expected[index], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(
            actual[f"{key}_by_lag"], expected_lags[:, index], rtol=2e-6, atol=1e-7
        )


def test_masked_nans_and_zero_land_gradient_and_finite_wet_gradients():
    members, target, mask, area = _inputs()
    mask[:, 1, 1:4] = False
    members[:, :, ~mask] = float("nan")
    target[:, ~mask] = float("nan")
    members.requires_grad_()
    result = variogram_score(members, target, mask, area, lags=(1, 2))
    result["fair"].backward()
    assert torch.isfinite(result["fair"])
    assert torch.isfinite(members.grad).all()
    assert members.grad[:, :, ~mask].count_nonzero() == 0
    assert members.grad[:, :, mask].abs().sum() > 0


def test_zero_increments_have_finite_gradients_and_deterministic_score():
    members, target, mask, area = _inputs()
    members = torch.zeros_like(members, requires_grad=True)
    result = variogram_score(members, target, mask, area, lags=(1,))
    result["fair"].backward()
    assert torch.isfinite(members.grad).all()
    assert result["correction"] == 0
    torch.testing.assert_close(result["fair"], result["plugin"])
    single = variogram_score(members[:1], target, mask, area, lags=(1,))
    torch.testing.assert_close(single["fair"], result["fair"])
    perfect = variogram_score(
        target[None].repeat(3, 1, 1, 1, 1), target, mask, area, lags=(1,)
    )
    torch.testing.assert_close(perfect["fair"], torch.tensor(0.0), atol=1e-12, rtol=0)


def test_fair_score_can_be_negative():
    target = (torch.arange(6) % 2).float()[None, None, None].expand(1, 1, 4, 6)
    members = torch.stack((torch.zeros_like(target), 2 * target))
    result = variogram_score(
        members,
        target,
        torch.ones(1, 4, 6, dtype=torch.bool),
        torch.ones(4, 6),
        lags=(1,),
    )
    assert result["plugin"] == 0
    assert result["correction"] > 0
    assert result["fair"] < 0


def test_member_identity_matters_spatially_and_global_permutation_does_not():
    members, target, mask, area = _inputs()
    original = variogram_score(members, target, mask, area, lags=(1,))
    permuted = variogram_score(members[[2, 0, 1]], target, mask, area, lags=(1,))
    torch.testing.assert_close(original["fair"], permuted["fair"])
    members = members.clone()
    members[..., 0::2] = members[[2, 0, 1], ..., 0::2]
    local = variogram_score(members, target, mask, area, lags=(1,))
    assert not torch.isclose(original["fair"], local["fair"])


def test_longitude_seam_is_scored_and_latitude_does_not_wrap():
    mask = torch.zeros(1, 4, 6, dtype=torch.bool)
    mask[0, 1, [0, 5]] = True
    target = torch.zeros(1, 1, 4, 6)
    members = target[None].clone()
    members[..., 1, 0] = 2
    result = variogram_score(members, target, mask, torch.ones(4, 6), lags=(1,))
    assert result["fair"] == 4
    mask.zero_()
    mask[:, [0, -1], :] = True
    members.zero_()
    members[..., 0, :] = 5
    result = variogram_score(members, target, mask, torch.ones(4, 6), lags=(1,))
    assert result["fair"] == 0


def test_cross_land_paths_are_excluded_not_just_endpoints():
    mask = torch.zeros(1, 5, 8, dtype=torch.bool)
    mask[0, 0, :3] = True  # One valid lag-two edge with zero error.
    mask[0, 3, [0, 2]] = True  # Land lies between these endpoints.
    target = torch.zeros(1, 1, 5, 8)
    members = target[None].clone()
    members[..., 3, 2] = 100
    result = variogram_score(members, target, mask, torch.ones(5, 8), lags=(2,))
    assert result["fair"] == 0


def test_longitude_rotation_invariance_and_rng_unchanged():
    members, target, mask, area = _inputs()
    mask[0, 1, 0] = False
    rng = torch.random.get_rng_state().clone()
    original = variogram_score(members, target, mask, area, lags=(1, 2))
    shifted = variogram_score(
        members.roll(2, dims=-1),
        target.roll(2, dims=-1),
        mask.roll(2, dims=-1),
        area.roll(2, dims=-1),
        lags=(1, 2),
    )
    torch.testing.assert_close(original["fair"], shifted["fair"])
    assert torch.equal(rng, torch.random.get_rng_state())


def test_all_land_fails_finite_loss_gate():
    members, target, mask, area = _inputs()
    result = variogram_score(members, target, mask & False, area, lags=(1,))
    assert not torch.isfinite(result["fair"])


@pytest.mark.parametrize(
    "lags", [(), (0,), (-1,), (1.0,), (True,), (2, 1), (1, 1), (4,)]
)
def test_invalid_lags_rejected(lags):
    with pytest.raises(ValueError):
        members, target, mask, area = _inputs()
        variogram_score(members, target, mask, area, lags=lags)


def test_latitude_only_area_matches_full_grid_score_and_gradient():
    torch.manual_seed(42)
    members = torch.randn(4, 2, 3, 6, 7, requires_grad=True)
    target = torch.randn(2, 3, 6, 7)
    mask = torch.ones(3, 6, 7, dtype=torch.bool)
    area = torch.rand(6, 1) + 0.1
    compact = variogram_score(members, target, mask, area, [1, 2])
    full = variogram_score(members, target, mask, area.expand(6, 7).clone(), [1, 2])
    for name in compact:
        torch.testing.assert_close(compact[name], full[name], rtol=0, atol=0)
    compact_gradient = torch.autograd.grad(compact["fair"], members, retain_graph=True)[
        0
    ]
    full_gradient = torch.autograd.grad(full["fair"], members)[0]
    torch.testing.assert_close(compact_gradient, full_gradient, rtol=0, atol=0)
