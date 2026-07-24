# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from scripts.audit_coarse_dynamics import (
    _empty_pair_stats,
    _finish_pair_stats,
    _parameter_summary,
    _rollout_from_state,
    _update_pair_stats,
)


def test_pair_stats_report_symmetric_error_and_token_cosine() -> None:
    left = torch.tensor([[[[1.0, 2.0]], [[0.0, 2.0]]]])
    right = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]]]])
    valid = torch.tensor([[[[True, False]]]])
    stats = _empty_pair_stats()

    _update_pair_stats(stats, left, right, valid)
    result = _finish_pair_stats(stats)

    assert result["values"] == 2
    assert result["tokens"] == 1
    assert result["mean_squared_difference"] == pytest.approx(0.0)
    assert result["symmetric_normalized_mse"] == pytest.approx(0.0)
    assert result["mean_token_cosine_similarity"] == pytest.approx(1.0)


def test_pair_stats_reject_nonbroadcastable_mask() -> None:
    with pytest.raises(ValueError, match="cannot broadcast"):
        _update_pair_stats(
            _empty_pair_stats(),
            torch.zeros(2, 3, 4, 5),
            torch.zeros(2, 3, 4, 5),
            torch.ones(4, 5, dtype=torch.bool),
        )


def test_parameter_summary_retains_shape_values_and_signs() -> None:
    summary = _parameter_summary(torch.tensor([[[[-2.0]], [[0.0]], [[1.0]]]]))

    assert summary["shape"] == [1, 3, 1, 1]
    assert summary["count"] == 3
    assert summary["minimum"] == pytest.approx(-2.0)
    assert summary["maximum"] == pytest.approx(1.0)
    assert summary["negative_fraction"] == pytest.approx(1 / 3)
    assert summary["near_zero_fraction"] == pytest.approx(1 / 3)
    assert summary["values"] == [-2.0, 0.0, 1.0]


def test_rollout_from_state_uses_one_aligned_boundary_per_depth() -> None:
    class FakeData:
        class Context:
            input_resolution_cpu = (torch.arange(1), torch.arange(1))

        ctx = Context()

        @staticmethod
        def get_input(step: int) -> tuple[None, torch.Tensor]:
            return None, torch.tensor(float(step + 1))

    class FakeModel:
        use_bfloat16 = False

        def __init__(self) -> None:
            self.boundaries: list[float] = []

        def process(
            self,
            state: torch.Tensor,
            _latent_resolution: tuple[torch.Tensor, torch.Tensor],
            *,
            iterations: int,
            boundary: torch.Tensor,
            boundary_resolution: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            assert iterations == 1
            assert boundary_resolution is FakeData.ctx.input_resolution_cpu
            self.boundaries.append(float(boundary))
            return state + boundary

    model = FakeModel()
    states = _rollout_from_state(
        model,  # type: ignore[arg-type]
        FakeData(),
        torch.tensor(0.0),
        (torch.arange(1), torch.arange(1)),
    )

    assert model.boundaries == [1.0, 2.0, 3.0, 4.0]
    assert {depth: float(state) for depth, state in states.items()} == {
        1: 1.0,
        2: 3.0,
        4: 10.0,
    }
