# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from scripts.audit_coarse_dynamics import (
    _empty_pair_stats,
    _finish_pair_stats,
    _parameter_summary,
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
