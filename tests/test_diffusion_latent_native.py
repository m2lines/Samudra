# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.diffusion_latent_native import portable, temporal_statistics


def test_temporal_statistics_distinguish_persistent_and_alternating_member_noise():
    truth = torch.arange(3.0).reshape(1, 3, 1, 1, 1).expand(1, 3, 1, 2, 2)
    signs = torch.tensor([-1.0, 1.0]).reshape(2, 1, 1, 1, 1, 1)
    persistent = truth[None] + signs
    weights = torch.ones(1, 2, 2)
    result = temporal_statistics(persistent, truth, weights)
    for key in result:
        torch.testing.assert_close(result[key], torch.ones(1))
    phase = torch.tensor([1.0, -1.0, 1.0]).reshape(1, 1, 3, 1, 1, 1)
    alternating = truth[None] + signs * phase
    result = temporal_statistics(alternating, truth, weights)
    torch.testing.assert_close(result["adjacent_member_correlation"], -torch.ones(1))
    torch.testing.assert_close(result["member_increment_mse"], torch.tensor([5.0]))
    torch.testing.assert_close(result["mean_increment_mse"], torch.ones(1))


def test_nested_diagnostics_preserve_missing_values_in_portable_json():
    assert portable({"velocity": {"nested": torch.tensor([float("nan"), 2.0])}}) == {
        "velocity": {"nested": [None, 2.0]}
    }
