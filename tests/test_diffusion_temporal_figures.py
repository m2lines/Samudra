# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from samudra.experiments.diffusion_temporal_figures import ratios


def test_increment_ratio_uses_rms_and_leaves_zero_reference_undefined():
    result = ratios(
        dict(
            errors=dict(channels=["a", "b"]),
            temporal=dict(
                truth_increment_mse=[4.0, 0.0],
                member_increment_mse=[36.0, 1.0],
                mean_increment_mse=[9.0, 1.0],
                adjacent_member_correlation=[0.0, 0.0],
            ),
        )
    )
    assert result["member"][0] == 3.0
    assert result["mean"][0] == 1.5
    assert np.isnan(result["member"][1])
    assert np.isnan(result["mean"][1])
