# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from samudra.experiments.diffusion_latent_diagnostics import temporal


def test_member_memory_and_masked_cells():
    # Two equally weighted members: persistent anomaly, then reversed anomaly.
    fields = np.ones((2, 3, 1, 2, 2))
    fields[1] *= -1
    fields[:, 2] *= -1
    fields[..., 0, 0] = np.nan
    weights = np.ones((1, 2, 2))
    weights[..., 0, 0] = 0
    result = temporal(fields, weights)
    np.testing.assert_allclose(result["member_anomaly_correlation"], [[1], [-1]])
    np.testing.assert_allclose(result["member_increment_mse"], [[0], [4]])
    np.testing.assert_allclose(result["mean_increment_mse"], [[0], [0]])
