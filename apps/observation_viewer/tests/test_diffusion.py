# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Sparse diffusion model metadata must not acquire deterministic descriptions."""

from model_info import description


def test_diffusion_description_preserves_sampling_and_missing_data_caveats():
    note = "Eight independently sampled decodes; missing depths remain unavailable."
    assert description("diffusion-test-draw1", {"description": note}) == note
