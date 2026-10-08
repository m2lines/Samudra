# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "annual_report",
    Path(__file__).parents[1] / "scripts/report_diffusion_global_annual.py",
)
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_temporal_report_distinguishes_noise_and_ignores_missing_reference():
    truth = np.broadcast_to(np.arange(3.0)[:, None, None], (3, 2, 2)).copy()
    signs = np.array([-1.0, 1.0])[:, None, None, None]
    members = truth[None] + signs
    truth[:, 0, 0] = np.nan
    members[:, :, 0, 0] = 1e9
    mask, area = np.ones((2, 2), dtype=bool), np.ones((2, 1))
    result = report.increments(members, truth, mask, area)
    np.testing.assert_allclose(list(result.values()), [1, 1, 1, 1])
    phase = np.array([1.0, -1.0, 1.0])[None, :, None, None]
    result = report.increments(truth[None] + signs * phase, truth, mask, area)
    np.testing.assert_allclose(list(result.values()), [np.sqrt(5), 1, -1, 1])
