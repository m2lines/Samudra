# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from samudra.experiments.diffusion_global_annual import CHANNELS, AnnualMembers


def test_annual_export_preserves_members_units_and_rng_while_returning_mean(tmp_path):
    class Forecast(nn.Module):
        def forecast(self, surface, *, members, generator):
            return (
                torch.randn(members, 1, 73, 77, 2, 3, generator=generator),
                torch.randn(members, 1, 2, 77, 2, 3, generator=generator),
            )

    model = Forecast()
    data = SimpleNamespace(
        grid=dict(
            mean=np.arange(77, dtype=np.float32),
            std=np.arange(1, 78, dtype=np.float32),
            names=np.array([f"c{i}" for i in range(77)]),
            lat=np.array([-40, 40]),
            lon=np.array([0, 120, 240]),
            mask=np.ones((77, 2, 3), dtype=bool),
        )
    )
    surface = torch.zeros(1)
    origins = [tmp_path / "2015-01-01" / "COMPLETE.json"]
    wrapper = AnnualMembers(model, data, origins, tmp_path, members=3, seed=42)
    point, initial = wrapper.forecast(surface)
    expected, expected_initial = model.forecast(
        surface, members=3, generator=torch.Generator().manual_seed(42)
    )
    torch.testing.assert_close(point, expected.mean(0))
    torch.testing.assert_close(initial, expected_initial.mean(0))
    with np.load(tmp_path / "members-2015-01-01.npz") as saved:
        assert saved["fields"].shape == (3, 73, 4, 2, 3)
        assert saved["initial"].shape == (3, 2, 4, 2, 3)
        for j, channel in enumerate(CHANNELS):
            np.testing.assert_allclose(
                saved["fields"][:, :, j],
                expected[:, 0, :, channel].numpy() * (channel + 1) + channel,
            )
        np.testing.assert_array_equal(saved["leads_days"][[5, 72]], [30, 365])
