# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from samudra.experiments.monthly_latent import (
    MonthlyLatentOcean,
    initializer_visibility,
)
from scripts.run_monthly_replay import phase_settings


def test_only_processor_forcing_is_removed_without_changing_weights():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = MonthlyLatentOcean(
        8, 8, (16, 32), history=2, encoder_width=8, processor_forcing=False
    )
    fields = torch.randn(1, 2, 77, 16, 32)
    forcing = torch.randn(1, 3, 3, 16, 32, requires_grad=True)
    context = torch.randn(1, 3, 5, 16, 32)
    wet = torch.ones(77, 16, 32, dtype=torch.bool)
    available = initializer_visibility(
        wet, "surface", torch.Generator().manual_seed(1)
    )[None, None]
    initial = model.initialize(fields, available, forcing[:, :2], context[:, 1])
    a = model.processor(initial, forcing[:, 2], context[:, 2])
    b = model.processor(initial, forcing[:, 2] * 100 + 37, context[:, 2])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().mean().backward()
    assert forcing.grad is not None
    assert forcing.grad[:, :2].abs().sum() > 0
    assert forcing.grad[:, 2].abs().sum() == 0
    model.processor.forcing_enabled = True
    c = model.processor(initial.detach(), forcing[:, 2].detach(), context[:, 2])
    d = model.processor(
        initial.detach(), forcing[:, 2].detach() * 100 + 37, context[:, 2]
    )
    assert not torch.equal(c, d)


def test_fresh_2k_schedule_has_matched_source_exposure_and_obs_cooldown():
    for arm, expected in [("A", 1000), ("E", 2000)]:
        assert (
            sum(phase_settings(i, "fresh-2k", arm)[1] == "obs" for i in range(2000))
            == expected
        )
        assert all(
            phase_settings(i, "fresh-2k", arm)[1] == "obs" for i in range(2000, 3000)
        )
