# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from samudra.utils.band_loss import BandLoss, masked_highpass
from samudra.utils.ctx import BatchGrid
from samudra.utils.loss import loss_fn_from_metric


class BandTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(15)
        self.mask = torch.ones(4, 12, 16, dtype=torch.bool)
        self.mask[:, 3:6, 5:8] = False
        self.mask[1, :, 0:4] = False
        self.mask[3] = self.mask[1]
        self.ctx = BatchGrid(
            self.mask,
            (torch.arange(12), torch.arange(16)),
            (torch.arange(12), torch.arange(16)),
        )
        self.args = dict(
            device=torch.device("cpu"),
            num_channels=2,
        )

    def test_constant_offset_and_coastline(self):
        x = torch.full((1, 4, 12, 16), 3.0)
        x[:, ~self.mask] = float("nan")
        hp = masked_highpass(x, self.mask)
        self.assertTrue(torch.isfinite(hp).all())
        self.assertLess(hp.abs().max().item(), 1e-5)

    def test_longitude_roll_and_foreign_depth_mask(self):
        x = torch.randn(1, 4, 12, 16)
        a = masked_highpass(x, self.mask)
        b = masked_highpass(x.roll(4, -1), self.mask.roll(4, -1))
        torch.testing.assert_close(a.roll(4, -1), b)
        changed = x.clone()
        changed[:, 1] += 20
        torch.testing.assert_close(masked_highpass(changed, self.mask)[:, 0], a[:, 0])

    def test_exact_addition_and_no_dynamic_updates(self):
        base = loss_fn_from_metric("mse")
        band = BandLoss(**self.args, band_scales=[0.3, 0.7], band_weight=0.1)
        x = torch.randn(2, 4, 12, 16, requires_grad=True)
        target = torch.randn_like(x)
        self.assertFalse(hasattr(band, "update"))
        self.assertFalse(hasattr(band, "loss_scale_per_channel"))
        expected = base(x, target, self.ctx) + 0.1 * masked_highpass(
            x - target, self.mask
        ).square().mean((0, 2, 3)) / torch.tensor([0.3, 0.7, 0.3, 0.7])
        torch.testing.assert_close(band(x, target, self.ctx), expected, rtol=0, atol=0)
        band(x, target, self.ctx).mean().backward()
        assert x.grad is not None
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertEqual(x.grad[:, ~self.mask].abs().sum().item(), 0)

    def test_resume_calibration_identity(self):
        a = BandLoss(**self.args, band_scales=[0.3, 0.7], band_weight=0.1)
        b = BandLoss(**self.args, band_scales=[0.3, 0.7], band_weight=0.1)
        b.load_state_dict(a.state_dict())
        c = BandLoss(**self.args, band_scales=[0.2, 0.7], band_weight=0.1)
        with self.assertRaises(ValueError):
            c.load_state_dict(a.state_dict())

    def test_amplitude_calibration_invariance(self):
        x = torch.randn(2, 4, 12, 16)
        hp = masked_highpass(x, self.mask)
        scale = hp.square().mean((0, 2, 3))
        hp2 = masked_highpass(10 * x, self.mask)
        torch.testing.assert_close(
            hp.square().mean((0, 2, 3)) / scale,
            hp2.square().mean((0, 2, 3)) / (100 * scale),
        )


if __name__ == "__main__":
    unittest.main()
