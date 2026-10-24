# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from types import SimpleNamespace
from typing import cast

import torch
from torch import nn

from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.models.samudra import Samudra


class GateTests(unittest.TestCase):
    kind = "features"

    def projection(self, model):
        gate = model.persistence_gate
        return gate[-1] if isinstance(gate, nn.Sequential) else gate

    def make(self, enabled=True, outputs=6):
        torch.manual_seed(15)
        unet = nn.Conv2d(8, 8, 1)
        unet.out_channels = 8
        return Samudra(
            8,
            outputs,
            False,
            3,
            "circular",
            cast(UNetBackbone, unet),
            0,
            None,
            2,
            (9, 12),
            0,
            False,
            persistence_gate=enabled,
            persistence_gate_kind=self.kind,
            prognostic_channels=6,
        )

    def setUp(self):
        torch.set_num_threads(2)
        self.x = torch.randn(2, 6, 9, 12)
        self.b = torch.randn(2, 2, 9, 12)
        self.mask = torch.ones(1, 6, 9, 12, dtype=torch.bool)
        self.mask[:, :, -1] = False
        self.ctx = SimpleNamespace(label_mask=self.mask)

    def test_paired_initialization_and_default(self):
        base = self.make(False)
        rng = torch.get_rng_state()
        gate = self.make()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in base.state_dict().items():
            torch.testing.assert_close(v, gate.state_dict()[k], rtol=0, atol=0)
        b = base.forward_once(self.x, self.b, self.ctx)
        g = gate.forward_once(self.x, self.b, self.ctx)
        latest = self.x[:, 3:].repeat(1, 2, 1, 1)
        torch.testing.assert_close(
            g,
            torch.where(
                self.mask, b + torch.sigmoid(torch.tensor(-6.0)) * (latest - b), 0
            ),
        )

    def test_latest_frame_for_each_lead_and_mask(self):
        for outputs in (3, 6):
            model = self.make(outputs=outputs)
            with torch.no_grad():
                self.projection(model).bias.fill_(30.0)
            ctx = SimpleNamespace(label_mask=self.mask[:, :outputs])
            result = model.forward_once(self.x, self.b, ctx)
            latest = self.x[:, 3:].repeat(1, outputs // 3, 1, 1)
            torch.testing.assert_close(result, torch.where(ctx.label_mask, latest, 0))

    def test_gradients_through_feedback_and_serialization(self):
        model = self.make()
        x = self.x.clone().requires_grad_()
        pred = model.forward_once(x, self.b, self.ctx)
        pred2 = model.forward_once(pred, self.b, self.ctx)
        pred2.square().mean().backward()
        for param in (
            self.projection(model).weight,
            model.decoder.weight,
            model.unet.weight,
            x,
        ):
            assert param.grad is not None
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(param.grad.abs().sum()), 0)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        opt.step()
        saved = copy.deepcopy(
            {"model": model.state_dict(), "optimizer": opt.state_dict()}
        )
        restored = self.make()
        restored.load_state_dict(saved["model"])
        opt2 = torch.optim.Adam(restored.parameters(), lr=0.001)
        opt2.load_state_dict(saved["optimizer"])
        for m, o in ((model, opt), (restored, opt2)):
            o.zero_grad()
            m.forward_once(self.x, self.b, self.ctx).square().mean().backward()
            o.step()
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


class DirectGateTests(GateTests):
    kind = "direct"

    def test_gate_receives_latest_and_forecasts_without_detaching(self):
        model = self.make()
        captured = []
        decoder = []

        def capture_gate(module, inputs):
            captured.append(inputs[0])

        def capture_decoder(module, inputs, output):
            decoder.append(output)

        h = model.persistence_gate.register_forward_pre_hook(capture_gate)
        d = model.decoder.register_forward_hook(capture_decoder)
        x = self.x.clone().requires_grad_()
        model.forward_once(x, self.b, self.ctx)
        h.remove()
        d.remove()
        gate_input = captured[0]
        torch.testing.assert_close(gate_input[:, 8:11], x[:, 3:])
        torch.testing.assert_close(gate_input[:, 11:], decoder[0])
        gate_input[:, 8:].square().mean().backward()
        assert x.grad is not None
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertGreater(float(model.decoder.weight.grad.abs().sum()), 0)


class NonlinearGateTests(DirectGateTests):
    kind = "nonlinear"

    def test_hidden_layer_learns_after_zero_projection_first_step(self):
        model = self.make()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(2):
            opt.zero_grad()
            model.forward_once(self.x, self.b, self.ctx).square().mean().backward()
            opt.step()
        self.assertGreater(float(model.persistence_gate[0].weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
