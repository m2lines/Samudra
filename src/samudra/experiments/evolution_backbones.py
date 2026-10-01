# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Versioned evolution architectures for controlled observation experiments."""

from pathlib import Path

import torch
import yaml
from torch import nn

from samudra.config import UNetBackboneConfig
from samudra.experiments.surface_state import Evolution


class Samudra2Output(nn.Module):
    """Reference Samudra output padding: periodic longitude, zero latitude."""

    def __init__(self, inputs, outputs, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        self.decoder = nn.Conv2d(inputs, outputs, kernel_size)

    def forward(self, value):
        p = self.padding
        value = torch.nn.functional.pad(value, (p, p, 0, 0), mode="circular")
        value = torch.nn.functional.pad(value, (0, 0, p, p), mode="constant")
        with torch.autocast(value.device.type, enabled=False):
            return self.decoder(value.float())


class Samudra2Evolution(Evolution):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.kind, self.leads = "ar", 6
        path = Path(__file__).parents[1] / "configs/samudra_om4_v2/model.yaml"
        config = yaml.safe_load(path.read_text())
        self.backbone_contract = config
        backbone = UNetBackboneConfig(**config["unet"]).build(
            2 * channels + 8, pad=config["pad"], checkpointing=None
        )
        self.net = nn.Sequential(
            backbone,
            Samudra2Output(backbone.out_channels, channels, config["last_kernel_size"]),
        )


def build_evolution(channels, architecture="d"):
    if architecture == "d":
        return Evolution(channels, [128, 192, 256, 384], "ar")
    if architecture == "samudra2":
        return Samudra2Evolution(channels)
    raise ValueError(f"Unknown evolution architecture: {architecture}")
