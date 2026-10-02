# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Physical-state processors with explicit extent and regional boundaries.

The local processor has radius four per step. Its locality statement conditions
on its supplied states and geometry; the common U-Net initializer is nonlocal.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from samudra.experiments.initializer_models import HistoryInitializer
from samudra.experiments.missingness import state_mask
from samudra.experiments.surface_state import make_unet


@contextmanager
def regional_padding(module, regional):
    """Scope padding to this forward, including activation recomputation.

    These backbones have no internal checkpointing. The caller checkpoints the
    whole forward, including this context, and supplies the task explicitly.
    """
    changed = []
    if regional:
        for child in module.modules():
            if getattr(child, "pad", None) == "circular":
                changed.append(child)
                child.pad = "constant"
    try:
        yield
    finally:
        for child in changed:
            child.pad = "circular"


def geometry_planes(context, mask, regional):
    """Geometry from supplied spherical coordinates, never 360 / tensor width.

    Channels: local zonal/meridional chord spacing, total zonal/meridional
    angular extent, regional flag, wet surface, fraction of wet physical slots.
    Spacings are scaled by 100 to keep magnitudes useful to a linear projection.
    """
    xyz = context[:, :3].float()
    h, w = xyz.shape[-2:]
    if min(h, w) < 2:
        raise ValueError("Geometry needs at least two cells in each direction")
    dx = (xyz[..., 1:] - xyz[..., :-1]).square().sum(1, keepdim=True).sqrt()
    dy = (xyz[..., 1:, :] - xyz[..., :-1, :]).square().sum(1, keepdim=True).sqrt()
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    lon = torch.atan2(xyz[:, 1:2], xyz[:, 0:1])
    delta = lon[..., 1:] - lon[..., :-1]
    delta = torch.atan2(delta.sin(), delta.cos()).abs().mean((-2, -1), keepdim=True)
    lat = xyz[:, 2:3].clamp(-1, 1).asin()
    height = (lat[..., 1:, :] - lat[..., :-1, :]).abs().mean((-2, -1), keepdim=True) * h
    shape = dx.shape
    return torch.cat(
        (
            100 * dx,
            100 * dy,
            (delta * w).expand(shape),
            height.expand(shape),
            torch.full_like(dx, float(regional)),
            mask.any(0)[None, None].expand(shape).float(),
            mask.float().mean(0)[None, None].expand(shape),
        ),
        1,
    )


class ExtentInitializer(HistoryInitializer):
    def forward(
        self, surface, past_forcing, context, mask, input_mask=None, task="observation"
    ):
        regional = task == "om4-patch"
        with regional_padding(self.net, regional):
            return super().forward(
                surface,
                past_forcing,
                context,
                mask,
                input_mask,
                "om4" if regional else task,
            )


class LocalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.spatial = nn.Conv2d(width, width, 3, groups=width)
        self.pointwise = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, width, 1)
        )
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 0.1))

    def forward(self, value, regional):
        padded = F.pad(value, (1, 1, 0, 0), mode="constant" if regional else "circular")
        padded = F.pad(padded, (0, 0, 1, 1))
        return value + self.scale * self.pointwise(self.spatial(padded))


class LocalNet(nn.Module):
    radius = 4

    def __init__(self, inputs, outputs, width=512):
        super().__init__()
        self.input = nn.Conv2d(inputs, width, 1)
        self.blocks = nn.ModuleList([LocalBlock(width) for _ in range(self.radius)])
        self.output = nn.Conv2d(width, outputs, 1)

    def forward(self, value, regional):
        value = self.input(value)
        for block in self.blocks:
            value = block(value, regional)
        return self.output(value)


class ExtentEvolution(nn.Module):
    def __init__(self, channels, local=False, local_width=512):
        super().__init__()
        self.local = local
        inputs = 2 * channels + 8
        self.geometry = nn.Conv2d(7, inputs, 1, bias=False)
        nn.init.zeros_(self.geometry.weight)
        self.net = (
            LocalNet(inputs, channels, local_width)
            if local
            else make_unet(inputs, channels, [128, 192, 256, 384])
        )

    def forward(self, states, forcing, context, mask, lead, task="observation"):
        regional = task == "om4-patch"
        inputs = torch.cat((states.flatten(1, 2), context, forcing[:, lead - 1]), 1)
        adapters = getattr(self, "input_adapters", None)
        if adapters is not None:
            assert isinstance(adapters, nn.ModuleDict)
            inputs = adapters["om4" if regional else task](inputs)
        inputs = inputs + self.geometry(geometry_planes(context, mask, regional))
        if self.local:
            result = self.net(inputs, regional)
        else:
            with regional_padding(self.net, regional):
                result = self.net(inputs)
        return result.float() * state_mask(mask, result.shape[1])
