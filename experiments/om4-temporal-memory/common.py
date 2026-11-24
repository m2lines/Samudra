"""Pinned official architecture and explicit-state experimental variants."""

import hashlib
import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
from legacy.activations import CappedGELU
from legacy.blocks import ConvNeXtBlock, CoreBlock, ZonallyPeriodicBilinearUpsample
from legacy.unet_backbone import UNetBackbone
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DEPTHS = [
    2.5,
    10.0,
    22.5,
    40.0,
    65.0,
    105.0,
    165.0,
    250.0,
    375.0,
    550.0,
    775.0,
    1050.0,
    1400.0,
    1850.0,
    2400.0,
    3100.0,
    4000.0,
    5000.0,
    6000.0,
]
PROG = [f"{v}_{k}" for v in ("uo", "vo", "thetao", "so") for k in range(19)] + ["zos"]
BOUND = ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
ARMS = ("full", "detach", "latent_full", "latent_only")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class preserve_batchnorm:
    """Reusable replay context: each backward restores BatchNorm buffers."""

    def __init__(self, module):
        self.module = module
        self.saved = []

    def __enter__(self):
        self.saved.append(
            [
                (b, b.clone())
                for m in self.module.modules()
                if isinstance(m, nn.BatchNorm2d)
                for b in (m.running_mean, m.running_var, m.num_batches_tracked)
            ]
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with torch.no_grad():
            for b, old in self.saved.pop():
                b.copy_(old)
        return False


class OceanModel(nn.Module):
    def __init__(
        self,
        mask,
        arm="full",
        latent_channels=16,
        widths=(280, 380, 480, 520),
        prog_channels=77,
        boundary_channels=4,
        checkpointing=True,
    ):
        super().__init__()
        self.arm = arm
        self.prog_channels = prog_channels
        self.checkpointing = checkpointing
        self.register_buffer("wet", mask.repeat(2, 1, 1)[None], persistent=False)
        self.unet = UNetBackbone(
            in_channels=2 * (prog_channels + boundary_channels),
            ch_width=list(widths),
            dilation=[1, 2, 4, 8][: len(widths)],
            n_layers=[1] * len(widths),
            pad="circular",
            create_block=partial(
                ConvNeXtBlock,
                kernel_size=3,
                activation=CappedGELU,
                upscale_factor=2,
                norm="batch",
            ),
            downsampling_block=nn.AvgPool2d(2),
            create_upsampling_block=lambda **_: ZonallyPeriodicBilinearUpsample(),
            checkpointing=None,
        )
        self.decoder = nn.Conv2d(widths[0], 2 * prog_channels, 3)
        self.middle_index = 2 * len(widths)
        if arm.startswith("latent"):
            # Modules are allocated AFTER the complete physical model, preserving
            # identical seeded physical parameter initialization in all four arms.
            self.latent_update = nn.Conv2d(
                widths[-1] + latent_channels, latent_channels, 1
            )
            self.latent_read = nn.Conv2d(latent_channels, widths[-1], 1, bias=False)
            nn.init.zeros_(self.latent_read.weight)
            self.latent_channels = latent_channels

    def _block(self, layer, x):
        if self.checkpointing and torch.is_grad_enabled():
            return checkpoint(
                layer,
                x,
                use_reentrant=False,
                context_fn=lambda: (nullcontext(), preserve_batchnorm(layer)),
            )
        return layer(x)

    def _features(self, x, h):
        skips = []
        up = 0
        for i, layer in enumerate(self.unet.layers):
            x = self._block(layer, x)
            if i < self.middle_index and isinstance(layer, CoreBlock):
                skips.append(x)
            if i == self.middle_index and self.arm.startswith("latent"):
                if h is None:
                    h = x.new_zeros(x.shape[0], self.latent_channels, *x.shape[-2:])
                h = torch.tanh(self.latent_update(torch.cat((x, h), dim=1)))
                x = x + self.latent_read(h)
            if isinstance(layer, ZonallyPeriodicBilinearUpsample):
                skip = skips[-1 - up]
                dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
                x = F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2)) + skip
                up += 1
        return x, h

    def forward(self, physical, boundary, h=None):
        x = torch.cat((physical, boundary), dim=1)
        x, h = self._features(x, h)
        x = F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))
        return self.decoder(x).masked_fill(~self.wet, 0), h

    def feedback(self, physical, h):
        if self.arm in ("detach", "latent_only"):
            physical = physical.detach()
        return physical, h


def load_official(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in saved["model"].items()}
    model.load_state_dict(state, strict=True)
    return {k: saved[k] for k in ("epoch", "wandb_id", "wandb_name")}


def mse(pred, truth, wet):
    """Fixed normalized MSE, equal channel weight, wet-cell mean per channel."""
    sq = (pred - truth).square() * wet
    per = sq.sum((-2, -1)) / wet.sum((-2, -1)).clamp_min(1)
    return per.mean()


def interior_mask(wet):
    result = wet.clone()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        result &= torch.roll(wet, (dy, dx), (-2, -1))
    result[..., 0, :] = False
    result[..., -1, :] = False
    return result


def highpass(x, interior):
    """Local curvature; periodic longitude, excluded latitude/mask edges."""
    neighbors = (
        sum(
            torch.roll(x, shift, (-2, -1))
            for shift in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        / 4
    )
    return (x - neighbors) * interior


class Prepared:
    def __init__(self, path):
        path = Path(path)
        self.meta = json.loads((path / "metadata.json").read_text())
        self.prog = np.load(path / "prognostic.npy", mmap_mode="r")
        self.bound = np.load(path / "boundary.npy", mmap_mode="r")
        self.mask = np.load(path / "mask.npy")
        self.mean = np.array(self.meta["means"][:77], dtype=np.float32)
        self.std = np.array(self.meta["stds"][:77], dtype=np.float32)
        self.times = self.meta["times"]

    def window(self, start, calls, device):
        n = 2 + 2 * calls
        original = self.meta["source_indices"][start : start + n]
        if len(original) != n or not np.all(np.diff(original) == 1):
            raise ValueError("Window crosses a gap or end in prepared timestamps")
        p = torch.from_numpy(np.array(self.prog[start : start + n])).to(device)
        b = torch.from_numpy(np.array(self.bound[start : start + n])).to(device)
        if len(p) != n:
            raise ValueError("Window exceeds available time range")
        return (
            p[:2].flatten(0, 1)[None],
            [b[2 * k : 2 * k + 2].flatten(0, 1)[None] for k in range(calls)],
            [p[2 * k + 2 : 2 * k + 4].flatten(0, 1)[None] for k in range(calls)],
        )


def metrics(pred, target, mask, std):
    p = pred.reshape(-1, 77, *pred.shape[-2:])
    t = target.reshape_as(p)
    wet = mask[None]
    inside = interior_mask(wet)

    def rms(a, valid):
        return (
            (a.square() * valid).sum((0, 2, 3))
            / (valid.sum((0, 2, 3)) * a.shape[0]).clamp_min(1)
        ).sqrt()

    error = rms(p - t, wet)
    hp_p = rms(highpass(p, inside), inside)
    hp_t = rms(highpass(t, inside), inside)
    hp_e = rms(highpass(p - t, inside), inside)
    return {
        "rmse_normalized": error.tolist(),
        "rmse_physical": (error * std).tolist(),
        "roughness_prediction": hp_p.tolist(),
        "roughness_truth": hp_t.tolist(),
        "roughness_error": hp_e.tolist(),
    }
