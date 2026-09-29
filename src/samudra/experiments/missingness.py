# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit missing-input contracts and source-specific identity input adapters."""

import torch
from torch import nn

MODEL_DEFAULTS = {
    "initializer_architecture": "wide",
    "surface_policy": "legacy-copy",
    "task_conditioning": "none",
    "latent_channels": 0,
}


def model_options(arguments):
    return {key: arguments.get(key, default) for key, default in MODEL_DEFAULTS.items()}


def state_mask(mask, channels):
    """Physical wet masks plus surface-ocean support for optional latent slots."""
    extra = channels - mask.shape[0]
    if extra < 0:
        raise ValueError("State has fewer channels than the physical wet mask")
    if not extra:
        return mask
    wet = mask.bool().any(0, keepdim=True).to(mask.dtype)
    return torch.cat((mask, wet.expand(extra, -1, -1)), 0)


def identity_adapters(channels):
    # Adding conditioning must not change the common backbone initialization.
    with torch.random.fork_rng(devices=[]):
        adapters = nn.ModuleDict()
        for task in ("om4", "observation"):
            layer = nn.Conv2d(channels, channels, 1)
            nn.init.dirac_(layer.weight)
            assert layer.bias is not None
            nn.init.zeros_(layer.bias)
            adapters[task] = layer
    return adapters


class TaskView(nn.Module):
    """Explicit source binding for the historical read-only OM4 evaluator."""

    def __init__(self, module, task):
        super().__init__()
        self.module, self.task = module, task

    @property
    def surface(self):
        return self.module.surface

    def forward(self, *args):
        return self.module(*args, task=self.task)


def structured_visibility(validity, seed):
    """Persistent blocks plus an optional latitude band; no global RNG changes.

    validity is [batch, time, variable, y, x]. Artificial hiding changes only
    inputs. Keep the original validity for labels, including in naturally missing
    regions. Longitude blocks wrap, latitude bands do not.
    """
    if validity.ndim != 5 or validity.shape[2] != 2:
        raise ValueError("Expected two surface variables with batch/time dimensions")
    batch, _, channels, h, w = validity.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    hidden = torch.zeros((batch, 1, channels, h, w), dtype=torch.bool)
    for b in range(batch):
        for c in range(channels):
            for _ in range(3):
                height = max(
                    1, int(h * (0.08 + 0.12 * torch.rand((), generator=generator)))
                )
                width = max(
                    1, int(w * (0.12 + 0.20 * torch.rand((), generator=generator)))
                )
                y = int(torch.randint(max(1, h - height + 1), (), generator=generator))
                x = int(torch.randint(w, (), generator=generator))
                columns = (torch.arange(width) + x) % w
                hidden[b, 0, c, y : y + height, columns] = True
            if bool(torch.rand((), generator=generator) < 0.5):
                height = max(1, h // 12)
                y = int(torch.randint(max(1, h - height + 1), (), generator=generator))
                hidden[b, 0, c, y : y + height] = True
    return validity.bool() & ~hidden.to(validity.device)


def corrupt_sample(sample, seed):
    """Return separate inputs/targets; never mutate cached or supplied labels."""
    result = dict(sample)
    visible = structured_visibility(sample["validity"], seed)
    result["surface"] = torch.where(visible, sample["surface"], 0)
    result["validity"] = visible
    result["completion_target"] = sample["surface"][:, 17:19]
    result["completion_valid"] = (
        sample["validity"][:, 17:19].bool() & ~visible[:, 17:19]
    )
    return result


def completion_loss(prediction, target, valid, latitude):
    """Mean normalized MSE by field/frame, full-grid wet observed support only."""
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Completion predictions, labels and support must align")
    if not torch.isfinite(target[valid]).all():
        raise ValueError("Nonfinite completion label on declared valid support")
    area = torch.cos(torch.deg2rad(torch.as_tensor(latitude, device=prediction.device)))
    weight = valid * area[None, None, None, :, None]
    denominator = weight.sum((-2, -1))
    error = torch.where(valid, prediction.float() - target, 0).square()
    channel = (error * weight).sum((-2, -1)) / denominator.clamp_min(1e-12)
    present = denominator > 0
    if not present.any():
        return prediction.sum() * 0
    return channel[present].mean()
