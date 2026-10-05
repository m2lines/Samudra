# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit normalization choice for fresh experimental models."""

from torch import nn


def configure_normalization(module: nn.Module, kind: str) -> nn.Module:
    """Replace spatial BatchNorm in a newly constructed model, before loading weights."""
    if kind not in ("batch", "instance"):
        raise ValueError(f"Unknown normalization: {kind}")
    if kind == "batch":
        return module
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.InstanceNorm2d(
                child.num_features,
                eps=child.eps,
                affine=True,
                track_running_stats=False,
            )
            reference = child.weight if child.affine else child.running_mean
            if reference is not None:
                replacement.to(device=reference.device, dtype=reference.dtype)
            module.add_module(name, replacement)
        elif isinstance(child, nn.modules.batchnorm._BatchNorm):
            raise TypeError(f"Unsupported spatial normalization: {type(child)}")
        else:
            configure_normalization(child, kind)
    return module
