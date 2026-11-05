# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Accumulate an importance-corrected global batch before an optimizer update."""

from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .coverage import corrected_mean


def accumulate_gradients(model, next_batch, objective, vocab, device, accumulation):
    """Equal microbatches; DDP averages ranks and this function averages microsteps.

    The caller zeros gradients, clips once, steps the optimizer, and updates EMA.
    Both forward and backward must be inside no_sync on intermediate microsteps.
    Returned statistics are unscaled mean loss, mean importance, and mean square
    importance, averaged over every microbatch in the optimizer update.
    """
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError("Accumulation must be a positive integer")
    stats = torch.zeros(3, device=device)
    expected_size = None
    for microstep in range(accumulation):
        batch = next_batch()
        size = len(batch["target"])
        if not size or (expected_size is not None and size != expected_size):
            raise ValueError("Accumulation requires equal nonempty microbatches")
        expected_size = size
        context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel)
            and microstep + 1 < accumulation
            else nullcontext()
        )
        with context:
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(batch)
            if objective == "regression":
                losses = F.mse_loss(
                    logits.squeeze(-1), batch["target"], reduction="none"
                )
            elif objective == "absolute_ce":
                if vocab.outside(batch["target"], batch["depth"]).any():
                    raise ValueError("Target outside training-only support")
                losses = F.cross_entropy(
                    logits,
                    vocab.encode(batch["target"], batch["depth"]),
                    reduction="none",
                )
            else:
                raise ValueError(f"Unsupported accumulated objective: {objective}")
            loss = corrected_mean(losses, batch["importance"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite importance-corrected loss")
            (loss / accumulation).backward()
        stats += (
            torch.stack(
                (
                    loss.detach(),
                    batch["importance"].mean(),
                    batch["importance"].square().mean(),
                )
            )
            / accumulation
        )
    return stats
