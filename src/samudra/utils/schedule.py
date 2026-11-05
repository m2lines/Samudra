# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal

import torch
from pydantic import BaseModel, Field


class CosineSchedulerConfig(BaseModel):
    """Cosine scheduler; see pytorch CosineAnnealingLR."""

    type: Literal["cosine"] = "cosine"
    target_epochs: int | None = None

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        max_epochs = self.target_epochs if self.target_epochs is not None else epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


class CosineWithTailSchedulerConfig(BaseModel):
    """Cosine scheduler which goes to tail_lr for the last tail_epochs."""

    type: Literal["cosine_with_tail"] = "cosine_with_tail"
    target_epochs: int | None = None

    tail_lr: float
    tail_epochs: int = 10

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        max_epochs = self.target_epochs if self.target_epochs is not None else epochs
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - self.tail_epochs, eta_min=self.tail_lr
        )
        tail = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=self.tail_lr / optimizer.param_groups[0]["lr"],
            total_iters=self.tail_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[cosine, tail],
            milestones=[max_epochs - self.tail_epochs],
        )


class CosineWithWarmupConfig(BaseModel):
    """Cosine scheduler which goes from warmup_lr to the default lr for the first warmup_epochs."""

    type: Literal["cosine_with_warmup"] = "cosine_with_warmup"
    target_epochs: int | None = None

    warmup_lr: float = 1e-6
    warmup_epochs: int = 5

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        max_epochs = self.target_epochs if self.target_epochs is not None else epochs
        assert len(optimizer.param_groups) == 1, (
            "There can only be one parameter group for the optimizer."
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.warmup_lr / optimizer.param_groups[0]["lr"],
            end_factor=1.0,  # Reaches full LR
            total_iters=self.warmup_epochs,
        )

        if self.warmup_epochs > max_epochs:
            raise ValueError(
                "'warmup_epochs' is too big; it must be smaller than the "
                "scheduler target epochs."
            )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - self.warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )


class LinearCooldownSchedulerConfig(BaseModel):
    """Reduce LR to final_lr over the supplied number of optimizer/epoch steps.

    The first step already decreases LR; the last step uses final_lr. The caller
    chooses the unit, as for the other scheduler configs, and calls step() after
    each optimizer update (or epoch) to prepare the next learning rate.
    """

    type: Literal["linear_cooldown"] = "linear_cooldown"
    final_lr: float = Field(gt=0)
    target_epochs: int | None = Field(default=None, ge=1)

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        steps = self.target_epochs if self.target_epochs is not None else epochs
        if steps < 1:
            raise ValueError("Linear cooldown requires at least one step")
        initial_rates = [float(group["lr"]) for group in optimizer.param_groups]
        if any(self.final_lr > rate for rate in initial_rates):
            raise ValueError("Cooldown final_lr must not exceed the starting LR")
        # LambdaLR uses initial_lr as its base, including across a phase change.
        for group, rate in zip(optimizer.param_groups, initial_rates, strict=True):
            group["initial_lr"] = rate

        def decay(rate: float) -> Callable[[int], float]:
            def factor(step: int) -> float:
                fraction = min(step + 1, steps) / steps
                return 1 + fraction * (self.final_lr / rate - 1)

            return factor

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[decay(rate) for rate in initial_rates]
        )


SchedulerConfig = (
    CosineSchedulerConfig
    | CosineWithTailSchedulerConfig
    | CosineWithWarmupConfig
    | LinearCooldownSchedulerConfig
)
