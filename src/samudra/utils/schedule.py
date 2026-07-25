# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import torch
from pydantic import BaseModel


class CosineSchedulerConfig(BaseModel):
    """Cosine scheduler; see pytorch CosineAnnealingLR."""

    type: Literal["cosine"] = "cosine"
    target_epochs: int | None = None
    interval: Literal["epoch"] = "epoch"

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        max_epochs = self.target_epochs if self.target_epochs is not None else epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


class CosineWithTailSchedulerConfig(BaseModel):
    """Cosine scheduler which goes to tail_lr for the last tail_epochs."""

    type: Literal["cosine_with_tail"] = "cosine_with_tail"
    target_epochs: int | None = None
    interval: Literal["epoch"] = "epoch"

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
    interval: Literal["epoch"] = "epoch"

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

        assert self.warmup_epochs <= epochs, (
            "'warmup_epochs' is too big; it must be smaller than 'epochs'."
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - self.warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )


class WarmupCosineUpdatesConfig(BaseModel):
    """Linear update warmup followed by update-wise cosine annealing."""

    type: Literal["warmup_cosine_updates"] = "warmup_cosine_updates"
    interval: Literal["update"] = "update"
    total_updates: int
    warmup_updates: int = 500
    min_lr: float = 0.0

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        del epochs
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive.")
        if not 0 <= self.warmup_updates < self.total_updates:
            raise ValueError("warmup_updates must be in [0, total_updates).")
        max_lr = optimizer.param_groups[0]["lr"]
        if not 0 <= self.min_lr <= max_lr:
            raise ValueError("min_lr must be between zero and the configured LR.")
        decay_steps = max(1, self.total_updates - self.warmup_updates - 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=self.min_lr,
        )
        if self.warmup_updates == 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_updates,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_updates],
        )


SchedulerConfig = (
    CosineSchedulerConfig
    | CosineWithTailSchedulerConfig
    | CosineWithWarmupConfig
    | WarmupCosineUpdatesConfig
)
