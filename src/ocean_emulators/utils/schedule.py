from typing import Literal

import torch
from pydantic import BaseModel


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


SchedulerConfig = CosineSchedulerConfig | CosineWithTailSchedulerConfig
