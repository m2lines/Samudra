from typing import Literal

import torch
from pydantic import BaseModel


class CosineSchedulerConfig(BaseModel):
    """Cosine scheduler; see pytorch CosineAnnealingLR."""

    type: Literal["cosine"] = "cosine"

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


class CosineWithTailSchedulerConfig(BaseModel):
    """Cosine scheduler which goes to tail_lr for the last tail_epochs."""

    type: Literal["cosine_with_tail"] = "cosine_with_tail"

    tail_lr: float
    tail_epochs: int = 10

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - self.tail_epochs, eta_min=self.tail_lr
        )
        tail = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=self.tail_lr / optimizer.param_groups[0]["lr"],
            total_iters=self.tail_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[cosine, tail], milestones=[epochs - self.tail_epochs]
        )


class CosineWithWarmupConfig(BaseModel):
    """Cosine scheduler which goes to head_lr for the first head_epochs."""

    type: Literal["cosine_with_warmup"] = "cosine_with_warmup"

    warmup_lr: float = 1e-6
    warmup_epochs: int = 5

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.warmup_lr / optimizer.param_groups[0]["lr"],
            end_factor=1.0,  # Reaches full LR
            total_iters=self.warmup_epochs,
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - self.warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )


SchedulerConfig = (
    CosineSchedulerConfig | CosineWithTailSchedulerConfig | CosineWithWarmupConfig
)
