from typing import Literal

import torch
from pydantic import BaseModel


class CosineScheduler(BaseModel):
    type: Literal["cosine"] = "cosine"

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


class CosineWithTailScheduler(BaseModel):
    type: Literal["cosine_with_tail"] = "cosine_with_tail"

    tail_lr: float
    tail_epochs: int = 10

    def build(
        self, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - self.tail_epochs, eta_min=self.tail_lr
        )
        tail = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[cosine, tail], milestones=[epochs - self.tail_epochs]
        )


Scheduler = CosineScheduler | CosineWithTailScheduler
