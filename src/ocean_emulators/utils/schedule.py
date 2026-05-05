from typing import Literal

import torch
from pydantic import BaseModel


class EpochMultiplierScheduler:
    """Wrap a scheduler and apply stage-wise LR multipliers by epoch."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        multipliers: list[float],
        transition_epochs: list[int],
        *,
        current_epoch: int = 1,
        apply_initial_multiplier: bool = True,
    ) -> None:
        assert len(multipliers) == len(transition_epochs) + 1
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.multipliers = [float(m) for m in multipliers]
        self.transition_epochs = transition_epochs
        self.current_epoch = current_epoch
        self.applied_multiplier = 1.0

        if apply_initial_multiplier:
            self._apply_multiplier_for_epoch(self.current_epoch)

    def _multiplier_for_epoch(self, epoch: int) -> float:
        for i, epoch_to_transition in enumerate(self.transition_epochs):
            if epoch < epoch_to_transition:
                return self.multipliers[i]
        return self.multipliers[-1]

    def _scale_optimizer_lr(self, factor: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor

    def _apply_multiplier_for_epoch(self, epoch: int) -> None:
        target_multiplier = self._multiplier_for_epoch(epoch)
        factor = target_multiplier / self.applied_multiplier
        self._scale_optimizer_lr(factor)
        self.applied_multiplier = target_multiplier
        self.current_epoch = epoch

    def step(self) -> None:
        if self.applied_multiplier != 1.0:
            self._scale_optimizer_lr(1.0 / self.applied_multiplier)
            self.applied_multiplier = 1.0

        if self.scheduler is not None:
            self.scheduler.step()

        self._apply_multiplier_for_epoch(self.current_epoch + 1)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "wrapped_scheduler": "epoch_multiplier",
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "current_epoch": self.current_epoch,
            "applied_multiplier": self.applied_multiplier,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict.get("wrapped_scheduler") == "epoch_multiplier":
            scheduler_state = state_dict.get("scheduler_state")
            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            self.current_epoch = state_dict["current_epoch"]
            self.applied_multiplier = state_dict["applied_multiplier"]
            return

        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict)
            self.current_epoch = int(state_dict.get("last_epoch", 0)) + 1
        else:
            self.current_epoch = 1
        self.applied_multiplier = 1.0
        self._apply_multiplier_for_epoch(self.current_epoch)


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


SchedulerConfig = (
    CosineSchedulerConfig | CosineWithTailSchedulerConfig | CosineWithWarmupConfig
)
