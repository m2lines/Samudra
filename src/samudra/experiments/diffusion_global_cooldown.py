# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Cool down the final updates of the fixed 8k/8k global experiment."""

import math
from dataclasses import dataclass

from samudra.experiments.task_schedule import TaskSchedule


@dataclass(frozen=True)
class CooldownSchedule:
    """Retain the original task schedule and cool down within its fixed budget."""

    updates: int = 2000
    minimum_lr: float = 1e-6

    def __post_init__(self):
        if not 2 <= self.updates <= 16000:
            raise ValueError("Cooldown length must lie between two and 16000 updates")
        if not 0 < self.minimum_lr < 1e-4:
            raise ValueError("Cooldown minimum LR must lie between zero and 1e-4")

    @property
    def total(self):
        return 16000

    @property
    def start(self):
        return self.total - self.updates

    def counts(self, completed):
        return TaskSchedule(8000, 8000, "mixed").counts(completed)

    def task(self, index):
        return TaskSchedule(8000, 8000, "mixed").task(index)

    def learning_rate(self, index):
        """LR for a zero-based update, including exact initial and final rates."""
        if not 0 <= index < self.total:
            raise ValueError("LR requested outside cooldown")
        fraction = max(0, index - self.start) / (self.updates - 1)
        return (
            self.minimum_lr
            + (1e-4 - self.minimum_lr) * (1 + math.cos(math.pi * fraction)) / 2
        )


def validate_parent(saved, expected_contract, schedule):
    """Require an original checkpoint at or before cooldown with real moments."""
    if saved["contract"] != expected_contract:
        raise ValueError("Cooldown parent contract differs")
    if not 0 <= saved["step"] <= schedule.start:
        raise ValueError("Parent must precede or coincide with cooldown start")
    if saved["counts"] != schedule.counts(saved["step"]):
        raise ValueError("Cooldown parent exposure differs")
    if not all(key in saved for key in ("optimizer", "cpu_rng", "cuda_rng")):
        raise ValueError("Cooldown requires optimizer and RNG state, not just weights")
    groups = saved["optimizer"]["param_groups"]
    if not groups or any(group["lr"] != 1e-4 for group in groups):
        raise ValueError("Cooldown parent optimizer LR differs")
