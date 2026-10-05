# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""A separately recorded cooldown after the fixed 8k/8k global experiment."""

import math
from dataclasses import dataclass

from samudra.experiments.task_schedule import TaskSchedule


@dataclass(frozen=True)
class CooldownSchedule:
    """Continue sample counters at the original schedule's terminal 7:1 mix."""

    updates: int = 2000
    minimum_lr: float = 1e-6

    def __post_init__(self):
        if self.updates < 8 or self.updates % 8:
            raise ValueError("Cooldown updates must be a positive multiple of eight")
        if not 0 < self.minimum_lr < 1e-4:
            raise ValueError("Cooldown minimum LR must lie between zero and 1e-4")

    @property
    def total(self):
        return 16000 + self.updates

    def counts(self, completed):
        if not 0 <= completed <= self.total:
            raise ValueError("Completed updates outside cooldown schedule")
        if completed <= 16000:
            return TaskSchedule(8000, 8000, "mixed").counts(completed)
        extra = completed - 16000
        observation = 7 * extra // 8
        return dict(om4=8000 + extra - observation, observation=8000 + observation)

    def task(self, index):
        if not 0 <= index < self.total:
            raise ValueError("Update index outside cooldown schedule")
        before, after = self.counts(index), self.counts(index + 1)
        return "observation" if after["observation"] > before["observation"] else "om4"

    def learning_rate(self, index):
        """LR for a zero-based update, including exact initial and final rates."""
        if not 16000 <= index < self.total:
            raise ValueError("LR requested outside cooldown")
        fraction = (index - 16000) / (self.updates - 1)
        return (
            self.minimum_lr
            + (1e-4 - self.minimum_lr) * (1 + math.cos(math.pi * fraction)) / 2
        )


def validate_parent(saved, expected_contract):
    """Require a completed, qualified original run with real optimizer state."""
    if saved["contract"] != expected_contract:
        raise ValueError("Cooldown parent contract differs")
    if saved["step"] != 16000 or saved["counts"] != dict(om4=8000, observation=8000):
        raise ValueError("Cooldown requires the completed 8k/8k endpoint")
    if not all(key in saved for key in ("optimizer", "cpu_rng", "cuda_rng")):
        raise ValueError("Cooldown requires optimizer and RNG state, not just weights")
    groups = saved["optimizer"]["param_groups"]
    if not groups or any(group["lr"] != 1e-4 for group in groups):
        raise ValueError("Cooldown parent optimizer LR differs")
