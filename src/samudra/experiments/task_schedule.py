# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exact task exposure and reproducible per-task samples for joint training."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class TaskSchedule:
    om4_updates: int
    observation_updates: int
    ordering: str

    def __post_init__(self):
        if self.om4_updates < 0 or self.observation_updates < 1:
            raise ValueError("Require nonnegative OM4 and positive observation counts")
        if self.ordering not in ("scratch", "sequential", "mixed"):
            raise ValueError("Unknown task ordering")
        if self.ordering == "scratch" and self.om4_updates:
            raise ValueError("Scratch must have zero OM4 updates")
        # The quadratic cumulative schedule's largest slope is 7/4 times
        # the overall observation fraction; one optimizer step handles one task.
        if self.ordering == "mixed" and 7 * self.observation_updates > 4 * self.total:
            raise ValueError("Mixed schedule requires observation fraction <= 4/7")

    @property
    def total(self):
        return self.om4_updates + self.observation_updates

    def counts(self, completed):
        """Counts after completed updates; integer arithmetic avoids drift."""
        if not 0 <= completed <= self.total:
            raise ValueError("Completed updates outside schedule")
        if self.ordering == "scratch":
            observation = completed
        elif self.ordering == "sequential":
            observation = max(0, completed - self.om4_updates)
        else:
            # Cumulative observation exposure M * (f/4 + 3*f*f/4).
            observation = (
                self.observation_updates
                * completed
                * (self.total + 3 * completed)
                // (4 * self.total * self.total)
            )
        return {"om4": completed - observation, "observation": observation}

    def task(self, index):
        """Task for zero-based global optimizer update index."""
        if not 0 <= index < self.total:
            raise ValueError("Update index outside schedule")
        before, after = self.counts(index), self.counts(index + 1)
        return "observation" if after["observation"] > before["observation"] else "om4"


@lru_cache(maxsize=8)
def _permutation(size, seed, epoch):
    # Task-specific stream calls this with a separate seed. No global RNG state
    # or sequence of calls affects the result, including after process restart.
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(
        size
    )


def sample_indices(size, seed, completed_task_updates, accumulate):
    """Stable epoch-shuffled sample stream indexed by task exposure, not time."""
    if size < 1 or seed < 0 or completed_task_updates < 0 or accumulate < 1:
        raise ValueError("Invalid sample stream arguments")
    first = completed_task_updates * accumulate
    return [
        int(_permutation(size, seed, position // size)[position % size])
        for position in range(first, first + accumulate)
    ]
