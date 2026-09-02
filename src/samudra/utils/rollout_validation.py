# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
from typing import Any

import numpy as np


def should_run_on_epoch_freq(epoch: int, frequency: int) -> bool:
    """Return whether a periodic, 1-based epoch counter is due to run."""
    if epoch < 1:
        raise ValueError(f"Epoch must be >= 1, got {epoch}")
    if frequency < 1:
        raise ValueError(f"Frequency must be >= 1, got {frequency}")
    return (epoch - 1) % frequency == 0


def should_log_validation_images(epoch: int, frequency: int) -> bool:
    """Return whether validation images should be logged for this epoch."""
    if epoch < 1:
        raise ValueError(f"Epoch must be >= 1, got {epoch}")
    if frequency < 1:
        raise ValueError(
            f"Validation image log frequency must be >= 1, got {frequency}"
        )
    return should_run_on_epoch_freq(epoch, frequency)


def resolve_rollout_validation_steps(requested_steps: int, available_steps: int) -> int:
    """Resolve requested rollout validation steps against available val steps."""
    if available_steps <= 0:
        return 0
    if requested_steps < 0 or requested_steps > available_steps:
        return available_steps
    return requested_steps


def _elapsed_days(start: Any, end: Any) -> float:
    delta = end - start
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "D"))
    if isinstance(delta, datetime.timedelta) or hasattr(delta, "total_seconds"):
        return float(delta.total_seconds() / (24 * 60 * 60))
    return float(delta)


@dataclasses.dataclass(frozen=True)
class RolloutValidationSpec:
    """Resolved horizon for one rollout validation metric stream.

    The rollout advances ``model_steps`` autoregressive model calls and records
    metrics over the corresponding ``target_timesteps`` raw timesteps under
    ``label``.
    """

    label: str
    model_steps: int
    target_timesteps: int

    @classmethod
    def from_model_steps(
        cls,
        *,
        requested_steps: int,
        available_steps: int,
        hist: int,
    ) -> "RolloutValidationSpec":
        model_steps = resolve_rollout_validation_steps(
            requested_steps,
            available_steps,
        )
        return cls(
            label="steps",
            model_steps=model_steps,
            target_timesteps=model_steps * (hist + 1),
        )

    @classmethod
    def from_day_horizon(
        cls,
        *,
        days: int,
        start_time: Any,
        target_times: Any,
        hist: int,
    ) -> "RolloutValidationSpec":
        if days <= 0:
            raise ValueError(f"rollout_validation.days must be positive, got {days}")

        target_elapsed_days = np.asarray(
            [_elapsed_days(start_time, target_time) for target_time in target_times]
        )
        if target_elapsed_days.size == 0:
            raise ValueError("Rollout validation dataset has no target timesteps")

        max_days = float(target_elapsed_days[-1])
        if max_days < days:
            raise ValueError(
                f"rollout_validation.days includes {days}, but val_time only covers "
                f"{max_days:.2f} forecast days"
            )

        target_timesteps = int(np.count_nonzero(target_elapsed_days <= days))
        model_steps = target_timesteps // (hist + 1)
        target_timesteps = model_steps * (hist + 1)
        if model_steps < 1:
            raise ValueError(
                f"rollout_validation.days={days} is shorter than one model step "
                f"for hist={hist}"
            )

        return cls(
            label=f"{days}d",
            model_steps=model_steps,
            target_timesteps=target_timesteps,
        )
