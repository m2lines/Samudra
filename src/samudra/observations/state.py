# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Concrete optimizer, EMA, sampling and replay state for the baseline recipe."""

import json
import os
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from samudra.observations.replay import ReplayPool
from samudra.utils.ema import EMATracker

if TYPE_CHECKING:
    from samudra.observations.config import ObservationTrainConfig


def atomic_save(value: object, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(value: object, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TrainingState[ModelT: torch.nn.Module]:
    def __init__(
        self,
        model: ModelT,
        optimizer: torch.optim.Optimizer,
        sources: Sequence[str],
        seed: int = 1729,
        *,
        config: "ObservationTrainConfig | None" = None,
        ema_decay: float = 0.5 ** (1 / 100),
        faster_decay_at_start: bool = False,
        replay_per_month: int = 2,
        replay_max_age: int = 250,
    ) -> None:
        self.model, self.optimizer = model, optimizer
        self.ema_decay = ema_decay
        self.faster_decay_at_start = faster_decay_at_start
        self.ema = self.new_ema()
        self.rng = np.random.default_rng(seed)
        self.pools = {
            name: ReplayPool(
                name, seed + i + 1, per_month=replay_per_month, max_age=replay_max_age
            )
            for i, name in enumerate(sources)
        }
        self.updates = 0
        self.cooled = False
        self.config = config
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.build_scheduler()
        self.best_score = float("inf")
        self.exposure = {
            name: {"updates": 0, "fresh": 0, "replay": 0} for name in sources
        }

    def build_scheduler(self) -> None:
        """Build the shared scheduler for the current phase, measured in updates."""
        if self.config is None:
            self.scheduler = None
            return
        cfg = self.config
        recipe = cfg.cooldown_scheduler if self.cooled else cfg.scheduler
        steps = cfg.cooldown_updates if self.cooled else cfg.main_updates
        self.scheduler = recipe.build(self.optimizer, steps) if recipe else None

    def new_ema(self) -> EMATracker:
        return EMATracker(
            self.model,
            decay=self.ema_decay,
            faster_decay_at_start=self.faster_decay_at_start,
        )

    @contextmanager
    def averaged_model(self) -> Iterator[ModelT]:
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model)
        try:
            yield self.model
        finally:
            self.ema.restore(self.model.parameters())

    def save(self, path: str | Path, manifest: dict[str, Any]) -> None:
        atomic_save(
            {
                "manifest": manifest,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "ema": self.ema.get_state(True),
                "updates": self.updates,
                "cooled": self.cooled,
                "best_score": self.best_score,
                "exposure": self.exposure,
                "sampler_rng": self.rng.bit_generator.state,
                "pools": {name: pool.state_dict() for name, pool in self.pools.items()},
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
            path,
        )

    def restore(self, path: str | Path, manifest: dict[str, Any]) -> None:
        # Only load trusted, locally produced checkpoints (optimizer/RNG objects).
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["manifest"] != manifest:
            raise ValueError("Checkpoint code/data/recipe manifest differs")
        if set(saved["pools"]) != set(self.pools):
            raise ValueError("Checkpoint source mixture differs")
        self.model.load_state_dict(saved["model"])
        self.cooled = saved["cooled"]
        # Rebuild the correct phase before restoring optimizer LR and scheduler
        # position. Cooldown's base is the LR at the phase boundary.
        for group, stored in zip(
            self.optimizer.param_groups, saved["optimizer"]["param_groups"], strict=True
        ):
            group["lr"] = stored.get("initial_lr", stored["lr"])
            group["initial_lr"] = group["lr"]
        self.build_scheduler()
        self.optimizer.load_state_dict(saved["optimizer"])
        if (self.scheduler is None) != (saved["scheduler"] is None):
            raise ValueError("Checkpoint scheduler differs from configured phase")
        if self.scheduler is not None:
            self.scheduler.load_state_dict(saved["scheduler"])
        self.ema = EMATracker.from_state(saved["ema"], self.model)
        for key in ("updates", "cooled", "best_score", "exposure"):
            setattr(self, key, saved[key])
        for name, pool in self.pools.items():
            pool.load_state_dict(saved["pools"][name])
        self.rng.bit_generator.state = saved["sampler_rng"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"]:
            if not torch.cuda.is_available():
                raise ValueError("CUDA checkpoint resume requires CUDA")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
