# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Concrete optimizer, EMA, sampling and replay state for the baseline recipe."""

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from samudra.observations.replay import ReplayPool
from samudra.utils.ema import EMATracker


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TrainingState:
    def __init__(
        self,
        model,
        optimizer,
        sources,
        seed=1729,
        *,
        ema_decay=0.5 ** (1 / 100),
        faster_decay_at_start=False,
        replay_per_month=2,
        replay_max_age=250,
    ):
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
        self.best_score = float("inf")
        self.pending_plan: Any = None
        self.rank_rng: Any = None
        self.exposure = {
            name: {"updates": 0, "fresh": 0, "replay": 0} for name in sources
        }

    def new_ema(self):
        return EMATracker(
            self.model,
            decay=self.ema_decay,
            faster_decay_at_start=self.faster_decay_at_start,
        )

    @contextmanager
    def averaged_model(self):
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model)
        try:
            yield self.model
        finally:
            self.ema.restore(self.model.parameters())

    def save(self, path, manifest):
        atomic_save(
            {
                "manifest": manifest,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ema": self.ema.get_state(True),
                "updates": self.updates,
                "cooled": self.cooled,
                "best_score": self.best_score,
                "exposure": self.exposure,
                "sampler_rng": self.rng.bit_generator.state,
                "pools": {name: pool.state_dict() for name, pool in self.pools.items()},
                "pending_plan": self.pending_plan,
                "rank_rng": self.rank_rng,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": [torch.cuda.get_rng_state()]
                if torch.cuda.is_available()
                else [],
                "cuda_rng_scope": "current",
            },
            path,
        )

    def restore(self, path, manifest, *, fork=False):
        # Only load trusted, locally produced checkpoints (optimizer/RNG objects).
        saved = torch.load(path, map_location="cpu", weights_only=False)
        parent, current = saved["manifest"], manifest
        if fork:
            # Forks may only change the duration and snapshot schedule. Data,
            # architecture, optimizer, sources, loss and code remain identical.
            import copy

            parent, current = copy.deepcopy(parent), copy.deepcopy(current)
            for item in (parent, current):
                for key in ("main_updates", "cooldown_updates", "milestone_updates"):
                    item["config"].pop(key, None)
            if saved["cooled"] or saved["updates"] > manifest["config"]["main_updates"]:
                raise ValueError(
                    "Fork requires an uncooled checkpoint before the new main endpoint"
                )
        if parent != current:
            raise ValueError("Checkpoint code/data/recipe manifest differs")
        if set(saved["pools"]) != set(self.pools):
            raise ValueError("Checkpoint source mixture differs")
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.ema = EMATracker.from_state(saved["ema"], self.model)
        for key in ("updates", "cooled", "best_score", "exposure"):
            setattr(self, key, saved[key])
        if fork:
            self.best_score = float("inf")
        for name, pool in self.pools.items():
            pool.load_state_dict(saved["pools"][name])
        self.pending_plan = saved.get("pending_plan")
        self.rank_rng = saved.get("rank_rng")
        self.rng.bit_generator.state = saved["sampler_rng"]
        rng = saved
        if self.rank_rng is not None:
            distributed = torch.distributed.is_initialized()
            world_size = torch.distributed.get_world_size() if distributed else 1
            rank = torch.distributed.get_rank() if distributed else 0
            if len(self.rank_rng) != world_size:
                raise ValueError("Checkpoint distributed world size differs")
            rng = self.rank_rng[rank]
        random.setstate(rng["python_rng"])
        np.random.set_state(rng["numpy_rng"])
        torch.set_rng_state(rng["torch_rng"])
        if rng["cuda_rng"]:
            if not torch.cuda.is_available():
                raise ValueError("CUDA checkpoint resume requires CUDA")
            if rng.get("cuda_rng_scope") == "current":
                torch.cuda.set_rng_state(rng["cuda_rng"][0])
            else:
                torch.cuda.set_rng_state_all(rng["cuda_rng"])
