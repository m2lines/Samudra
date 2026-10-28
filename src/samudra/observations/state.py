# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Concrete optimizer, EMA, sampling and replay state for the baseline recipe."""

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

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
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
            path,
        )

    def restore(self, path, manifest):
        # Only load trusted, locally produced checkpoints (optimizer/RNG objects).
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["manifest"] != manifest:
            raise ValueError("Checkpoint code/data/recipe manifest differs")
        if set(saved["pools"]) != set(self.pools):
            raise ValueError("Checkpoint source mixture differs")
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
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
