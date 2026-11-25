# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Read-only spike diagnostics and an optional loss guard for update training.

The probe reads gradients and parameters around each optimizer step and never
writes to them, consumes no random numbers, and adds no autograd edges, so a run
with the probe enabled matches one without it exactly. The loss guard does change
training: it is an intervention, configured and recorded as a scientific choice.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from samudra.observations.config import LossGuardConfig, ProbeConfig
from samudra.observations.state import atomic_save


def parameter_group(name: str) -> str:
    """Group parameters by network and component, e.g. ``evolution.net.0``.

    Task adapters keep their task name (``initializer.input_adapters.llc``), so
    gradients from each source's adapter are visible separately.
    """
    return ".".join(name.split(".")[:3])


class LossGuard:
    """Per-source running median of accepted losses; decides whether to skip.

    State is a plain dict so TrainingState checkpoints it and exact resume holds.
    """

    def __init__(self, cfg: LossGuardConfig, state: dict[str, Any]) -> None:
        self.cfg = cfg
        self.state = state
        state.setdefault("history", {})
        state.setdefault("consecutive", {})

    def skip(self, source: str, loss: float) -> bool:
        history = self.state["history"].setdefault(source, [])
        consecutive = self.state["consecutive"]
        if len(history) >= self.cfg.window and loss > self.cfg.factor * (
            statistics.median(history)
        ):
            consecutive[source] = consecutive.get(source, 0) + 1
            if consecutive[source] >= self.cfg.max_consecutive:
                raise FloatingPointError(
                    f"{consecutive[source]} consecutive {source} updates exceeded "
                    f"{self.cfg.factor}x the running median loss; the weights, "
                    "not a batch, are producing the high loss"
                )
            return True
        consecutive[source] = 0
        history.append(loss)
        del history[: -self.cfg.window]
        return False


class Probe:
    """Collect per-group gradient and update norms on rank zero."""

    def __init__(self, cfg: ProbeConfig, model: torch.nn.Module, output: Path) -> None:
        self.cfg = cfg
        self.path = output / "probe.jsonl"
        self.weights = output / "probe-weights"
        groups: dict[str, list[torch.nn.Parameter]] = defaultdict(list)
        for name, parameter in model.named_parameters():
            groups[parameter_group(name)].append(parameter)
        self.groups = dict(sorted(groups.items()))
        self.before: list[torch.Tensor] | None = None

    def _norms(self, tensors: list[list[torch.Tensor]]) -> list[float]:
        values = torch.stack(
            [
                torch.stack([t.detach().float().square().sum() for t in group]).sum()
                if group
                else torch.zeros((), device=self._device)
                for group in tensors
            ]
        )
        return values.sqrt().cpu().tolist()

    @property
    def _device(self) -> torch.device:
        return next(iter(self.groups.values()))[0].device

    def before_step(self) -> dict[str, float]:
        """Gradient norms by group (pre-clipping), and a copy of the weights."""
        grads = [
            [p.grad for p in params if p.grad is not None]
            for params in self.groups.values()
        ]
        record = {
            f"grad_norm/{name}": value
            for name, value in zip(self.groups, self._norms(grads), strict=True)
        }
        self.before = [
            p.detach().clone() for params in self.groups.values() for p in params
        ]
        return record

    def after_step(self) -> dict[str, float]:
        """Update size relative to weight size, by group."""
        if self.before is None:
            raise RuntimeError("Probe.after_step requires before_step")
        before = iter(self.before)
        deltas, weights = [], []
        for params in self.groups.values():
            old = [next(before) for _ in params]
            deltas.append([p.detach() - b for p, b in zip(params, old, strict=True)])
            weights.append(old)
        self.before = None
        record = {}
        for name, delta, weight in zip(
            self.groups, self._norms(deltas), self._norms(weights), strict=True
        ):
            record[f"update_norm/{name}"] = delta
            record[f"relative_update/{name}"] = delta / weight if weight else 0.0
        return record

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=True) + "\n")

    def save_weights(
        self, updates: int, model: torch.nn.Module, ema: dict[str, Any]
    ) -> None:
        if self.cfg.weights_every is None or updates % self.cfg.weights_every:
            return
        self.weights.mkdir(exist_ok=True)
        atomic_save(
            {"updates": updates, "model": model.state_dict(), "ema": ema},
            self.weights / f"{updates:06d}.pt",
        )
