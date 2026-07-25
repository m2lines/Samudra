# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import torch
from pydantic import BaseModel, Field


class AdamOptimizerConfig(BaseModel):
    type: Literal["adam"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)

    def build(
        self, model: torch.nn.Module, learning_rate: float
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.weight_decay,
        )


class MuonOptimizerConfig(BaseModel):
    """Muon for matrices plus AdamW for all unsupported parameter shapes."""

    type: Literal["muon"] = "muon"
    momentum: float = Field(default=0.95, ge=0.0)
    weight_decay: float = Field(default=0.15, ge=0.0)
    auxiliary_betas: tuple[float, float] = (0.9, 0.95)

    def build(
        self, model: torch.nn.Module, learning_rate: float
    ) -> torch.optim.Optimizer:
        matrix_params = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim == 2
        ]
        auxiliary_params = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim != 2
        ]
        if not matrix_params:
            raise ValueError("Muon requires at least one trainable matrix parameter.")

        optimizers: list[torch.optim.Optimizer] = [
            torch.optim.Muon(
                matrix_params,
                lr=learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
        ]
        if auxiliary_params:
            optimizers.append(
                torch.optim.AdamW(
                    auxiliary_params,
                    lr=learning_rate,
                    betas=self.auxiliary_betas,
                    weight_decay=self.weight_decay,
                )
            )
        return CompositeOptimizer(optimizers)


OptimizerConfig = AdamOptimizerConfig | MuonOptimizerConfig


class CompositeOptimizer(torch.optim.Optimizer):
    """Expose several disjoint optimizers as one scheduler/checkpoint target."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer.")
        self.optimizers = optimizers
        parameters = [
            parameter
            for optimizer in optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        super().__init__(parameters, defaults={})
        self.param_groups = [
            group for optimizer in optimizers for group in optimizer.param_groups
        ]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for index, optimizer in enumerate(self.optimizers):
            current = optimizer.step(closure if index == 0 else None)
            if current is not None:
                loss = current
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "composite_optimizer": [
                optimizer.state_dict() for optimizer in self.optimizers
            ]
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        states = state_dict.get("composite_optimizer")
        if states is None or len(states) != len(self.optimizers):
            raise ValueError(
                "Checkpoint optimizer state does not match the composite optimizer."
            )
        for optimizer, state in zip(self.optimizers, states):
            optimizer.load_state_dict(state)
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
