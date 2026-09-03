# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic GPU and batch planning for search candidates."""

from __future__ import annotations

import warnings

from samudra.config import TrainConfig
from samudra.search.config import AdaptiveDataParallelResourceConfig, ResourceConfig
from samudra.search.state import CandidateResourceState


def plan_candidate_resources(
    policy: ResourceConfig,
    candidates: dict[str, TrainConfig],
    *,
    gpu_capacity: int,
    candidate_concurrency: int | None = None,
    force_single_gpu: bool = False,
) -> dict[str, CandidateResourceState]:
    """Resolve reproducible world sizes without changing configured batch sizes."""
    if not candidates:
        return {}
    if not isinstance(policy, AdaptiveDataParallelResourceConfig):
        return {
            name: _configured_plan(config, world_size=1)
            for name, config in candidates.items()
        }
    incompatible_backends = [
        name for name, config in candidates.items() if config.backend != "auto"
    ]
    if incompatible_backends:
        raise ValueError(
            "adaptive_data_parallel requires backend='auto' so a candidate can "
            "move between single-process and distributed rungs; incompatible "
            f"candidates: {', '.join(incompatible_backends)}"
        )

    requested = 1
    if not force_single_gpu:
        concurrent = candidate_concurrency or len(candidates)
        per_candidate = max(1, gpu_capacity // concurrent)
        requested = max(
            size
            for size in policy.allowed_world_sizes
            if size <= min(per_candidate, policy.max_gpus_per_candidate)
        )

    plans: dict[str, CandidateResourceState] = {}
    for name, config in candidates.items():
        compatible = [
            size
            for size in policy.allowed_world_sizes
            if size <= requested
            and policy.effective_global_batch_size % (config.batch_size * size) == 0
        ]
        if not compatible:
            warnings.warn(
                f"Candidate {name!r} cannot realize effective_global_batch_size="
                f"{policy.effective_global_batch_size} with its configured "
                f"batch_size={config.batch_size}, even on one GPU. Its batch and "
                "gradient accumulation settings are unchanged and adaptive "
                "scaling is disabled for this candidate. Choose a batch size "
                "that divides effective_global_batch_size to enable it.",
                UserWarning,
                stacklevel=2,
            )
            plans[name] = _configured_plan(config, world_size=1)
            continue

        world_size = max(compatible)
        if world_size != requested:
            warnings.warn(
                f"Adaptive data parallelism requested world_size={requested} "
                f"for candidate {name!r}, but effective_global_batch_size="
                f"{policy.effective_global_batch_size} is not divisible by "
                f"batch_size={config.batch_size} times that world size. Keeping "
                f"the configured batch size and using world_size={world_size}. "
                "Choose a local batch size that divides "
                "effective_global_batch_size / world_size to use the requested "
                "GPUs.",
                UserWarning,
                stacklevel=2,
            )
        denominator = config.batch_size * world_size
        accumulation = policy.effective_global_batch_size // denominator
        plans[name] = CandidateResourceState(
            world_size=world_size,
            local_batch_size=config.batch_size,
            gradient_accumulation_steps=accumulation,
            effective_global_batch_size=policy.effective_global_batch_size,
        )
    return plans


def _configured_plan(config: TrainConfig, *, world_size: int) -> CandidateResourceState:
    return CandidateResourceState(
        world_size=world_size,
        local_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        effective_global_batch_size=(
            config.batch_size * config.gradient_accumulation_steps * world_size
        ),
    )
