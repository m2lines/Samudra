# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from samudra.config import TrainConfig
from samudra.search.config import AdaptiveDataParallelResourceConfig
from samudra.search.resources import plan_candidate_resources


def train_config(*, batch_size: int, accumulation_steps: int = 1) -> TrainConfig:
    config = TrainConfig.from_yaml_and_cli(
        [str(Path("tests/configs/train_default.yaml").resolve())]
    )
    config.batch_size = batch_size
    config.gradient_accumulation_steps = accumulation_steps
    return config


def policy(*, effective_batch: int = 64) -> AdaptiveDataParallelResourceConfig:
    return AdaptiveDataParallelResourceConfig(
        max_gpus_per_candidate=8,
        effective_global_batch_size=effective_batch,
    )


def test_adaptive_plan_expands_as_candidate_concurrency_falls():
    candidates = {name: train_config(batch_size=2) for name in ("a", "b", "c", "d")}

    crowded = plan_candidate_resources(
        policy(), candidates, gpu_capacity=8, candidate_concurrency=4
    )
    survivors = plan_candidate_resources(
        policy(), {"a": candidates["a"]}, gpu_capacity=8, candidate_concurrency=1
    )

    assert {plan.world_size for plan in crowded.values()} == {2}
    assert {plan.gradient_accumulation_steps for plan in crowded.values()} == {16}
    assert survivors["a"].world_size == 8
    assert survivors["a"].gradient_accumulation_steps == 4
    assert all(
        plan.local_batch_size * plan.world_size * plan.gradient_accumulation_steps == 64
        for plan in [*crowded.values(), *survivors.values()]
    )


def test_incompatible_batch_warns_and_preserves_user_choice():
    candidate = train_config(batch_size=4, accumulation_steps=7)

    with pytest.warns(UserWarning, match="Choose a local batch size"):
        plan = plan_candidate_resources(
            policy(effective_batch=48), {"model": candidate}, gpu_capacity=8
        )["model"]

    assert candidate.batch_size == 4
    assert plan.local_batch_size == 4
    assert plan.world_size == 4
    assert plan.gradient_accumulation_steps == 3
    assert plan.effective_global_batch_size == 48


def test_irreconcilable_batch_warns_and_disables_adaptation():
    candidate = train_config(batch_size=3, accumulation_steps=7)

    with pytest.warns(UserWarning) as caught:
        plan = plan_candidate_resources(
            policy(effective_batch=64), {"model": candidate}, gpu_capacity=8
        )["model"]

    assert len(caught) == 1
    assert plan.world_size == 1
    assert plan.local_batch_size == 3
    assert plan.gradient_accumulation_steps == 7
    assert plan.effective_global_batch_size == 21


def test_adaptive_plan_requires_auto_backend():
    candidate = train_config(batch_size=2)
    candidate.backend = "cuda"

    with pytest.raises(ValueError, match="requires backend='auto'"):
        plan_candidate_resources(policy(), {"model": candidate}, gpu_capacity=8)
