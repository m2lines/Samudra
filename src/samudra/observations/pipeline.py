# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded update prefetch with one global replay pool and DDP example sharding.

Rank zero owns sample planning and replay storage. Every update has the same
global examples regardless of world size. Its successor is planned before
computing it, so data reads overlap compute without racing replay insertion.
The pending plan is checkpointed; restoring never resamples the next update.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from samudra.models.surface_initialized import SurfaceInitializedModel
from samudra.observations import training as baseline
from samudra.observations.config import ObservationLossConfig, ObservationTrainConfig
from samudra.observations.replay import ReplayEntry
from samudra.observations.sample_types import ForecastSample
from samudra.observations.samples import Samples
from samudra.observations.state import TrainingState, write_json
from samudra.utils.device import autocast
from samudra.utils.train import CheckpointPaths


class PlannedExample(TypedDict):
    origin: str | int
    month: int
    coverage: str | int | None
    seed: int
    entry: ReplayEntry | None


class UpdatePlan(TypedDict):
    update: int
    source: str
    examples: list[PlannedExample]
    sampler_rng_before: Mapping[str, Any]
    pool_rng_before: dict[str, Mapping[str, Any]]
    pool_entries_before: dict[str, dict[int, list[ReplayEntry]]]


class ExampleMetadata(TypedDict):
    origin: str | int
    month: int
    coverage: str | int | None
    seed: int
    cached_shape: tuple[int, ...] | None
    cached_dtype: torch.dtype | None
    start_lead: int
    origin_update: int


class UpdateMetadata(TypedDict):
    update: int
    source: str
    examples: list[ExampleMetadata]


class LocalExample(ExampleMetadata):
    index: int
    cached: torch.Tensor | None  # [batch, state_history, channel, y, x], on CPU


class LoadedExample(TypedDict):
    sample: ForecastSample
    auxiliary: ForecastSample
    coverage: torch.Tensor | None  # [batch, history+forecast, surface_channel, y, x]
    cached: torch.Tensor | None  # [batch, state_history, physical_channel, y, x]
    seed: int


def plan_update(
    cfg: ObservationTrainConfig,
    state: TrainingState[SurfaceInitializedModel],
    sources: baseline.Sources,
    update: int,
) -> UpdatePlan:
    """Advance only the global sampling RNG; never load data or run the model."""
    source = cfg.source_at(update)
    sampler_rng_before = deepcopy(state.rng.bit_generator.state)
    pool_rng_before = {
        name: deepcopy(pool.rng.bit_generator.state)
        for name, pool in state.pools.items()
    }
    pool_entries_before = {
        name: {month: list(entries) for month, entries in pool.entries.items()}
        for name, pool in state.pools.items()
    }
    if state.replan_pool_entries is not None:
        planning_pool = state.replan_pool_entries[source]
    else:
        planning_pool = pool_entries_before[source]
    examples: list[PlannedExample] = []
    for index in range(cfg.gradient_accumulation_steps):
        pool = state.pools[source]
        committed_entries = pool.entries
        try:
            pool.entries = planning_pool
            entry = pool.draw(update) if index >= cfg.fresh_examples else None
            planning_pool = pool.entries
        finally:
            pool.entries = committed_entries
        origin = (
            sources.fresh_origin(source, state.rng) if entry is None else entry.origin
        )
        coverage = sources.fresh_origin("obs", state.rng) if source != "obs" else None
        examples.append(
            {
                "origin": origin,
                "month": 0 if entry is None else entry.month,
                "coverage": coverage,
                "seed": int(state.rng.integers(2**31)),
                "entry": entry,
            }
        )
    state.replan_pool_entries = None
    return {
        "update": update,
        "source": source,
        "examples": examples,
        "sampler_rng_before": sampler_rng_before,
        "pool_rng_before": pool_rng_before,
        "pool_entries_before": pool_entries_before,
    }


class World:
    """Small metadata and CPU replay transfers, separate from DDP gradient traffic."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.group = dist.new_group(backend="gloo") if self.size > 1 else None

    def broadcast[T](self, value: T) -> T:
        values = [value]
        if self.size > 1:
            dist.broadcast_object_list(values, src=0, group=self.group)
        return values[0]

    def local_plan(
        self, plan: UpdatePlan | None
    ) -> tuple[UpdateMetadata, list[LocalExample]]:
        metadata: UpdateMetadata | None = None
        if self.rank == 0:
            assert plan is not None
            metadata = {
                "update": plan["update"],
                "source": plan["source"],
                "examples": [],
            }
            for example in plan["examples"]:
                entry = example["entry"]
                metadata["examples"].append(
                    {
                        "origin": example["origin"],
                        "month": example["month"],
                        "coverage": example["coverage"],
                        "seed": example["seed"],
                        "cached_shape": tuple(entry.states.shape)
                        if entry is not None
                        else None,
                        "cached_dtype": entry.states.dtype
                        if entry is not None
                        else None,
                        "start_lead": entry.start_lead if entry is not None else 0,
                        "origin_update": entry.origin_update
                        if entry is not None
                        else plan["update"],
                    }
                )
        metadata = self.broadcast(metadata)
        assert metadata is not None
        local: list[LocalExample] = []
        sends, buffers = [], []
        for index, metadata_example in enumerate(metadata["examples"]):
            owner = index % self.size
            cached = None
            if metadata_example["cached_shape"] is not None:
                if self.rank == 0:
                    assert plan is not None
                    entry = plan["examples"][index]["entry"]
                    assert entry is not None
                    cached = entry.states.contiguous()
                    if owner != 0:
                        buffers.append(cached)
                        sends.append(
                            dist.isend(cached, dst=owner, group=self.group, tag=index)
                        )
                elif self.rank == owner:
                    cached = torch.empty(
                        metadata_example["cached_shape"],
                        dtype=metadata_example["cached_dtype"],
                    )
                    dist.recv(cached, src=0, group=self.group, tag=index)
            if owner == self.rank:
                local.append({**metadata_example, "index": index, "cached": cached})
        for send in sends:
            assert send is not None
            send.wait()
        return metadata, local

    def gather_states(self, states: list[torch.Tensor]) -> list[torch.Tensor] | None:
        local = torch.stack(states).detach().cpu().contiguous()
        if self.size == 1:
            return list(local)
        received = (
            [torch.empty_like(local) for _ in range(self.size)]
            if self.rank == 0
            else None
        )
        dist.gather(local, gather_list=received, dst=0, group=self.group)
        if received is None:
            return None
        # Restore global example order, independent of rank count.
        return [
            received[i % self.size][i // self.size]
            for i in range(len(states) * self.size)
        ]

    def mean(self, metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
        keys = sorted(metrics)
        values = torch.stack(
            [torch.as_tensor(metrics[k], device=self.device).float() for k in keys]
        )
        if self.size > 1:
            dist.all_reduce(values)
            values /= self.size
        return dict(zip(keys, values.cpu().tolist(), strict=True))

    def maximum(self, values: Sequence[float]) -> list[float]:
        tensor = torch.tensor(values, dtype=torch.float64)
        if self.size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.group)
        return tensor.tolist()

    def capture_rng(self, state: TrainingState) -> None:
        rng = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng_scope": "current",
            "cuda_rng": [torch.cuda.get_rng_state()]
            if self.device.type == "cuda"
            else [],
        }
        if self.size == 1:
            state.rank_rng = [rng]
        else:
            result = [None] * self.size if self.rank == 0 else None
            dist.gather_object(rng, object_gather_list=result, dst=0, group=self.group)
            if self.rank == 0:
                state.rank_rng = result

    def close(self) -> None:
        if self.group is not None:
            dist.destroy_process_group(self.group)


@dataclass
class LoadedUpdate:
    examples: list[LoadedExample]
    read_seconds: float
    ready: torch.cuda.Event | None = None
    host: list[LoadedExample] | None = None


def map_tensors(
    value: Any,
    function: Callable[[torch.Tensor], torch.Tensor | None],
    memo: dict[int, torch.Tensor | None] | None = None,
) -> Any:
    # The forecast and reconstruction often share the same sample, and forcing
    # has two dictionary aliases. Stage each tensor once per update.
    if memo is None:
        memo = {}
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = function(value)
        return memo[id(value)]
    if isinstance(value, dict):
        # Raw NPZ payloads are not used by the objective; avoid retaining duplicates.
        return {
            k: map_tensors(v, function, memo) for k, v in value.items() if k != "raw"
        }
    if isinstance(value, list):
        return [map_tensors(v, function, memo) for v in value]
    return value


def pin_host_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize broadcast views before pinning their host storage.

    Tensor.pin_memory preserves strides. Expanded validity masks have zero
    strides, so copying into that layout would write repeatedly to one address.
    """
    return tensor.contiguous().pin_memory()


class UpdateLoader:
    """One producer and one in-flight update per rank, including all auxiliary reads."""

    def __init__(
        self,
        sources: baseline.Sources,
        device: str | torch.device,
        *,
        enabled: bool,
        to_device: bool,
    ) -> None:
        if torch.device(sources.obs.device).type != "cpu":
            raise ValueError("Background observation loading requires CPU sources")
        self.sources, self.device = sources, torch.device(device)
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="samudra-obs-prefetch")
            if enabled
            else None
        )
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if to_device and self.device.type == "cuda"
            else None
        )
        self.future: Future[LoadedUpdate] | None = None
        self.request: tuple[str, list[LocalExample]] | None = None

    def read(self, source: str, plan: list[LocalExample]) -> LoadedUpdate:
        started = time.perf_counter()
        examples: list[LoadedExample] = []
        for item in plan:
            sample = self.sources.load(source, item["origin"], item["month"])
            if (
                item["cached"] is not None
                and item["start_lead"] != sample["start_lead"]
            ):
                raise ValueError("Replay state time differs from the planned target")
            auxiliary = self.sources.auxiliary(
                source, item["origin"], item["month"], sample
            )
            coverage = None
            if source != "obs":
                assert item["coverage"] is not None
                coverage = self.sources.load("obs", item["coverage"])["validity"]
            examples.append(
                {
                    "sample": sample,
                    "auxiliary": auxiliary,
                    "coverage": coverage,
                    "cached": item["cached"],
                    "seed": item["seed"],
                }
            )
        result = LoadedUpdate(examples, time.perf_counter() - started)
        if self.stream is not None:
            torch.cuda.set_device(self.device)
            host = map_tensors(examples, pin_host_tensor)
            with torch.cuda.stream(self.stream):
                result.examples = map_tensors(
                    host, lambda t: t.to(self.device, non_blocking=True)
                )
                result.ready = torch.cuda.Event()
                result.ready.record(self.stream)
            result.host = host  # Hold pinned buffers until the ready event completes.
        return result

    def submit(self, source: str, plan: list[LocalExample]) -> None:
        if self.request is not None or self.future is not None:
            raise RuntimeError("Only one pending update is allowed")
        if self.executor is None:
            self.request = (source, plan)
        else:
            self.future = self.executor.submit(self.read, source, plan)

    def consume(self) -> LoadedUpdate:
        if self.future is not None:
            result = self.future.result()
            self.future = None
        else:
            assert self.request is not None
            result = self.read(*self.request)
            self.request = None
        if result.ready is not None:
            result.ready.synchronize()
            stream = torch.cuda.current_stream(self.device)
            map_tensors(result.examples, lambda t: t.record_stream(stream))
            result.host = None
        else:
            result.examples = map_tensors(result.examples, lambda t: t.to(self.device))
        return result

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        if self.future is not None and not self.future.cancelled():
            result = self.future.result()
            if result.ready is not None:
                result.ready.synchronize()
        self.future, self.request = None, None


class Objective(torch.nn.Module):
    def __init__(
        self,
        model: SurfaceInitializedModel,
        data: Samples,
        weights: ObservationLossConfig,
    ) -> None:
        super().__init__()
        self.model, self.data, self.weights = model, data, weights

    def forward(
        self, source: str, item: LoadedExample
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        loss, parts, states = baseline.objective(
            self.model,
            self.data,
            source,
            item["sample"],
            item["auxiliary"],
            item["seed"],
            cached=item["cached"],
            coverage=item["coverage"],
            weights=self.weights,
        )
        return loss, {k: v.detach() for k, v in parts.items()}, states.detach()


def train(
    cfg: ObservationTrainConfig,
    state: TrainingState[SurfaceInitializedModel],
    sources: baseline.Sources,
    output: Path,
    manifest: baseline.Manifest,
    validate: Callable[[SurfaceInitializedModel], dict[str, float]],
    emit: Callable[[baseline.Metrics], None],
    *,
    stop_after: int | None = None,
    stopping: Callable[[], bool] = lambda: False,
) -> None:
    """Use the same global sample plan for sequential accumulation and DDP."""
    device = next(state.model.parameters()).device
    world = World(device)
    if cfg.gradient_accumulation_steps % world.size:
        world.close()
        raise ValueError("Global example count must be divisible by world size")
    per_rank = cfg.gradient_accumulation_steps // world.size
    data = sources.obs.to(device)
    objective: torch.nn.Module = Objective(state.model, data, cfg.loss)
    if world.size > 1:
        objective = DistributedDataParallel(
            objective,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    loader = UpdateLoader(
        sources,
        device,
        enabled=cfg.prefetch_enabled,
        to_device=cfg.data.loading.prefetch_to_device if cfg.data.loading else False,
    )
    total = cfg.total_updates
    limit = total if stop_after is None else min(total, stop_after)
    checkpoints = CheckpointPaths(output)
    try:
        if state.updates >= limit:
            return
        if world.rank == 0 and state.pending_plan is None:
            state.pending_plan = plan_update(cfg, state, sources, state.updates)
        metadata, local = world.local_plan(state.pending_plan)
        loader.submit(metadata["source"], local)
        for update in range(state.updates, limit):
            started = time.perf_counter()
            if metadata["update"] != update or metadata["source"] != cfg.source_at(
                update
            ):
                raise ValueError(
                    "Checkpoint pending plan differs from the training schedule"
                )
            source = metadata["source"]
            wait_started = time.perf_counter()
            batch = loader.consume()
            data_wait = time.perf_counter() - wait_started
            # Prepare N+1 from pools through N-1. N's states become eligible at N+2.
            following = None
            if update + 1 < total:
                if world.rank == 0:
                    state.pending_plan = plan_update(cfg, state, sources, update + 1)
                following, next_local = world.local_plan(state.pending_plan)
                loader.submit(following["source"], next_local)
            elif world.rank == 0:
                state.pending_plan = None
            if update == cfg.main_updates and not state.in_post_training:
                state.ema = state.new_ema()
                state.in_post_training = True
                rate = cfg.post_training_learning_rate or cfg.learning_rate
                for group in state.optimizer.param_groups:
                    group["lr"] = group["initial_lr"] = rate
                state.build_scheduler()
            learning_rate = state.optimizer.param_groups[0]["lr"]
            state.model.train()
            state.optimizer.zero_grad(set_to_none=True)
            parts_sum: dict[str, torch.Tensor] = {}
            next_states = []
            for index, item in enumerate(batch.examples):
                sync = (
                    objective.no_sync()
                    if isinstance(objective, DistributedDataParallel)
                    and index + 1 < per_rank
                    else nullcontext()
                )
                with sync:
                    with autocast(device.type == "cuda", torch.bfloat16):
                        loss, parts, states = objective(source, item)
                    (loss / per_rank).backward()
                next_states.append(states)
                for key, value in {"loss": loss.detach(), **parts}.items():
                    parts_sum[key] = parts_sum.get(key, 0) + value / per_rank
            norm = torch.nn.utils.clip_grad_norm_(
                state.model.parameters(), cfg.gradient_clip, error_if_nonfinite=True
            )
            state.optimizer.step()
            if state.scheduler is not None:
                state.scheduler.step()
            state.ema(state.model)
            means = world.mean(parts_sum)
            if cfg.fresh_examples < cfg.gradient_accumulation_steps:
                gathered = world.gather_states(next_states)
                if world.rank == 0:
                    assert gathered is not None
                    for planned, current in zip(
                        metadata["examples"], gathered, strict=True
                    ):
                        layout = sources.layout(
                            source, planned["origin"], planned["month"] + 1
                        )
                        if layout is not None:
                            state.pools[source].add(
                                ReplayEntry(
                                    source,
                                    planned["origin"],
                                    planned["month"] + 1,
                                    layout["start_lead"],
                                    planned["origin_update"],
                                    current,
                                )
                            )
            fresh = sum(item["cached_shape"] is None for item in metadata["examples"])
            replay = cfg.gradient_accumulation_steps - fresh
            state.updates = update + 1
            state.exposure[source]["updates"] += 1
            state.exposure[source]["fresh"] += fresh
            state.exposure[source]["replay"] += replay
            seconds, exposed_wait, read_seconds, stop = world.maximum(
                [
                    time.perf_counter() - started,
                    data_wait,
                    batch.read_seconds,
                    float(stopping()),
                ]
            )
            if world.rank == 0:
                emit(
                    {
                        "updates": state.updates,
                        "train/gradient_norm": float(norm),
                        **{f"train/{source}/{k}": v for k, v in means.items()},
                        f"train/{source}/lr": learning_rate,
                        f"train/{source}/fresh": fresh,
                        f"train/{source}/replay": replay,
                        f"train/{source}/seconds": seconds,
                        f"train/{source}/data_seconds": exposed_wait,
                        f"train/{source}/background_read_seconds": read_seconds,
                        f"train/{source}/max_start_lead_days": max(
                            item["start_lead"] for item in metadata["examples"]
                        )
                        * 5,
                        "world_size": world.size,
                        "global_examples": cfg.gradient_accumulation_steps,
                        **{
                            f"exposure/{s}/{k}": v
                            for s, counts in state.exposure.items()
                            for k, v in counts.items()
                        },
                    }
                )
            save_paths = []
            if state.updates % cfg.validation_freq == 0 or state.updates == total:
                validation_started = time.perf_counter()
                metrics = None
                if world.rank == 0:
                    with state.averaged_model():
                        metrics = validate(state.model)
                metrics = world.broadcast(metrics)
                assert metrics is not None
                score = metrics[baseline.SELECTION_METRIC]
                if not math.isfinite(score):
                    raise FloatingPointError("Annual selection score is nonfinite")
                if score < state.best_score:
                    state.best_score = score
                    save_paths.append(checkpoints.best_validation_checkpoint_path)
                if world.rank == 0:
                    emit(
                        {
                            "updates": state.updates,
                            **metrics,
                            "timing/validation_seconds": time.perf_counter()
                            - validation_started,
                        }
                    )
            if state.updates in cfg.milestone_updates:
                save_paths.append(output / f"main-{state.updates}.pt")
            if state.updates == cfg.main_updates:
                save_paths.append(output / "main.pt")
            if state.updates % cfg.save_freq == 0 or state.updates == limit or stop:
                save_paths.append(checkpoints.latest_checkpoint_path)
            if state.updates == total:
                save_paths.append(output / "endpoint.pt")
            if save_paths:
                world.capture_rng(state)
                if world.rank == 0:
                    before = time.perf_counter()
                    for path in save_paths:
                        state.save(path, manifest)
                    emit(
                        {
                            "updates": state.updates,
                            "timing/checkpoint_seconds": time.perf_counter() - before,
                        }
                    )
                world.broadcast(True if world.rank == 0 else None)
            if following is not None:
                metadata = following
            if stop:
                break
        if state.updates == total and world.rank == 0:
            write_json(
                {"updates": total, "exposure": state.exposure}, output / "COMPLETE.json"
            )
    finally:
        loader.close()
        world.close()
