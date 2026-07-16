import contextlib
import dataclasses
import datetime
import itertools
import logging
import multiprocessing
import os
import queue
import signal
import tempfile
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any

import dask
import torch
import torch.nn as nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from ocean_emulators import config
from ocean_emulators.aggregator import Aggregator
from ocean_emulators.aggregator.loss import (
    get_channel_loss_dict,
    get_channel_loss_scale_dict,
    get_depth_loss_dict,
    get_variable_loss_dict,
)
from ocean_emulators.backend import (
    init_domain_parallel_backend,
    init_train_backend,
)
from ocean_emulators.config import TrainConfig, build_loss_fn
from ocean_emulators.constants import (
    BOUNDARY_VARS,
    MAX_TRAIN_MODEL_STEPS_FORWARD,
    PROGNOSTIC_VARS,
    BoundaryVarNames,
    Grid,
    PrognosticVarNames,
    TensorMap,
    construct_metadata,
)
from ocean_emulators.datasets import (
    InferenceDataset,
    InferenceDatasets,
    RawTrainData,
    RawReplayBatch,
    RawReplayTransition,
    ReplayBatchRequest,
    ReplayBatchSlot,
    ReplayCursor,
    ReplaySeedSlot,
    TorchTrainDataset,
    TrainData,
    TrainDataLoader,
)
from ocean_emulators.models.base import BaseModel
from ocean_emulators.replay import ReplayBuffer, ReplayEntry, replay_sidecar_path
from ocean_emulators.shardtensor import DomainParallelContext, validate_shardable
from ocean_emulators.stepper import Stepper, TrainBatchOutput, ValBatchOutput
from ocean_emulators.utils.data import (
    LoadStats,
    Normalize,
    get_inference_steps,
    spherical_area_weights,
)
from ocean_emulators.utils.device import using_gpu
from ocean_emulators.utils.distributed import (
    all_reduce_mean,
    get_rank,
    get_world_size,
    is_main_process,
    set_seed,
)
from ocean_emulators.utils.ema import EMATracker
from ocean_emulators.utils.logging import (
    MetricLogger,
    SmoothedValue,
    get_model_summary,
    handle_logging,
    handle_warnings,
)
from ocean_emulators.utils.loss import LossFn
from ocean_emulators.utils.schedule import EpochMultiplierScheduler
from ocean_emulators.utils.train import (
    CheckpointPaths,
    collate_inference_data,
    collate_raw_train_data,
)
from ocean_emulators.utils.wandb import WandBLogger

logger = logging.getLogger(__name__)


class GracefulStopRequested(RuntimeError):
    """Raised when training should stop after writing an emergency checkpoint."""


class _OffsetBatchSampler(Sampler[list[int]]):
    """Skip a fixed number of batches from an existing batch sampler."""

    def __init__(self, batch_sampler: Sampler[list[int]], start_batch: int):
        self._batch_sampler = batch_sampler
        self._start_batch = max(start_batch, 0)

    def __iter__(self):
        return itertools.islice(iter(self._batch_sampler), self._start_batch, None)

    def __len__(self) -> int:
        return max(len(self._batch_sampler) - self._start_batch, 0)


class _DomainFollowerLoader:
    """Yield metadata-only batches on ranks that do not read the global patch.

    The domain leader performs the Phase 2 full-patch read. Followers still
    enter every training collective in lockstep, but avoid redundant storage
    reads and host-to-device copies.
    """

    def __init__(self, num_batches: int, num_prognostic_channels: int):
        self._num_batches = num_batches
        self._num_prognostic_channels = num_prognostic_channels

    def __iter__(self):
        for _ in range(self._num_batches):
            yield TrainData(self._num_prognostic_channels)

    def __len__(self) -> int:
        return self._num_batches

    def with_offset(self, start_batch: int) -> "_DomainFollowerLoader":
        return _DomainFollowerLoader(
            max(0, self._num_batches - start_batch),
            self._num_prognostic_channels,
        )


@dataclasses.dataclass
class _ReplayPreparedBatch:
    request: ReplayBatchRequest
    data: TrainData
    cursors: list[ReplayCursor]
    seed_entries: dict[int, ReplayEntry]
    load_stats: LoadStats | None
    ready_event: torch.cuda.Event | None = None

    def wait_ready(self) -> None:
        if self.ready_event is not None:
            torch.cuda.current_stream().wait_event(self.ready_event)


def _replay_cursor_from_dict(raw: dict[str, int]) -> ReplayCursor:
    return ReplayCursor(
        dataset_index=int(raw["dataset_index"]),
        source_index=int(raw["source_index"]),
        lead_step=int(raw["lead_step"]),
        stride=int(raw["stride"]),
        temporal_stride=int(raw["temporal_stride"]),
    )


def _replay_request_to_dict(request: ReplayBatchRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "temporal_bundle_size": request.temporal_bundle_size,
        "train_slots": [
            {
                "replay_index": slot.replay_index,
                "cursor": dataclasses.asdict(slot.cursor),
            }
            for slot in request.train_slots
        ],
        "seed_slots": [
            {
                "replay_index": slot.replay_index,
                "cursor": dataclasses.asdict(slot.cursor),
                "reason": slot.reason,
            }
            for slot in request.seed_slots
        ],
    }


def _replay_request_from_dict(raw: dict[str, Any]) -> ReplayBatchRequest:
    return ReplayBatchRequest(
        request_id=int(raw["request_id"]),
        temporal_bundle_size=int(raw.get("temporal_bundle_size", 1)),
        train_slots=tuple(
            ReplayBatchSlot(
                replay_index=int(slot["replay_index"]),
                cursor=_replay_cursor_from_dict(slot["cursor"]),
            )
            for slot in raw["train_slots"]
        ),
        seed_slots=tuple(
            ReplaySeedSlot(
                replay_index=int(slot["replay_index"]),
                cursor=_replay_cursor_from_dict(slot["cursor"]),
                reason=str(slot["reason"]),
            )
            for slot in raw["seed_slots"]
        ),
    )


class _ReplayPrefetchPipeline:
    def __init__(
        self,
        trainer: "Trainer",
        *,
        start_batch_in_epoch: int,
        total_batches: int,
        max_lead_steps: int,
        refresh_every_n_microbatches: int,
    ) -> None:
        self.trainer = trainer
        self.start_batch_in_epoch = start_batch_in_epoch
        self.total_batches = total_batches
        self.max_lead_steps = max_lead_steps
        self.refresh_every_n_microbatches = refresh_every_n_microbatches
        self.horizon = trainer.replay_prefetch_horizon()
        self._next_to_yield = start_batch_in_epoch
        self._next_to_plan = start_batch_in_epoch
        self._yielded_count = 0
        self._completed_count = 0
        self._completed_request_ids: set[int] = set()
        self._planned_requests: OrderedDict[int, ReplayBatchRequest] = OrderedDict()
        self._reserved_indices: set[int] = set()
        self._raw_cache: dict[int, RawReplayBatch] = {}
        self._prepared_cache: dict[int, _ReplayPreparedBatch] = {}
        self._closed = False
        trainer._replay_active_planner = self

        self._request_queue: queue.Queue[ReplayBatchRequest | None] = queue.Queue(
            maxsize=max(2, self.horizon + 1)
        )
        self._result_queue: queue.Queue[RawReplayBatch | BaseException] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_threads = [
            threading.Thread(
                target=self._worker_loop,
                name=f"replay_prefetch_{index}",
                daemon=True,
            )
            for index in range(self._worker_count())
        ]
        for thread in self._worker_threads:
            thread.start()

        self._restore_state(trainer.consume_replay_planner_resume_state())
        self._fill_window()

        logger.info(
            "Replay prefetch using %s in-process thread(s), horizon=%s. "
            "This avoids multiprocessing shareFd/pickle copies for large gold frames.",
            len(self._worker_threads),
            self.horizon,
        )

    def __len__(self) -> int:
        return max(self.total_batches - self.start_batch_in_epoch, 0)

    def __iter__(self):
        try:
            while self._yielded_count < len(self):
                yield self.next_prepared()
        finally:
            self.close()

    def next_prepared(self) -> _ReplayPreparedBatch:
        self._drain_raw_results_nonblocking()
        prepared = self._pop_ready_prepared()
        if prepared is None:
            raw_batch = self._pop_ready_raw()
            if raw_batch is None:
                raw_batch = self._wait_for_any_raw_batch()
            prepared = self._prepare_async(raw_batch)
        prepared.wait_ready()
        self._yielded_count += 1
        return prepared

    def _pop_ready_prepared(self) -> _ReplayPreparedBatch | None:
        if not self._prepared_cache:
            return None
        if self._next_to_yield in self._prepared_cache:
            return self._prepared_cache.pop(self._next_to_yield)
        request_id = next(iter(self._prepared_cache))
        return self._prepared_cache.pop(request_id)

    def _pop_ready_raw(self) -> RawReplayBatch | None:
        if not self._raw_cache:
            return None
        if self._next_to_yield in self._raw_cache:
            return self._raw_cache.pop(self._next_to_yield)
        request_id = next(iter(self._raw_cache))
        return self._raw_cache.pop(request_id)

    def _prepare_cached_ready_without_blocking(self) -> None:
        if self._prepared_cache:
            return
        raw_batch = self._pop_ready_raw()
        if raw_batch is not None:
            self._prepared_cache[raw_batch.request.request_id] = self._prepare_async(
                raw_batch
            )

    def complete(self, prepared: _ReplayPreparedBatch) -> None:
        request = prepared.request
        self._planned_requests.pop(request.request_id, None)
        self._reserved_indices.difference_update(request.reserved_indices)
        self._completed_request_ids.add(request.request_id)
        self._completed_count += 1
        while self._next_to_yield in self._completed_request_ids:
            self._completed_request_ids.remove(self._next_to_yield)
            self._next_to_yield += 1
        self._fill_window()
        self._drain_raw_results_nonblocking()
        self._prepare_cached_ready_without_blocking()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        sentinel_count = max(len(self._worker_threads), 1)
        for _ in range(sentinel_count):
            try:
                self._request_queue.put_nowait(None)
            except (BrokenPipeError, EOFError):
                break
            except queue.Full:
                break
        for thread in self._worker_threads:
            thread.join(timeout=1.0)
        if self._completed_count >= len(self):
            self.trainer._replay_active_planner = None
            self.trainer._replay_active_planner_state = None
        else:
            self.trainer._replay_active_planner = None
            self.trainer._replay_active_planner_state = self.state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "start_batch_in_epoch": self.start_batch_in_epoch,
            "total_batches": self.total_batches,
            "max_lead_steps": self.max_lead_steps,
            "refresh_every_n_microbatches": self.refresh_every_n_microbatches,
            "horizon": self.horizon,
            "next_to_yield": self._next_to_yield,
            "next_to_plan": self._next_to_plan,
            "yielded_count": self._yielded_count,
            "completed_count": self._completed_count,
            "completed_request_ids": sorted(self._completed_request_ids),
            "reserved_indices": sorted(self._reserved_indices),
            "planned_requests": [
                _replay_request_to_dict(request)
                for request in self._planned_requests.values()
                if request.request_id >= self._next_to_yield
            ],
        }

    def _restore_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        if (
            state.get("version") != 2
            or state.get("total_batches") != self.total_batches
            or state.get("max_lead_steps") != self.max_lead_steps
            or state.get("refresh_every_n_microbatches")
            != self.refresh_every_n_microbatches
            or state.get("completed_count", 0) != 0
        ):
            logger.warning(
                "Ignoring replay planner resume state because it does not match "
                "the current replay epoch position."
            )
            return

        self._next_to_yield = int(state["next_to_yield"])
        self._next_to_plan = int(state["next_to_plan"])
        self._yielded_count = int(state.get("yielded_count", 0))
        self._completed_count = int(state.get("completed_count", 0))
        self._completed_request_ids = {
            int(index) for index in state.get("completed_request_ids", [])
        }
        self._reserved_indices = {
            int(index) for index in state.get("reserved_indices", [])
        }
        for raw_request in state.get("planned_requests", []):
            request = _replay_request_from_dict(raw_request)
            if request.request_id < self._next_to_yield:
                continue
            self._planned_requests[request.request_id] = request
            self._request_queue.put(request)

    def _fill_window(self) -> None:
        while (
            len(self._planned_requests) < self.horizon
            and self._next_to_plan < self.total_batches
        ):
            try:
                request = self.trainer.plan_replay_batch(
                    global_batch_index=self._next_to_plan,
                    max_lead_steps=self.max_lead_steps,
                    refresh_every_n_microbatches=(
                        self.refresh_every_n_microbatches
                    ),
                    exclude_reserved=self._reserved_indices,
                )
            except RuntimeError:
                if self._planned_requests:
                    break
                raise

            self._reserved_indices.update(request.reserved_indices)
            self._planned_requests[request.request_id] = request
            self._request_queue.put(request)
            self._next_to_plan += 1

    def _wait_for_any_raw_batch(self) -> RawReplayBatch:
        while True:
            result = self._result_queue.get()
            if isinstance(result, BaseException):
                raise result
            raw_batch = result
            return raw_batch

    def _drain_raw_results_nonblocking(self) -> None:
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(result, BaseException):
                raise result
            self._raw_cache[result.request.request_id] = result

    def _worker_count(self) -> int:
        return max(1, self.trainer.num_workers)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return

            try:
                self._result_queue.put(self._load_raw_batch(request))
            except BaseException as e:
                self._result_queue.put(e)
                return

    def _load_raw_batch(self, request: ReplayBatchRequest) -> RawReplayBatch:
        start_time = time.perf_counter()
        train_transitions = [
            self._prepare_raw_transition_for_transport(
                self.trainer.train_datasets[slot.cursor.dataset_index]
                .get_raw_replay_train_transition(
                    dataset_index=slot.cursor.dataset_index,
                    source_index=slot.cursor.source_index,
                    lead_step=slot.cursor.lead_step,
                )
            )
            for slot in request.train_slots
        ]
        seed_transitions = [
            self._prepare_raw_transition_for_transport(
                self.trainer.train_datasets[slot.cursor.dataset_index]
                .get_raw_replay_seed_transition(
                    dataset_index=slot.cursor.dataset_index,
                    source_index=slot.cursor.source_index,
                    lead_step=slot.cursor.lead_step,
                )
            )
            for slot in request.seed_slots
        ]
        return RawReplayBatch(
            request=request,
            train_transitions=train_transitions,
            seed_transitions=seed_transitions,
            load_stats=LoadStats(time.perf_counter() - start_time),
        )

    def _prepare_raw_transition_for_transport(
        self,
        transition: RawReplayTransition,
    ) -> RawReplayTransition:
        def convert(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if tensor.dtype != self.trainer.replay_storage_dtype:
                tensor = tensor.to(dtype=self.trainer.replay_storage_dtype)
            return tensor

        transition.target_prognostic = convert(transition.target_prognostic)
        transition.seed_prognostic = convert(transition.seed_prognostic)
        transition.boundary = convert(transition.boundary)
        return transition

    def _prepare_async(self, raw_batch: RawReplayBatch) -> _ReplayPreparedBatch:
        if self.trainer.device.type != "cuda":
            return self.trainer.prepare_raw_replay_batch(raw_batch, ready_event=None)

        if self.trainer.pin_mem and torch.cuda.is_available():
            raw_batch.pin_memory()

        assert self.trainer.replay_copy_stream is not None
        with torch.cuda.stream(self.trainer.replay_copy_stream):
            prepared = self.trainer.prepare_raw_replay_batch(raw_batch, ready_event=None)
            prepared.ready_event = torch.cuda.Event()
            prepared.ready_event.record(self.trainer.replay_copy_stream)
            return prepared

class Trainer:
    model: nn.Module

    def __init__(self, cfg: TrainConfig) -> None:
        cfg.prepare_output_dirs()
        cfg.save_yaml(cfg.experiment.output_dir / "config.yaml")

        if cfg.domain_parallel.enabled:
            if cfg.replay.enabled:
                raise ValueError(
                    "Domain-parallel replay is deferred to Phase 4. Set "
                    "replay.enabled=false for Phase 2 curriculum training."
                )
            if cfg.inference_epochs:
                raise ValueError(
                    "Domain-parallel rollout inference is deferred to Phase 3. "
                    "Set inference_epochs=[]; one-step validation is supported."
                )
            if not isinstance(cfg.model, config.SamudraConfig):
                raise ValueError("Domain-parallel Phase 2 supports Samudra only.")
            if not isinstance(cfg.loss, str) or cfg.loss not in {
                "mse",
                "mae",
                "mse_mae",
            }:
                raise ValueError(
                    "Domain-parallel Phase 2 supports loss='mse', 'mae', or "
                    "'mse_mae'. Gradient and adaptive losses are later gates."
                )
            if not cfg.domain_parallel.leader_scatter:
                raise ValueError(
                    "Phase 2 requires domain_parallel.leader_scatter=true. "
                    "Per-rank spatial reads are deferred to the PatchCatalog phase."
                )
            if cfg.domain_parallel.use_fsdp:
                raise ValueError("FSDP is deferred until multi-cluster Phase 5.")

        # Backend. PhysicsNeMo owns process-group initialization in DP mode;
        # the ordinary single-GPU/DDP path remains exactly as before.
        self.dp_ctx: DomainParallelContext | None = None
        self._physicsnemo_dm = None
        if cfg.domain_parallel.enabled:
            self.device, self.distributed, self._physicsnemo_dm = (
                init_domain_parallel_backend(cfg.backend)
            )
        else:
            self.device, self.distributed = init_train_backend(
                cfg.backend, ddp_timeout_minutes=cfg.ddp_timeout_minutes
            )

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.pin_mem = False
        elif cfg.disk_mode:
            cfg.pin_mem = True

        # Distributed mode
        dask.config.set(scheduler="synchronous")

        # Set seeds
        set_seed(cfg.experiment.rand_seed)

        # Getting prognostic and boundary variables
        self.prognostic_var_names: PrognosticVarNames = PROGNOSTIC_VARS[
            cfg.experiment.prognostic_vars_key
        ]
        self.boundary_var_names: BoundaryVarNames = BOUNDARY_VARS[
            cfg.experiment.boundary_vars_key
        ]

        levels = cfg.experiment.prognostic_vars_key.split("_")[-1]
        if "all" in levels:
            self.levels = 51
        else:
            self.levels = int(levels)

        str_prognostics = ", ".join([i for i in self.prognostic_var_names])
        str_boundaries = ", ".join([i for i in self.boundary_var_names])

        logger.info(f"Prognostic variables: {str_prognostics}")
        logger.info(f"Boundary variables: {str_boundaries}")
        logger.info(f"Levels: {self.levels}")

        self.N_bound = len(self.boundary_var_names)
        self.N_prog = len(self.prognostic_var_names)

        self.data_container = cfg.data.build(
            data_root=cfg.experiment.resolved_data_root,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
        )

        if (
            self.distributed is not None
            and not cfg.domain_parallel.enabled
            and cfg.data.num_workers > 0
        ):
            world_size = get_world_size()
            scaled_workers = max(1, cfg.data.num_workers // world_size)
            capped_workers = (
                min(scaled_workers, cfg.ddp_max_data_workers_per_rank)
                if world_size > 1
                else scaled_workers
            )
            if scaled_workers != cfg.data.num_workers:
                logger.info(
                    "Scaling data.num_workers from "
                    f"{cfg.data.num_workers} to {scaled_workers} per rank "
                    f"for world_size={world_size}."
                )
            if capped_workers != scaled_workers:
                logger.info(
                    "Capping data.num_workers per rank from "
                    f"{scaled_workers} to {capped_workers} "
                    f"(ddp_max_data_workers_per_rank="
                    f"{cfg.ddp_max_data_workers_per_rank})."
                )
            cfg.data.num_workers = capped_workers

        self.mp_context: BaseContext | None = None
        if cfg.data.num_workers > 0:
            if self.data_container.supports_fork:
                self.mp_context = multiprocessing.get_context("fork")
            else:
                self.mp_context = multiprocessing.get_context("spawn")

        self.num_in = int((cfg.data.hist + 1) * (self.N_prog + self.N_bound))
        self.num_out = int((cfg.data.hist + 1) * self.N_prog)

        self.tensor_map = TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        logger.info(f"Number of inputs (prognostic + boundary): {self.num_in}")
        logger.info(f"Number of outputs (prognostic): {self.num_out}")

        assert isinstance(cfg.data_stride, list)
        if isinstance(cfg.temporal_stride, int):
            temporal_strides = [cfg.temporal_stride]
        else:
            temporal_strides = cfg.temporal_stride
        assert isinstance(temporal_strides, list)
        assert all(
            isinstance(temporal_stride, int) and temporal_stride >= 1
            for temporal_stride in temporal_strides
        )
        assert isinstance(cfg.steps, list)
        assert isinstance(cfg.step_transition, list)
        assert len(cfg.step_transition) == len(cfg.steps) - 1
        assert isinstance(cfg.replay.max_lead_steps, list)
        assert isinstance(cfg.replay.max_lead_transition, list)
        assert all(max_lead > 0 for max_lead in cfg.replay.max_lead_steps)
        assert cfg.replay.max_lead_steps == sorted(cfg.replay.max_lead_steps)
        assert cfg.replay.max_lead_transition == sorted(
            cfg.replay.max_lead_transition
        )
        assert (
            len(cfg.replay.max_lead_transition)
            == len(cfg.replay.max_lead_steps) - 1
        )
        if isinstance(cfg.replay.refresh_every_n_microbatches, int):
            replay_refresh_values = [cfg.replay.refresh_every_n_microbatches]
        else:
            replay_refresh_values = cfg.replay.refresh_every_n_microbatches
        assert isinstance(replay_refresh_values, list)
        assert all(refresh > 0 for refresh in replay_refresh_values)
        assert isinstance(cfg.replay.refresh_every_n_microbatches_transition, list)
        assert cfg.replay.refresh_every_n_microbatches_transition == sorted(
            cfg.replay.refresh_every_n_microbatches_transition
        )
        assert (
            len(cfg.replay.refresh_every_n_microbatches_transition)
            == len(replay_refresh_values) - 1
        )
        if cfg.replay.enabled and cfg.data.hist != 0:
            raise ValueError(
                "Replay buffer training currently supports data.hist=0 only "
                "(one timestamp in, one next timestamp out)."
            )
        assert isinstance(cfg.lr_multipliers, list)
        assert isinstance(cfg.lr_multiplier_transition, list)
        assert len(cfg.lr_multiplier_transition) == len(cfg.lr_multipliers) - 1
        assert isinstance(cfg.temporal_stride_transition, list)
        assert len(cfg.temporal_stride_transition) == len(temporal_strides) - 1
        max_steps = str(cfg.steps[-1])
        self.str_video = "steps_" + max_steps + "_" + "_Lateral_Data_025_no_smooth"

        # Dataloaders
        logger.info(f"Loading data")
        if cfg.train_time.overlaps(cfg.val_time):
            raise ValueError(
                f"Training time range {cfg.train_time} overlaps "
                f"with validation time range {cfg.val_time}"
            )

        if cfg.data.concurrent_compute:
            logger.warning(
                "Forcing data.concurrent_compute=false for training stability."
            )
            cfg.data.concurrent_compute = False

        self.executor: ThreadPoolExecutor | None = None
        if cfg.data.concurrent_compute:
            self.executor = ThreadPoolExecutor(
                max_workers=None, thread_name_prefix="concurrent_compute"
            )

        self.src = self.data_container.source
        self.data = self.data_container.source.data
        self.static_data = self.data_container.static_data

        # We use dask for inference since it has memory issues otherwise.
        # TODO(jder): Could rewrite inference dataset like we did for TorchTrainDataset
        # see https://github.com/suryadheeshjith/Ocean_Emulator/issues/208
        self.inference_src = self.data_container.source_using_dask

        self.loader_version = self.data_container.loader_version

        self.metadata = construct_metadata(self.data)
        self.wet = self.src.masks.prognostic_with_hist(cfg.data.hist).to(self.device)
        self.area_weights: Grid = spherical_area_weights(self.data)

        self.area_weights = self.area_weights.to(self.device)

        if cfg.domain_parallel.enabled:
            assert self._physicsnemo_dm is not None
            global_h, global_w = self.wet.shape[-2:]
            validate_shardable(
                global_h,
                global_w,
                cfg.domain_parallel.cluster_shape,
                num_downsamples=len(cfg.model.unet.ch_width),
                max_halo=(
                    (cfg.model.unet.core_block.kernel_size - 1)
                    // 2
                    * max(cfg.model.unet.dilation)
                ),
            )
            self.dp_ctx = DomainParallelContext(
                cfg.domain_parallel,
                self._physicsnemo_dm,
                self.device,
            )

        self.normalize = Normalize.init_instance(
            self.src,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
        )

        model_build_kwargs = {
            "in_channels": self.num_in,
            "out_channels": self.num_out,
            "hist": cfg.data.hist,
            "wet": self.wet,
            "area_weights": self.area_weights,
            "static_data": self.static_data,
            "lat": torch.from_numpy(self.data.lat.values),
            "lon": torch.from_numpy(self.data.lon.values),
        }
        if isinstance(cfg.model, config.SamudraConfig):
            model_build_kwargs["domain_parallel"] = self.dp_ctx is not None
        self.model = cfg.model.build(**model_build_kwargs).to(self.device)

        if self.dp_ctx is not None:
            self.model = self.dp_ctx.distribute_model(self.model)
            leader_wet = self.wet if self.dp_ctx.is_domain_leader else None
            leader_area_weights = (
                self.area_weights if self.dp_ctx.is_domain_leader else None
            )
            self.domain_wet = self.dp_ctx.scatter_spatial(leader_wet, ndim=3)
            self.domain_model_wet = self.dp_ctx.scatter_spatial(
                leader_wet.unsqueeze(0) if leader_wet is not None else None,
                ndim=4,
            )
            self.domain_area_weights = self.dp_ctx.scatter_spatial(
                leader_area_weights, ndim=2
            )
            # BaseModel stores wet as a plain attribute rather than a buffer.
            # Replace it after distribute_module so torch.where sees the same
            # spatial layout as the model output.
            self.model.wet = self.domain_model_wet
        else:
            self.domain_wet = self.wet
            self.domain_model_wet = self.wet
            self.domain_area_weights = self.area_weights

        self.nets_dir = cfg.experiment.nets_dir
        self.network = self.model.__class__.__name__

        # Loss function
        self.loss_fn: LossFn = build_loss_fn(
            cfg.loss,
            wet=self.domain_wet,
            y_coord=self.data.lat,
            device=self.device,
            num_channels=self.N_prog,
            pad_mode=cfg.model.pad,
            num_halo=getattr(cfg.model, "num_halo", 0),
            num_sponge=getattr(cfg.model, "num_sponge", 0),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # Scheduler
        self.scheduler = None
        if cfg.scheduler:
            self.scheduler = cfg.scheduler.build(self.optimizer, cfg.epochs)
        self.lr_multipliers = cfg.lr_multipliers
        self.lr_multiplier_transition = cfg.lr_multiplier_transition
        self.use_lr_multipliers = (
            self.lr_multipliers != [1.0] or len(self.lr_multiplier_transition) > 0
        )
        if self.use_lr_multipliers:
            self.scheduler = EpochMultiplierScheduler(
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                multipliers=self.lr_multipliers,
                transition_epochs=self.lr_multiplier_transition,
                current_epoch=1,
            )

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(
            cfg.experiment.wandb.mode != "disabled", is_main_process()
        )

        self.ckpt_paths = CheckpointPaths(self.nets_dir)

        # Check for preemption
        if cfg.preemptible:
            assert not cfg.finetune, "Finetune is not supported with preemptible"
            if resumable_ckpt := self.ckpt_paths.latest_resumable_checkpoint_path():
                cfg.resume_ckpt_path = str(resumable_ckpt)

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            cfg.resume_ckpt_path,
            cfg,
            data_container=self.data_container,
            finetune=cfg.finetune,
        )

        # Log effective batch size (wandb logging happens after resume state is loaded)
        effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
        logger.info(
            f"Effective batch size: {effective_batch_size} "
            f"(batch_size={cfg.batch_size} × "
            f"gradient_accumulation_steps={cfg.gradient_accumulation_steps})"
        )

        self.num_batches_seen = 0
        self.start_batch_in_epoch = 0
        loaded_checkpoint = False
        if cfg.resume_ckpt_path is not None:
            if self.dp_ctx is not None:
                raise NotImplementedError(
                    "Resuming optimizer/EMA state from a domain-parallel checkpoint "
                    "is not validated in Phase 2. Start the smoke run without "
                    "resume_ckpt_path; dense model checkpoints are still written."
                )
            if cfg.finetune:
                self.load_checkpoint(cfg.resume_ckpt_path, finetune=True)
                self.start_epoch = 1
                self.start_batch_in_epoch = 0
            else:
                self.load_checkpoint(cfg.resume_ckpt_path)
                if cfg.reset_optimizer_on_resume or cfg.reset_scheduler_on_resume:
                    if cfg.reset_optimizer_on_resume:
                        self.optimizer = torch.optim.Adam(
                            self.model.parameters(), lr=cfg.learning_rate
                        )
                        logger.info(
                            "Reset optimizer state on resume (lr=%s).",
                            cfg.learning_rate,
                        )
                    # Scheduler is tied to the optimizer; rebuild if either reset is requested.
                    if cfg.scheduler:
                        self.scheduler = cfg.scheduler.build(
                            self.optimizer, cfg.epochs
                        )
                        logger.info("Reset scheduler state on resume.")
                    else:
                        self.scheduler = None
                        if cfg.reset_scheduler_on_resume:
                            logger.info(
                                "No scheduler configured; skipping scheduler reset."
                            )
                    if self.use_lr_multipliers:
                        self.scheduler = EpochMultiplierScheduler(
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            multipliers=self.lr_multipliers,
                            transition_epochs=self.lr_multiplier_transition,
                            current_epoch=self.start_epoch,
                        )
                    logger.info(
                        "Optimizer LR after reset: %s",
                        self.optimizer.param_groups[-1]["lr"],
                    )
                if not self.wandb_logger.enabled and is_main_process():
                    warnings.warn(
                        "This checkpoint had wandb enabled, \
                            but wandb is not enabled now!"
                    )
            loaded_checkpoint = True
        else:
            self.start_epoch = 1
            self.start_batch_in_epoch = 0

        if self.is_wandb_enabled():
            self.wandb_logger.log(
                {
                    "config/effective_batch_size": effective_batch_size,
                },
                step=self.num_batches_seen,
            )

        self._stop_requested = False
        self._stop_reason: str | None = None
        self._active_epoch: int | None = None
        self._last_completed_batch_in_epoch = -1
        self._finished = False

        # The DP model was distributed before optimizer/checkpoint/EMA setup so
        # all of them reference the final DTensor parameters.
        if self.dp_ctx is not None:
            logger.info(
                "Using PhysicsNeMo domain parallelism with cluster_shape=%s",
                cfg.domain_parallel.cluster_shape,
            )
        elif self.distributed is not None:
            ddp_options = {
                "device_ids": [self.distributed.gpu],
                "bucket_cap_mb": cfg.ddp_bucket_cap_mb,
                "gradient_as_bucket_view": cfg.ddp_gradient_as_bucket_view,
                "static_graph": cfg.ddp_static_graph,
                "find_unused_parameters": cfg.ddp_find_unused_parameters,
                "broadcast_buffers": cfg.ddp_broadcast_buffers,
            }
            logger.info(f"Initializing DDP with options: {ddp_options}")
            self.model = nn.parallel.DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(self.model),
                **ddp_options,
            )

        # EMA (must come after DDP setup so parameter names match final self.model)
        if not loaded_checkpoint:
            self._ema = EMATracker(
                self.model,
                decay=cfg.ema_decay,
                faster_decay_at_start=cfg.faster_decay_at_start,
            )

        # Training
        self.epochs = cfg.epochs
        self.test_using_ema = cfg.test_using_ema
        self.hist: int = cfg.data.hist
        self.steps = cfg.steps
        self.step_transition = cfg.step_transition
        self.replay_cfg = cfg.replay
        self.replay_enabled = cfg.replay.enabled
        self.replay_buffer: ReplayBuffer | None = None
        self.replay_resume_checkpoint_path = (
            Path(cfg.resume_ckpt_path)
            if cfg.resume_ckpt_path is not None and not cfg.finetune
            else None
        )
        self._replay_resume_consumed = False
        self.replay_storage_dtype = (
            torch.bfloat16 if getattr(cfg.model, "use_bfloat16", False) else torch.float32
        )
        self.replay_generator = torch.Generator(device="cpu")
        self.replay_generator.manual_seed(cfg.experiment.rand_seed + 104729 * get_rank())
        self.replay_copy_stream = (
            torch.cuda.Stream() if self.device.type == "cuda" else None
        )
        self._replay_resume_planner_state: dict[str, Any] | None = None
        self._replay_active_planner: _ReplayPrefetchPipeline | None = None
        self._replay_active_planner_state: dict[str, Any] | None = None
        self.save_freq = cfg.save_freq
        self.output_dir = cfg.experiment.output_dir
        self.debug = cfg.debug
        self.surface_snapshot: bool = cfg.surface_snapshot
        self.data_stride: list[int] = cfg.data_stride
        self.temporal_strides: list[int] = temporal_strides
        self.temporal_stride_transition: list[int] = cfg.temporal_stride_transition
        self.temporal_stride: int = self.temporal_strides[0]
        self.batch_size: int = cfg.batch_size
        self.gradient_accumulation_steps: int = cfg.gradient_accumulation_steps
        self.ddp_use_no_sync_for_accumulation = (
            cfg.ddp_use_no_sync_for_accumulation
        )
        self.slow_batch_log_threshold_seconds: float = (
            cfg.slow_batch_log_threshold_seconds
        )
        self.ddp_static_graph: bool = cfg.ddp_static_graph
        if (
            self.ddp_static_graph
            and self.ddp_use_no_sync_for_accumulation
            and self.gradient_accumulation_steps > 1
        ):
            logger.warning(
                "Disabling DDP no_sync accumulation because ddp_static_graph=true. "
                "This avoids known DDP reducer assertions in this PyTorch setup."
            )
            self.ddp_use_no_sync_for_accumulation = False
        self.num_workers: int = cfg.data.num_workers
        self.prefetch_factor: int = cfg.data.prefetch_factor
        self.pin_mem: bool = cfg.pin_mem
        self.train_shuffle: bool = cfg.data.train_shuffle
        self.train_time: config.TimeConfig = cfg.train_time
        self.val_time = cfg.val_time
        self.inference_times = cfg.inference_times
        self.inference_epochs = cfg.inference_epochs
        self.max_train_model_steps_forward = MAX_TRAIN_MODEL_STEPS_FORWARD // (
            self.hist + 1
        )
        self.normalize_before_mask: bool = cfg.data.normalize_before_mask
        self.normalize_fill_value: float = cfg.data.masked_fill_value

        self.profiler = cfg.profiler.build(self.output_dir, self.device)
        self.emergency_checkpoint_interval_seconds = (
            cfg.emergency_checkpoint_interval_minutes * 60.0
        )
        self._next_emergency_checkpoint_time: float | None = None
        if self.emergency_checkpoint_interval_seconds > 0:
            self._next_emergency_checkpoint_time = (
                time.perf_counter() + self.emergency_checkpoint_interval_seconds
            )
            logger.info(
                "Periodic emergency checkpointing enabled: every %.1f minutes",
                cfg.emergency_checkpoint_interval_minutes,
            )

        assert self.tensor_map is not None

        if self.inference_epochs:
            self.init_inference_stores()

        # Add type annotations for samplers
        self.train_sampler: DistributedSampler | RandomSampler | SequentialSampler
        self.val_sampler: DistributedSampler | RandomSampler
        self.inference_sampler: DistributedSampler | RandomSampler
        self.train_datasets: list[TorchTrainDataset]
        self.val_datasets: list[TorchTrainDataset]

        # Add type annotations for loaders
        self.train_loader: TrainDataLoader
        self.val_loader: TrainDataLoader
        self.inference_loader: DataLoader[TrainData]

        self._install_signal_handlers()

    def init_inference_stores(self):
        # Determine number of processes based on device
        if using_gpu():
            num_splits = get_world_size()
            logger.info(f"Number of processes: {num_splits}, preferably use 8")
        else:
            num_splits = 1

        # Create datasets
        inference_datasets = []
        num_steps_inf_set = []
        for i in range(num_splits):
            sliced_src = self.inference_src.slice(self.inference_times[i])
            num_time_steps = get_inference_steps(
                sliced_src,
                hist=self.hist,
            )
            inference_dataset = InferenceDataset(
                src=sliced_src,
                prognostic_var_names=self.prognostic_var_names,
                boundary_var_names=self.boundary_var_names,
                hist=self.hist,
                normalize_before_mask=self.normalize_before_mask,
                masked_fill_value=self.normalize_fill_value,
                long_rollout=True,
            )

            inference_datasets.append(inference_dataset)
            num_steps_inf_set.append(num_time_steps)

        inference_data_combined = InferenceDatasets(
            inference_datasets, num_steps_inf_set
        )

        if self.distributed is not None:
            self.inference_sampler = DistributedSampler(
                inference_data_combined, shuffle=True
            )
        else:
            self.inference_sampler = RandomSampler(inference_data_combined)

        # Create data loaders
        self.inference_loader = DataLoader(
            inference_data_combined,
            batch_size=1,
            sampler=self.inference_sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_inference_data,
            multiprocessing_context=self.mp_context,
        )

    def run(self) -> None:
        logger.info(f"Starting training")

        self.best_val_loss = 1e8
        self.best_inf_loss = 1e8
        self.wandb_logger.watch(self.model, log="all")

        self.profiler.start()

        start_time = time.perf_counter()
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                self._active_epoch = epoch

                # Iterative step training
                if self.replay_enabled:
                    if epoch == self.start_epoch or epoch in self.temporal_stride_transition:
                        cur_temporal_stride = self.get_current_temporal_stride(epoch)
                        self.temporal_stride = cur_temporal_stride
                        self.init_data_loaders(
                            max(self.replay_cfg.max_lead_steps),
                            cur_temporal_stride,
                        )
                        self.init_replay_buffer(
                            force_reseed=epoch in self.temporal_stride_transition
                        )
                    cur_replay_max_lead = self.get_current_replay_max_lead(epoch)
                    cur_replay_refresh_every = (
                        self.get_current_replay_refresh_every_n_microbatches(epoch)
                    )
                else:
                    if (
                        epoch == self.start_epoch
                        or epoch in self.step_transition
                        or epoch in self.temporal_stride_transition
                    ):
                        cur_step = self.get_current_step(epoch)
                        cur_temporal_stride = self.get_current_temporal_stride(epoch)
                        self.temporal_stride = cur_temporal_stride
                        self.init_data_loaders(cur_step, cur_temporal_stride)

                if isinstance(self.train_sampler, DistributedSampler):
                    self.train_sampler.set_epoch(epoch)
                if isinstance(self.val_sampler, DistributedSampler):
                    self.val_sampler.set_epoch(epoch)

                start_batch_in_epoch = (
                    self.start_batch_in_epoch if epoch == self.start_epoch else 0
                )
                if start_batch_in_epoch > 0:
                    logger.info(
                        f"Resuming epoch {epoch} from batch {start_batch_in_epoch}"
                    )
                self._last_completed_batch_in_epoch = start_batch_in_epoch - 1

                start_epoch_train_time = time.perf_counter()
                if self.replay_enabled:
                    train_stats = self.train_one_epoch_replay(
                        epoch,
                        start_batch_in_epoch,
                        cur_replay_max_lead,
                        cur_replay_refresh_every,
                    )
                else:
                    train_stats = self.train_one_epoch(epoch, start_batch_in_epoch)
                self.start_batch_in_epoch = 0
                end_epoch_train_time = time.perf_counter()

                if self._stop_requested:
                    reason = self._stop_reason or "stop requested"
                    self.save_emergency_checkpoint(
                        epoch=epoch,
                        batch_in_epoch=self._last_completed_batch_in_epoch,
                        reason=reason,
                    )
                    raise GracefulStopRequested(
                        f"Stopping after emergency checkpoint ({reason})."
                    )

                val_stats = self.validate_one_epoch(epoch)
                end_epoch_val_time = time.perf_counter()

                if -1 in self.inference_epochs or epoch in self.inference_epochs:
                    inf_stats = self.inference_one_epoch(epoch)
                    end_epoch_inf_time = time.perf_counter()
                else:
                    inf_stats = {}
                    end_epoch_inf_time = None

                train_loss = train_stats["train/mean/loss"]
                v_loss = val_stats["val/mean/loss"]
                inf_loss = inf_stats.get(
                    "inference/time_mean_norm/rmse/channel_mean", None
                )

                logger.info(f"Achieved Train Loss = {train_loss:.3f}")
                logger.info(f"Achieved Validation Loss = {v_loss:.3f}")
                if inf_loss is not None:
                    logger.info(f"Achieved Inference Loss = {inf_loss:.3f}")

                if is_main_process():
                    self.save_all_checkpoints(epoch, v_loss, inf_loss)
                if self.replay_enabled:
                    self.save_replay_buffer_sidecars_for_epoch(epoch)

                time_elapsed = time.perf_counter() - start_epoch_train_time

                log_stats = {
                    **train_stats,
                    **val_stats,
                    **inf_stats,
                    "epoch": epoch,
                    "lr_multiplier": (
                        self.scheduler.applied_multiplier
                        if isinstance(self.scheduler, EpochMultiplierScheduler)
                        else 1.0
                    ),
                    "epoch_train_seconds": end_epoch_train_time
                    - start_epoch_train_time,
                    "epoch_validation_seconds": end_epoch_val_time
                    - end_epoch_train_time,
                    "epoch_total_seconds": time_elapsed,
                }

                if end_epoch_inf_time is not None:
                    log_stats["epoch_inference_seconds"] = (
                        end_epoch_inf_time - end_epoch_val_time
                    )

                if is_main_process():
                    self.wandb_logger.log(log_stats, step=self.num_batches_seen)
        except GracefulStopRequested as e:
            logger.warning(str(e))
        except Exception:
            if self._active_epoch is not None:
                self.save_emergency_checkpoint(
                    epoch=self._active_epoch,
                    batch_in_epoch=self._last_completed_batch_in_epoch,
                    reason="uncaught_exception",
                )
            raise
        finally:
            total_time = time.perf_counter() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logger.info(f"Training time {total_time_str}")
            self.finish()

    def _scatter_domain_batch(
        self,
        data: TrainData,
        *,
        expected_steps: int,
    ) -> TrainData:
        """Scatter one leader-owned curriculum batch over the spatial mesh."""
        if self.dp_ctx is None:
            return data
        if self.dp_ctx.is_domain_leader and len(data) != expected_steps:
            raise ValueError(
                f"Expected {expected_steps} rollout steps from the domain leader, "
                f"got {len(data)}."
            )

        sharded = TrainData(self.num_out)
        for step in range(expected_steps):
            input_tensor = data.get_input(step) if self.dp_ctx.is_domain_leader else None
            label = data.get_label(step) if self.dp_ctx.is_domain_leader else None
            sharded.append(
                self.dp_ctx.scatter_spatial(input_tensor, ndim=4),
                self.dp_ctx.scatter_spatial(label, ndim=4),
            )
        if self.dp_ctx.is_domain_leader:
            sharded.load_stats = data.load_stats
            sharded.source_indices = list(data.source_indices)
        return sharded

    def _materialize_train_output(
        self, output: TrainBatchOutput
    ) -> TrainBatchOutput:
        """Gather replicated loss artifacts into ordinary tensors for logging."""
        if self.dp_ctx is None:
            return output
        return TrainBatchOutput(
            self.dp_ctx.gather(output.loss.detach()),
            self.dp_ctx.gather(output.loss_per_channel.detach()),
        )

    def _materialize_val_output(self, output: ValBatchOutput) -> ValBatchOutput:
        """Gather one-step validation artifacts for the existing aggregators."""
        if self.dp_ctx is None:
            return output
        return ValBatchOutput(
            self.dp_ctx.gather(output.loss.detach()),
            self.dp_ctx.gather(output.loss_per_channel.detach()),
            self.dp_ctx.gather(output.input_data),
            self.dp_ctx.gather(output.target_data),
            self.dp_ctx.gather(output.gen_data),
        )

    def train_one_epoch(self, epoch, start_batch_in_epoch: int = 0):
        self.model.train(True)
        train_aggregator = Aggregator.get_train_aggregator()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Training Epoch: [{epoch}]"

        total_batches = len(self.train_loader)
        start_batch_in_epoch = max(0, start_batch_in_epoch)
        active_train_loader = self.train_loader
        if start_batch_in_epoch > 0:
            active_train_loader = self._train_loader_with_batch_offset(
                start_batch_in_epoch
            )
        processed_batches = 0

        # Ensure gradients are zeroed at the start of the epoch so we don't
        # accidentally accumulate leftovers from checkpoint/loading.
        self.optimizer.zero_grad()

        # Calculate how many batches will be in the final incomplete accumulation cycle (if any)
        remaining_batches = total_batches % self.gradient_accumulation_steps
        final_cycle_start = (
            total_batches - remaining_batches
            if remaining_batches > 0
            else total_batches
        )
        ddp_model = (
            self.model
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else None
        )
        use_no_sync = (
            ddp_model is not None
            and self.ddp_use_no_sync_for_accumulation
            and self.gradient_accumulation_steps > 1
            and not self.ddp_static_graph
        )

        for data_iter_step, data in enumerate(
            metric_logger.log_every(
                active_train_loader,
                1,
                header,
                start_index=start_batch_in_epoch,
                total_steps=total_batches,
            )
        ):
            global_data_iter_step = start_batch_in_epoch + data_iter_step

            data = self._scatter_domain_batch(
                data,
                expected_steps=self.current_train_steps,
            )

            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            in_final_cycle = (
                global_data_iter_step + 1 > final_cycle_start
            ) and remaining_batches > 0

            # Determine the actual number of microbatches in this accumulation cycle
            if in_final_cycle:
                r = remaining_batches
            else:
                r = self.gradient_accumulation_steps

            if self.num_batches_seen == 0:
                get_model_summary(
                    self.model,
                    data if self.debug else None,
                    self.debug,
                )

            is_last = global_data_iter_step + 1 == total_batches
            should_step = (
                global_data_iter_step + 1
            ) % self.gradient_accumulation_steps == 0
            sync_gradients = should_step or is_last

            sync_context: contextlib.AbstractContextManager
            if use_no_sync and not sync_gradients:
                sync_context = ddp_model.no_sync()
            else:
                sync_context = contextlib.nullcontext()

            with sync_context:
                TO: TrainBatchOutput = Stepper.train_batch(
                    self.model, data, self.loss_fn
                )
                # Scale loss by the actual number of microbatches that will be accumulated
                scaled_loss = TO.loss / r
                scaled_loss.backward()

            logging_output = self._materialize_train_output(TO)
            train_aggregator.record_batch(logging_output)
            processed_batches += 1

            self.num_batches_seen += 1

            # Step optimizer after accumulating enough batches or at the end
            if sync_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._ema(model=self.model)

            lr = (
                self.optimizer.param_groups[-1]["lr"]
                if self.scheduler is None
                else self.scheduler.get_last_lr()[0]
            )

            with torch.no_grad():
                # Reduce losses
                if self.dp_ctx is not None:
                    # Loss reductions are already global over the spatial mesh.
                    # Gathering strips the ShardTensor wrapper for log consumers.
                    loss_value_reduce = logging_output.loss
                    loss_per_channel_reduce = logging_output.loss_per_channel
                elif sync_gradients:
                    loss_value_reduce = all_reduce_mean(TO.loss.detach())
                    loss_per_channel_reduce = all_reduce_mean(
                        TO.loss_per_channel.detach()
                    )
                else:
                    # Skip cross-rank synchronization on intermediate microbatches.
                    # This keeps communication low during gradient accumulation.
                    loss_value_reduce = TO.loss.detach()
                    loss_per_channel_reduce = TO.loss_per_channel.detach()
                metrics = {
                    "train/batch/loss": loss_value_reduce,
                    "train/batch/lr": lr,
                    "train/batch/ema_cur_decay": self._ema.cur_decay.item(),
                    **get_channel_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                    **get_depth_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                    **get_variable_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                    "train/batch/data_load_time": metric_logger.meters[
                        "data_load_time"
                    ].value,
                    "train/batch/data_wait_time": metric_logger.meters[
                        "data_wait_time"
                    ].value,
                }

                if loss_scale_per_channel_fn := getattr(
                    self.loss_fn, "loss_scale_per_channel", None
                ):
                    loss_scale_per_channel = loss_scale_per_channel_fn()
                    # Reshape from time-major channels to [hist, var] and
                    # average along the history dimension.
                    loss_per_channel = logging_output.loss_per_channel.reshape(
                        -1, loss_scale_per_channel.shape[0]
                    ).mean(dim=0)

                    unscaled_loss_per_channel = (
                        loss_per_channel / loss_scale_per_channel
                    )
                    unscaled_loss = torch.mean(unscaled_loss_per_channel)

                    metrics.update(
                        {
                            **get_channel_loss_scale_dict(
                                label="train",
                                loss_scale_per_channel=loss_scale_per_channel,
                            ),
                            **get_channel_loss_dict(
                                label="train",
                                loss_per_channel=unscaled_loss_per_channel,
                                loss_name="loss_unscaled",
                            ),
                            "train/batch/loss_unscaled": unscaled_loss,
                        }
                    )

            if (it_time := metric_logger.meters["iter_time"]).count > 0:
                metrics["train/batch/iter_time"] = it_time.value

            if (
                self.slow_batch_log_threshold_seconds > 0
                and data.load_stats is not None
                and data.load_stats.load_time_seconds
                >= self.slow_batch_log_threshold_seconds
            ):
                logger.warning(
                    "Slow batch load detected: rank=%s epoch=%s batch=%s "
                    "load_time=%.3fs threshold=%.3fs source_indices=%s",
                    get_rank(),
                    epoch,
                    global_data_iter_step,
                    data.load_stats.load_time_seconds,
                    self.slow_batch_log_threshold_seconds,
                    data.source_indices,
                )

            self.wandb_logger.log(metrics, step=self.num_batches_seen)

            metric_logger.update(loss=loss_value_reduce.item())
            metric_logger.update(lr=lr)

            self._call_loss_update(data)

            self.profiler.after_batch(self.num_batches_seen)
            self._last_completed_batch_in_epoch = global_data_iter_step

            if sync_gradients:
                self._maybe_save_periodic_emergency_checkpoint(
                    epoch=epoch,
                    batch_in_epoch=global_data_iter_step,
                )

            if self._stop_requested and sync_gradients:
                reason = self._stop_reason or "stop requested"
                self.save_emergency_checkpoint(
                    epoch=epoch,
                    batch_in_epoch=global_data_iter_step,
                    reason=reason,
                )
                raise GracefulStopRequested(
                    f"Received stop request during epoch {epoch} at "
                    f"batch {global_data_iter_step}."
                )

        if self.scheduler is not None:
            self.scheduler.step()

        if processed_batches == 0:
            logger.warning(
                "No training batches were processed in epoch %s "
                "(start_batch_in_epoch=%s, total_batches=%s).",
                epoch,
                start_batch_in_epoch,
                total_batches,
            )
            return {"train/mean/loss": float("nan")}

        logger.info(f"Aggregating train logs")
        return train_aggregator.get_logs()

    def train_one_epoch_replay(
        self,
        epoch: int,
        start_batch_in_epoch: int,
        max_lead_steps: int,
        refresh_every_n_microbatches: int,
    ):
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer was not initialized before training")

        self.model.train(True)
        train_aggregator = Aggregator.get_train_aggregator()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Replay Training Epoch: [{epoch}]"

        total_batches = self.replay_cfg.steps_per_epoch
        start_batch_in_epoch = max(0, start_batch_in_epoch)
        replay_batches = _ReplayPrefetchPipeline(
            self,
            start_batch_in_epoch=start_batch_in_epoch,
            total_batches=total_batches,
            max_lead_steps=max_lead_steps,
            refresh_every_n_microbatches=refresh_every_n_microbatches,
        )
        processed_batches = 0

        self.optimizer.zero_grad()

        remaining_batches = total_batches % self.gradient_accumulation_steps
        final_cycle_start = (
            total_batches - remaining_batches
            if remaining_batches > 0
            else total_batches
        )
        ddp_model = (
            self.model
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else None
        )
        use_no_sync = (
            ddp_model is not None
            and self.ddp_use_no_sync_for_accumulation
            and self.gradient_accumulation_steps > 1
            and not self.ddp_static_graph
        )

        for data_iter_step, prepared in enumerate(
            metric_logger.log_every(
                replay_batches,
                1,
                header,
                start_index=start_batch_in_epoch,
                total_steps=total_batches,
            )
        ):
            global_data_iter_step = start_batch_in_epoch + data_iter_step

            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            in_final_cycle = (
                global_data_iter_step + 1 > final_cycle_start
            ) and remaining_batches > 0
            r = remaining_batches if in_final_cycle else self.gradient_accumulation_steps

            data = prepared.data
            replay_cursors = prepared.cursors

            if self.num_batches_seen == 0:
                get_model_summary(
                    self.model,
                    data if self.debug else None,
                    self.debug,
                )

            is_last = global_data_iter_step + 1 == total_batches
            should_step = (
                global_data_iter_step + 1
            ) % self.gradient_accumulation_steps == 0
            sync_gradients = should_step or is_last

            sync_context: contextlib.AbstractContextManager
            if use_no_sync and not sync_gradients:
                sync_context = ddp_model.no_sync()
            else:
                sync_context = contextlib.nullcontext()

            with sync_context:
                outputs = self.model(data)
                pred = outputs[0]
                label = data.get_label(0)
                loss_per_channel = self.loss_fn(pred, label)
                loss = torch.mean(loss_per_channel)
                TO = TrainBatchOutput(loss, loss_per_channel)
                scaled_loss = TO.loss / r
                scaled_loss.backward()

            cap_refreshes, scheduled_refreshes = self.apply_replay_prefetch_updates(
                prepared,
                pred,
            )
            replay_batches.complete(prepared)

            train_aggregator.record_batch(TO)
            processed_batches += 1

            self.num_batches_seen += 1

            if sync_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._ema(model=self.model)

            lr = (
                self.optimizer.param_groups[-1]["lr"]
                if self.scheduler is None
                else self.scheduler.get_last_lr()[0]
            )

            with torch.no_grad():
                if sync_gradients:
                    loss_value_reduce = all_reduce_mean(TO.loss.detach())
                    loss_per_channel_reduce = all_reduce_mean(
                        TO.loss_per_channel.detach()
                    )
                else:
                    loss_value_reduce = TO.loss.detach()
                    loss_per_channel_reduce = TO.loss_per_channel.detach()
                lead_steps = torch.tensor(
                    [cursor.lead_step for cursor in replay_cursors],
                    device=self.device,
                    dtype=torch.float32,
                )
                metrics = {
                    "train/batch/loss": loss_value_reduce,
                    "train/batch/lr": lr,
                    "train/batch/ema_cur_decay": self._ema.cur_decay.item(),
                    "train/batch/replay_max_lead_steps": max_lead_steps,
                    "train/batch/replay_refresh_every_n_microbatches": (
                        refresh_every_n_microbatches
                    ),
                    "train/batch/replay_mean_lead_step": lead_steps.mean(),
                    "train/batch/replay_buffer_size": len(self.replay_buffer),
                    "train/batch/replay_cap_refreshes": cap_refreshes,
                    "train/batch/replay_scheduled_refreshes": scheduled_refreshes,
                    **get_channel_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                    **get_depth_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                    **get_variable_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                    "train/batch/data_load_time": metric_logger.meters[
                        "data_load_time"
                    ].value,
                    "train/batch/data_wait_time": metric_logger.meters[
                        "data_wait_time"
                    ].value,
                }

                if loss_scale_per_channel_fn := getattr(
                    self.loss_fn, "loss_scale_per_channel", None
                ):
                    loss_scale_per_channel = loss_scale_per_channel_fn()
                    loss_per_channel = TO.loss_per_channel.reshape(
                        -1, loss_scale_per_channel.shape[0]
                    ).mean(dim=0)

                    unscaled_loss_per_channel = (
                        loss_per_channel / loss_scale_per_channel
                    )
                    unscaled_loss = torch.mean(unscaled_loss_per_channel)

                    metrics.update(
                        {
                            **get_channel_loss_scale_dict(
                                label="train",
                                loss_scale_per_channel=loss_scale_per_channel,
                            ),
                            **get_channel_loss_dict(
                                label="train",
                                loss_per_channel=unscaled_loss_per_channel,
                                loss_name="loss_unscaled",
                            ),
                            "train/batch/loss_unscaled": unscaled_loss,
                        }
                    )

            if (it_time := metric_logger.meters["iter_time"]).count > 0:
                metrics["train/batch/iter_time"] = it_time.value

            self.wandb_logger.log(metrics, step=self.num_batches_seen)

            metric_logger.update(loss=loss_value_reduce.item())
            metric_logger.update(lr=lr)

            self._call_loss_update(data)

            self.profiler.after_batch(self.num_batches_seen)
            self._last_completed_batch_in_epoch = global_data_iter_step

            if sync_gradients:
                self._maybe_save_periodic_emergency_checkpoint(
                    epoch=epoch,
                    batch_in_epoch=global_data_iter_step,
                )

            if self._stop_requested and sync_gradients:
                reason = self._stop_reason or "stop requested"
                self.save_emergency_checkpoint(
                    epoch=epoch,
                    batch_in_epoch=global_data_iter_step,
                    reason=reason,
                )
                raise GracefulStopRequested(
                    f"Received stop request during epoch {epoch} at "
                    f"batch {global_data_iter_step}."
                )

        if self.scheduler is not None:
            self.scheduler.step()

        if processed_batches == 0:
            logger.warning(
                "No replay training batches were processed in epoch %s "
                "(start_batch_in_epoch=%s, total_batches=%s).",
                epoch,
                start_batch_in_epoch,
                total_batches,
            )
            return {"train/mean/loss": float("nan")}

        logger.info("Aggregating replay train logs")
        return train_aggregator.get_logs()

    def _train_loader_with_batch_offset(
        self, start_batch_in_epoch: int
    ) -> TrainDataLoader | _DomainFollowerLoader:
        """Build a train loader that starts at a specific batch index."""
        if isinstance(self.train_loader, _DomainFollowerLoader):
            return self.train_loader.with_offset(start_batch_in_epoch)
        raw_loader = self.train_loader._dataloader
        batch_sampler = _OffsetBatchSampler(raw_loader.batch_sampler, start_batch_in_epoch)
        loader_kwargs: dict[str, Any] = {
            "dataset": raw_loader.dataset,
            "batch_sampler": batch_sampler,
            "num_workers": raw_loader.num_workers,
            "collate_fn": raw_loader.collate_fn,
            "pin_memory": raw_loader.pin_memory,
            "timeout": raw_loader.timeout,
            "worker_init_fn": raw_loader.worker_init_fn,
            "multiprocessing_context": raw_loader.multiprocessing_context,
            "generator": raw_loader.generator,
            "persistent_workers": raw_loader.persistent_workers,
        }
        if raw_loader.num_workers > 0:
            loader_kwargs["prefetch_factor"] = raw_loader.prefetch_factor
        if pin_memory_device := getattr(raw_loader, "pin_memory_device", ""):
            loader_kwargs["pin_memory_device"] = pin_memory_device
        offset_loader = DataLoader(**loader_kwargs)
        datasets = list(self.train_loader._datasets.values())
        return TrainDataLoader(offset_loader, datasets, self.device)

    def _call_loss_update(self, data: TrainData):
        # This is a separate function to ensure locals are dropped immediately after use
        if update := getattr(self.loss_fn, "update", None):
            with torch.no_grad():
                single_step_data = TrainData(data.num_prognostic_channels)
                # Each entry in data is one step in a rollout.
                input, label = data[0]
                single_step_data.append(input, label)
                pred = self.model(single_step_data)
                update(pred[0], label)

    def replay_prefetch_horizon(self) -> int:
        worker_horizon = (
            max(1, self.num_workers * self.prefetch_factor)
            if self.num_workers > 0
            else 1
        )
        slot_horizon = max(
            1,
            self.replay_cfg.buffer_size // max(self.batch_size, 1) - 1,
        )
        return max(1, min(worker_horizon, slot_horizon))

    def consume_replay_planner_resume_state(self) -> dict[str, Any] | None:
        state = self._replay_resume_planner_state
        self._replay_resume_planner_state = None
        return state

    def sample_replay_seed_cursor(self) -> ReplayCursor:
        if not self.train_datasets:
            raise RuntimeError("Cannot seed replay buffer before train datasets exist")
        dataset_lengths = [len(dataset) for dataset in self.train_datasets]
        total_length = sum(dataset_lengths)
        if total_length == 0:
            raise RuntimeError("Cannot seed replay buffer from empty datasets")
        flat_source_index = int(
            torch.randint(
                total_length,
                (1,),
                generator=self.replay_generator,
                device="cpu",
            ).item()
        )
        dataset_index = 0
        source_index = flat_source_index
        for candidate_index, dataset_length in enumerate(dataset_lengths):
            if source_index < dataset_length:
                dataset_index = candidate_index
                break
            source_index -= dataset_length
        dataset = self.train_datasets[dataset_index]
        return ReplayCursor(
            dataset_index=dataset_index,
            source_index=source_index,
            lead_step=0,
            stride=dataset.stride,
            temporal_stride=dataset.temporal_stride,
        )

    def plan_replay_batch(
        self,
        *,
        global_batch_index: int,
        max_lead_steps: int,
        refresh_every_n_microbatches: int,
        exclude_reserved: set[int],
    ) -> ReplayBatchRequest:
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer is not initialized")

        replay_indices = self.replay_buffer.sample_indices(
            self.batch_size,
            max_lead_steps=max_lead_steps,
            exclude_reserved=exclude_reserved,
        )
        train_slots = tuple(
            ReplayBatchSlot(
                replay_index=replay_index,
                cursor=self.replay_buffer.entries[replay_index].cursor,
            )
            for replay_index in replay_indices
        )

        seed_slots: list[ReplaySeedSlot] = []
        for slot in train_slots:
            if slot.cursor.lead_step + 1 >= max_lead_steps:
                seed_slots.append(
                    ReplaySeedSlot(
                        replay_index=slot.replay_index,
                        cursor=self.sample_replay_seed_cursor(),
                        reason="cap",
                    )
                )

        reserved_for_request = {slot.replay_index for slot in train_slots}
        if (
            global_batch_index + 1
        ) % refresh_every_n_microbatches == 0:
            refresh_indices = self.replay_buffer.random_indices(
                self.batch_size,
                exclude_reserved=exclude_reserved | reserved_for_request,
            )
            seen_refresh_indices: set[int] = set()
            for replay_index in refresh_indices:
                if replay_index in seen_refresh_indices:
                    continue
                seen_refresh_indices.add(replay_index)
                seed_slots.append(
                    ReplaySeedSlot(
                        replay_index=replay_index,
                        cursor=self.sample_replay_seed_cursor(),
                        reason="scheduled",
                    )
                )

        return ReplayBatchRequest(
            request_id=global_batch_index,
            train_slots=train_slots,
            seed_slots=tuple(seed_slots),
            temporal_bundle_size=1,
        )

    def prepare_raw_replay_batch(
        self,
        raw_batch: RawReplayBatch,
        *,
        ready_event: torch.cuda.Event | None,
    ) -> _ReplayPreparedBatch:
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer is not initialized")

        inputs = []
        labels = []
        cursors = []
        for slot, transition in zip(
            raw_batch.request.train_slots,
            raw_batch.train_transitions,
            strict=True,
        ):
            entry = self.replay_buffer.entries[slot.replay_index]
            if entry.cursor != slot.cursor:
                raise RuntimeError(
                    "Replay buffer slot changed while a prefetched request was "
                    "in flight. This indicates a reservation bug."
                )
            dataset = self.train_datasets[slot.cursor.dataset_index]
            self.validate_replay_cursor(slot.cursor, dataset)
            input, label = self._raw_replay_train_transition_to_example(
                dataset,
                transition,
                entry,
            )
            inputs.append(input)
            labels.append(label)
            cursors.append(slot.cursor)

        seed_entries: dict[int, ReplayEntry] = {}
        for slot, transition in zip(
            raw_batch.request.seed_slots,
            raw_batch.seed_transitions,
            strict=True,
        ):
            dataset = self.train_datasets[slot.cursor.dataset_index]
            self.validate_replay_cursor(slot.cursor, dataset)
            state = self._raw_replay_seed_transition_to_state(dataset, transition)
            seed_entries[slot.replay_index] = self._stage_replay_state_for_buffer(
                state,
                slot.cursor,
            )

        train_data = TrainData(self.num_out)
        if inputs:
            train_data.append(torch.cat(inputs, dim=0), torch.cat(labels, dim=0))
        train_data.load_stats = raw_batch.load_stats
        train_data.source_indices = [
            slot.cursor.source_index for slot in raw_batch.request.train_slots
        ]
        return _ReplayPreparedBatch(
            request=raw_batch.request,
            data=train_data,
            cursors=cursors,
            seed_entries=seed_entries,
            load_stats=raw_batch.load_stats,
            ready_event=ready_event,
        )

    def _raw_replay_train_transition_to_example(
        self,
        dataset: TorchTrainDataset,
        transition: RawReplayTransition,
        entry: ReplayEntry,
    ):
        if transition.target_prognostic is None or transition.boundary is None:
            raise RuntimeError("Replay train transition is missing target or boundary")
        label = dataset._prep_tensor_steps(
            transition.target_prognostic.unsqueeze(0).to(
                device=self.device,
                non_blocking=True,
            )
        ).to(dtype=torch.float32)
        boundary = dataset._prep_boundary_steps(
            transition.boundary.unsqueeze(0).to(
                device=self.device,
                non_blocking=True,
            )
        )
        self._wait_replay_entry_ready(entry)
        prognostic_state = entry.state
        if prognostic_state.ndim == 3:
            prognostic_state = prognostic_state.unsqueeze(0)
        if prognostic_state.shape[1:] != (dataset.num_prognostic_channels, *boundary.shape[-2:]):
            raise ValueError(
                "Replay prognostic state shape does not match model input: "
                f"{prognostic_state.shape} vs boundary spatial shape {boundary.shape}"
            )
        prognostic_state = prognostic_state.to(
            device=boundary.device,
            dtype=boundary.dtype,
            non_blocking=True,
        )
        input = torch.cat((prognostic_state, boundary), dim=1)
        return input, label

    def _wait_replay_entry_ready(self, entry: ReplayEntry) -> None:
        if entry.ready_event is None:
            return
        if self.device.type == "cuda":
            torch.cuda.current_stream().wait_event(entry.ready_event)
        else:
            entry.ready_event.synchronize()

    def _stage_replay_state_for_buffer(
        self,
        state: torch.Tensor,
        cursor: ReplayCursor,
    ) -> ReplayEntry:
        source = state.detach()
        if (
            source.device.type == "cuda"
            and self.pin_mem
            and torch.cuda.is_available()
        ):
            assert self.replay_copy_stream is not None
            cpu_state = torch.empty(
                source.shape,
                device="cpu",
                dtype=self.replay_storage_dtype,
                pin_memory=True,
            )
            current_stream = torch.cuda.current_stream()
            self.replay_copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.replay_copy_stream):
                cpu_state.copy_(source, non_blocking=True)
                source.record_stream(self.replay_copy_stream)
                ready_event = torch.cuda.Event()
                ready_event.record(self.replay_copy_stream)
            return ReplayEntry(
                state=cpu_state,
                cursor=cursor,
                ready_event=ready_event,
            )

        return ReplayEntry(state=source, cursor=cursor)

    def _raw_replay_seed_transition_to_state(
        self,
        dataset: TorchTrainDataset,
        transition: RawReplayTransition,
    ) -> torch.Tensor:
        if transition.seed_prognostic is None:
            raise RuntimeError("Replay seed transition is missing seed prognostic")
        state = dataset._prep_tensor_steps(
            transition.seed_prognostic.unsqueeze(0).to(
                device=self.device,
                non_blocking=True,
            )
        )
        return dataset.remask_prognostic_state(state)[0]

    def apply_replay_prefetch_updates(
        self,
        prepared: _ReplayPreparedBatch,
        pred: torch.Tensor,
    ) -> tuple[int, int]:
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer is not initialized")

        cap_refreshes = 0
        scheduled_refreshes = 0
        seed_reasons = {
            slot.replay_index: slot.reason for slot in prepared.request.seed_slots
        }
        with torch.no_grad():
            for batch_index, slot in enumerate(prepared.request.train_slots):
                seed_entry = prepared.seed_entries.get(slot.replay_index)
                if seed_entry is not None and seed_reasons[slot.replay_index] == "cap":
                    self.replay_buffer.replace(slot.replay_index, seed_entry)
                    cap_refreshes += 1
                    continue

                dataset = self.train_datasets[slot.cursor.dataset_index]
                state = dataset.remask_prognostic_state(pred[batch_index])
                self.replay_buffer.replace(
                    slot.replay_index,
                    self._stage_replay_state_for_buffer(
                        state=state,
                        cursor=slot.cursor.advance(),
                    ),
                )

            for slot in prepared.request.seed_slots:
                if slot.reason != "scheduled":
                    continue
                seed_entry = prepared.seed_entries[slot.replay_index]
                self.replay_buffer.replace(slot.replay_index, seed_entry)
                scheduled_refreshes += 1

        return cap_refreshes, scheduled_refreshes

    def init_replay_buffer(self, *, force_reseed: bool = False) -> None:
        if self.replay_buffer is not None and not force_reseed:
            return

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.replay_cfg.buffer_size,
            storage_dtype=self.replay_storage_dtype,
            generator=self.replay_generator,
            pin_memory=self.pin_mem,
        )

        loaded = False
        if (
            not force_reseed
            and not self._replay_resume_consumed
            and self.replay_resume_checkpoint_path is not None
        ):
            loaded = self.load_replay_buffer_sidecar(self.replay_resume_checkpoint_path)
            self._replay_resume_consumed = True

        if not loaded:
            if force_reseed:
                logger.warning(
                    "Reseeding replay buffer because temporal_stride changed."
                )
            self.seed_replay_buffer()
        elif not self.replay_buffer.is_full:
            logger.warning(
                "Loaded replay buffer has %s/%s entries; filling the remainder "
                "from fresh gold data.",
                len(self.replay_buffer),
                self.replay_cfg.buffer_size,
            )
            self.seed_replay_buffer()

        active_refresh_every = self.get_current_replay_refresh_every_n_microbatches(
            self._active_epoch or self.start_epoch,
            log=False,
        )
        fresh_per_rank = self.replay_fresh_gold_samples_per_epoch(
            active_refresh_every
        )
        logger.info(
            "Replay buffer ready: size=%s storage_dtype=%s refresh_batch_size=%s "
            "steps_per_epoch=%s active_refresh_every_n_microbatches=%s "
            "configured_refresh_every_n_microbatches=%s "
            "prefetch_horizon=%s "
            "fresh_gold_samples_per_rank_per_epoch≈%.1f "
            "fresh_gold_samples_global_per_epoch≈%.1f",
            len(self.replay_buffer),
            self.replay_storage_dtype,
            self.batch_size,
            self.replay_cfg.steps_per_epoch,
            active_refresh_every,
            self.replay_cfg.refresh_every_n_microbatches,
            self.replay_prefetch_horizon(),
            fresh_per_rank,
            fresh_per_rank * get_world_size(),
        )

    def seed_replay_buffer(self) -> None:
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer must be created before seeding")
        needed = self.replay_buffer.buffer_size - len(self.replay_buffer)
        if needed <= 0:
            return
        for entry in self.load_replay_seed_entries(needed):
            self.replay_buffer.append(entry)

    def load_replay_seed_entries(self, count: int) -> list[ReplayEntry]:
        if count <= 0:
            return []

        next_replay_index = len(self.replay_buffer.entries) if self.replay_buffer else 0
        remaining = count
        request_id = 0
        entries_by_index: dict[int, ReplayEntry] = {}
        while remaining > 0:
            group_size = min(self.batch_size, remaining)
            seed_slots = []
            for _ in range(group_size):
                seed_slots.append(
                    ReplaySeedSlot(
                        replay_index=next_replay_index,
                        cursor=self.sample_replay_seed_cursor(),
                        reason="seed",
                    )
                )
                next_replay_index += 1
            request = ReplayBatchRequest(
                request_id=request_id,
                train_slots=(),
                seed_slots=tuple(seed_slots),
                temporal_bundle_size=1,
            )
            raw_batch = RawReplayBatch(
                request=request,
                train_transitions=[],
                seed_transitions=[
                    self.train_datasets[slot.cursor.dataset_index]
                    .get_raw_replay_seed_transition(
                        dataset_index=slot.cursor.dataset_index,
                        source_index=slot.cursor.source_index,
                        lead_step=slot.cursor.lead_step,
                    )
                    for slot in request.seed_slots
                ],
            )
            prepared = self.prepare_raw_seed_batch(raw_batch)
            entries_by_index.update(prepared.seed_entries)
            remaining -= group_size
            request_id += 1

        first_index = len(self.replay_buffer.entries) if self.replay_buffer else 0
        return [
            entries_by_index[replay_index]
            for replay_index in range(first_index, first_index + count)
        ]

    def prepare_raw_seed_batch(self, raw_batch: RawReplayBatch) -> _ReplayPreparedBatch:
        if self.device.type != "cuda":
            return self.prepare_raw_replay_batch(raw_batch, ready_event=None)

        assert self.replay_copy_stream is not None
        with torch.cuda.stream(self.replay_copy_stream):
            prepared = self.prepare_raw_replay_batch(raw_batch, ready_event=None)
            prepared.ready_event = torch.cuda.Event()
            prepared.ready_event.record(self.replay_copy_stream)
        prepared.wait_ready()
        return prepared

    @staticmethod
    def validate_replay_cursor(
        cursor: ReplayCursor,
        dataset: TorchTrainDataset,
    ) -> None:
        if cursor.stride != dataset.stride:
            raise RuntimeError(
                "Replay cursor stride does not match its dataset: "
                f"{cursor.stride} vs {dataset.stride}"
            )
        if cursor.temporal_stride != dataset.temporal_stride:
            raise RuntimeError(
                "Replay cursor temporal_stride does not match its dataset: "
                f"{cursor.temporal_stride} vs {dataset.temporal_stride}"
            )

    def replay_dataset_signature(self) -> list[dict[str, int]]:
        return [
            {
                "stride": dataset.stride,
                "temporal_stride": dataset.temporal_stride,
                "length": len(dataset),
            }
            for dataset in self.train_datasets
        ]

    def save_replay_buffer_sidecars_for_epoch(self, epoch: int) -> None:
        if not self.replay_cfg.checkpoint_buffer:
            return
        self.save_replay_buffer_sidecar(self.ckpt_paths.latest_checkpoint_path, epoch)
        if epoch > 0 and epoch % self.save_freq == 0:
            self.save_replay_buffer_sidecar(
                self.ckpt_paths.latest_checkpoint_path_with_epoch(epoch),
                epoch,
            )

    def save_replay_buffer_sidecar(self, checkpoint_path: Path, epoch: int) -> None:
        if self.replay_buffer is None or not self.replay_cfg.checkpoint_buffer:
            return
        sidecar_path = replay_sidecar_path(checkpoint_path, get_rank())
        temp_dir = os.path.dirname(sidecar_path)
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as tmp:
            temporary_location = tmp.name
            state = self.replay_buffer.state_dict(
                world_size=get_world_size(),
                rank=get_rank(),
            )
            state["epoch"] = epoch
            state["num_batches_seen"] = self.num_batches_seen
            state["dataset_signature"] = self.replay_dataset_signature()
            planner_state = (
                self._replay_active_planner.state_dict()
                if self._replay_active_planner is not None
                else self._replay_active_planner_state
            )
            if planner_state is not None:
                state["planner_state"] = planner_state
            torch.save(state, temporary_location)
            os.replace(temporary_location, sidecar_path)

    def load_replay_buffer_sidecar(self, checkpoint_path: Path) -> bool:
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer must exist before loading sidecar")
        sidecar_path = replay_sidecar_path(checkpoint_path, get_rank())
        if not sidecar_path.exists():
            logger.warning(
                "Replay buffer sidecar %s was not found; reseeding from gold data.",
                sidecar_path,
            )
            return False
        state = torch.load(sidecar_path, map_location=torch.device("cpu"))
        saved_world_size = state.get("world_size")
        if saved_world_size != get_world_size():
            logger.warning(
                "Replay buffer sidecar was saved with world_size=%s but current "
                "world_size=%s; reseeding all rank-local replay buffers from gold.",
                saved_world_size,
                get_world_size(),
            )
            return False
        if state.get("dataset_signature") != self.replay_dataset_signature():
            logger.warning(
                "Replay buffer dataset signature changed on resume; reseeding "
                "from gold data."
            )
            return False
        self.replay_buffer.load_state_dict(state)
        self._replay_resume_planner_state = state.get("planner_state")
        logger.info("Loaded replay buffer sidecar from %s", sidecar_path)
        return True

    def validate_one_epoch(self, epoch):
        self.model.eval()

        val_aggregator = Aggregator.get_validation_aggregator(
            self.metadata,
            self.hist,
            self.area_weights,
            self.src.masks.prognostic.to(self.device),
            self.num_out,
            surface_snapshot=self.surface_snapshot,
        )
        metric_logger = MetricLogger(delimiter="  ")
        header = f"One-Step Validation Epoch: [{epoch}]"

        with torch.no_grad(), self._test_context():
            for data_iter_step, data in enumerate(
                metric_logger.log_every(self.val_loader, 1, header)
            ):
                if self.debug and (data_iter_step + 1) % 5 == 0:
                    break

                data = self._scatter_domain_batch(data, expected_steps=1)

                VO: ValBatchOutput = Stepper.validate_batch(
                    self.model, data, self.loss_fn
                )
                VO = self._materialize_val_output(VO)
                val_aggregator.record_validation_batch(VO)
                metric_logger.update(loss=VO.loss)

        logger.info(f"Aggregating validation logs")
        return val_aggregator.get_logs(label="val")

    def inference_one_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad(), self._test_context():
            for data_iter_step, (inference_dataset, num_steps) in enumerate(
                self.inference_loader
            ):
                inf_aggregator = Aggregator.get_inline_inference_aggregator(
                    num_steps,
                    self.metadata,
                    self.hist,
                    self.area_weights,
                    self.src.masks.prognostic.to(self.device),
                    self.num_out,
                    self.prognostic_var_names,
                )

                # TODO(jder): we need the underlying model so we can use forward_once;
                # see https://github.com/suryadheeshjith/Ocean_Emulator/issues/51
                Stepper.inference(
                    model=self.model.module
                    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                    else self.model,
                    dataset=inference_dataset,
                    inf_aggregator=inf_aggregator,
                    epoch=epoch,
                    num_model_steps_forward=min(
                        num_steps // 2, self.max_train_model_steps_forward
                    ),
                )

        logger.info(f"Aggregating inference logs")
        logs = inf_aggregator.get_summary_logs()
        return {f"inference/{k}": v for k, v in logs.items()}

    def get_current_step(self, epoch):
        """Determine the current step based on the epoch and transition points.

        Args:
            epoch (int): Current epoch number

        Returns:
            int: current_step
        """
        cur_step = self.steps[self._get_schedule_stage_index(epoch, self.step_transition)]
        if epoch == self.start_epoch:
            logger.info(f"Starting training at step {cur_step}")
        elif epoch in self.step_transition:
            logger.info(f"Transitioning to step {cur_step}")

        return cur_step

    @staticmethod
    def _get_schedule_stage_index(epoch: int, transition_epochs: list[int]) -> int:
        """Return the active stage index for an epoch-based transition schedule.

        A transition epoch is interpreted as the first epoch of the next stage.
        For example, with transition epochs [5, 9], epochs 1-4 map to stage 0,
        epochs 5-8 map to stage 1, and epoch 9 onward maps to stage 2.
        """
        return sum(epoch >= transition_epoch for transition_epoch in transition_epochs)

    def get_current_temporal_stride(self, epoch: int) -> int:
        """Determine the current temporal stride based on epoch transitions."""
        cur_temporal_stride = self.temporal_strides[
            self._get_schedule_stage_index(epoch, self.temporal_stride_transition)
        ]
        if epoch == self.start_epoch:
            logger.info(
                f"Starting training at temporal_stride {cur_temporal_stride}"
            )
        elif epoch in self.temporal_stride_transition:
            logger.info(
                f"Transitioning to temporal_stride {cur_temporal_stride}"
            )

        return cur_temporal_stride

    def get_current_replay_max_lead(self, epoch: int) -> int:
        cur_max_lead = self.replay_cfg.max_lead_steps[
            self._get_schedule_stage_index(epoch, self.replay_cfg.max_lead_transition)
        ]
        if epoch == self.start_epoch:
            logger.info(f"Starting replay training at max_lead_steps {cur_max_lead}")
        elif epoch in self.replay_cfg.max_lead_transition:
            logger.info(f"Transitioning replay max_lead_steps to {cur_max_lead}")
        return cur_max_lead

    def get_current_replay_refresh_every_n_microbatches(
        self,
        epoch: int,
        *,
        log: bool = True,
    ) -> int:
        refresh_values = self.replay_cfg.refresh_every_n_microbatches
        if isinstance(refresh_values, int):
            cur_refresh = refresh_values
        else:
            cur_refresh = refresh_values[
                self._get_schedule_stage_index(
                    epoch,
                    self.replay_cfg.refresh_every_n_microbatches_transition,
                )
            ]
        if log and epoch == self.start_epoch:
            logger.info(
                "Starting replay training at "
                f"refresh_every_n_microbatches {cur_refresh}"
            )
        elif log and epoch in self.replay_cfg.refresh_every_n_microbatches_transition:
            logger.info(
                "Transitioning replay refresh_every_n_microbatches "
                f"to {cur_refresh}"
            )
        return cur_refresh

    def replay_fresh_gold_samples_per_epoch(
        self,
        refresh_every_n_microbatches: int,
    ) -> float:
        return (
            self.replay_cfg.steps_per_epoch
            / refresh_every_n_microbatches
            * self.batch_size
        )

    def init_data_loaders(
        self,
        cur_step: int,
        cur_temporal_stride: int | None = None,
    ) -> None:
        """Initialize training and validation data loaders.

        Args:
            cur_step: Current training step size
            cur_temporal_stride: Current temporal stride
        """
        if cur_temporal_stride is None:
            cur_temporal_stride = self.temporal_stride
        self.current_train_steps = cur_step
        train_datasets = [
            TorchTrainDataset(
                src=self.src.slice(self.train_time),
                prognostic_var_names=self.prognostic_var_names,
                boundary_var_names=self.boundary_var_names,
                hist=self.hist,
                steps=cur_step,
                normalize_before_mask=self.normalize_before_mask,
                masked_fill_value=self.normalize_fill_value,
                stride=stride,
                temporal_stride=cur_temporal_stride,
                executor=self.executor,
            )
            for stride in self.data_stride
        ]
        self.train_datasets = train_datasets

        val_datasets = [
            TorchTrainDataset(
                src=self.src.slice(self.val_time),
                prognostic_var_names=self.prognostic_var_names,
                boundary_var_names=self.boundary_var_names,
                hist=self.hist,
                steps=1,  # current_step set to 1 for validation
                normalize_before_mask=self.normalize_before_mask,
                masked_fill_value=self.normalize_fill_value,
                stride=stride,
                temporal_stride=cur_temporal_stride,
                executor=self.executor,
            )
            for stride in self.data_stride
        ]
        self.val_datasets = val_datasets

        # Create datasets
        match self.loader_version:
            case TorchTrainDataset.FLAG:
                train_data: torch.utils.data.Dataset[RawTrainData] = ConcatDataset(
                    train_datasets
                )

                val_data: torch.utils.data.Dataset[RawTrainData] = ConcatDataset(
                    val_datasets
                )

            case _:
                raise NotImplementedError(
                    f"Loader version {self.loader_version} not supported."
                )

        logger.info("Instantiating torch loaders")

        if self.dp_ctx is not None:
            # One logical sample spans the whole cluster. Only the leader reads
            # it; followers use metadata-only iterators with matching lengths.
            self.train_sampler = (
                RandomSampler(train_data)
                if self.train_shuffle
                else SequentialSampler(train_data)
            )
            self.val_sampler = SequentialSampler(val_data)
        elif self.distributed is not None:
            self.train_sampler = DistributedSampler(
                train_data,
                shuffle=self.train_shuffle,
            )
            self.val_sampler = DistributedSampler(val_data, shuffle=False)
        else:
            self.train_sampler = (
                RandomSampler(train_data)
                if self.train_shuffle
                else SequentialSampler(train_data)
            )  # type: ignore
            self.val_sampler = RandomSampler(val_data)  # type: ignore

        match self.loader_version:
            case TorchTrainDataset.FLAG:
                collate_fn = collate_raw_train_data
            case _:
                raise NotImplementedError(
                    f"Collate function not defined for loader version "
                    f"{self.loader_version}"
                )

        # Create data loaders
        train_loader_kwargs: dict[str, Any] = {
            "dataset": train_data,
            "batch_size": self.batch_size,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_mem,
            "drop_last": True,
            "collate_fn": collate_fn,
            "multiprocessing_context": self.mp_context,
        }
        val_loader_kwargs: dict[str, Any] = {
            "dataset": val_data,
            "batch_size": self.batch_size,
            "sampler": self.val_sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_mem,
            "drop_last": False,
            "collate_fn": collate_fn,
            "multiprocessing_context": self.mp_context,
        }
        if self.num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = self.prefetch_factor
            val_loader_kwargs["prefetch_factor"] = self.prefetch_factor

        if self.dp_ctx is not None and not self.dp_ctx.is_domain_leader:
            train_batches = len(train_data) // self.batch_size
            val_batches = (len(val_data) + self.batch_size - 1) // self.batch_size
            self.train_loader = _DomainFollowerLoader(train_batches, self.num_out)
            self.val_loader = _DomainFollowerLoader(val_batches, self.num_out)
        else:
            train_dataloader = DataLoader(**train_loader_kwargs)
            val_dataloader = DataLoader(**val_loader_kwargs)

            # Wrap dataloaders to handle GPU post-processing on the domain leader
            # (or on every rank in the ordinary DDP path).
            self.train_loader = TrainDataLoader(
                train_dataloader, train_datasets, self.device
            )
            self.val_loader = TrainDataLoader(
                val_dataloader, val_datasets, self.device
            )

    def _install_signal_handlers(self) -> None:
        handled_signals = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGUSR1"):
            handled_signals.append(signal.SIGUSR1)

        for signum in handled_signals:
            signal.signal(signum, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        if self._stop_requested:
            return

        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"signal_{signum}"

        self._stop_requested = True
        self._stop_reason = signame
        logger.warning(
            f"Received {signame}; saving emergency minibatch checkpoint at "
            "next safe optimization step."
        )

    def _maybe_save_periodic_emergency_checkpoint(
        self,
        epoch: int,
        batch_in_epoch: int,
    ) -> None:
        if self._next_emergency_checkpoint_time is None:
            return

        now = time.perf_counter()
        if now < self._next_emergency_checkpoint_time:
            return

        self.save_emergency_checkpoint(
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            reason="periodic_interval",
        )

        while self._next_emergency_checkpoint_time <= now:
            self._next_emergency_checkpoint_time += (
                self.emergency_checkpoint_interval_seconds
            )

    def save_emergency_checkpoint(
        self,
        epoch: int,
        batch_in_epoch: int,
        reason: str,
    ) -> Path | None:
        batch_in_epoch = max(-1, batch_in_epoch)
        checkpoint_path = (
            self.ckpt_paths.latest_batch_checkpoint_path
            if is_main_process()
            else self.ckpt_paths.latest_batch_checkpoint_path_for_rank(get_rank())
        )

        try:
            self.save_checkpoint(
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                batch_in_epoch=batch_in_epoch,
                epoch_complete=False,
                save_reason=reason,
            )
            if self.replay_enabled:
                self.save_replay_buffer_sidecar(
                    self.ckpt_paths.latest_batch_checkpoint_path,
                    epoch,
                )
            logger.warning(
                f"Saved emergency minibatch checkpoint to {checkpoint_path} "
                f"(epoch={epoch}, batch_in_epoch={batch_in_epoch}, reason={reason})"
            )
            return checkpoint_path
        except Exception:
            logger.exception(
                f"Failed to save emergency minibatch checkpoint to "
                f"{checkpoint_path}"
            )
            return None

    def save_all_checkpoints(self, epoch: int, v_loss: float, inf_loss: float):
        with self._test_context():
            is_best_val_loss = False
            if v_loss <= self.best_val_loss:
                logger.info(
                    f"Epoch validation loss ({v_loss:.3f}) is lower than "
                    f"previous best validation loss ({self.best_val_loss:.3f})."
                )
                logger.info(
                    "Saving lowest validation loss checkpoint to "
                    f"{self.ckpt_paths.best_validation_checkpoint_path}"
                )
                self.best_val_loss = v_loss
                is_best_val_loss = True  # wait until inference error is updated
            if inf_loss is not None and (inf_loss <= self.best_inf_loss):
                logger.info(
                    f"Epoch inference error ({inf_loss:.3f}) is lower than "
                    f"previous best inference error ({self.best_inf_loss:.3f})."
                )
                logger.info(
                    "Saving lowest inference error checkpoint to "
                    f"{self.ckpt_paths.best_inference_checkpoint_path}"
                )
                self.best_inf_loss = inf_loss
                self.save_checkpoint(
                    epoch, self.ckpt_paths.best_inference_checkpoint_path
                )
            if is_best_val_loss:
                self.save_checkpoint(
                    epoch, self.ckpt_paths.best_validation_checkpoint_path
                )

        logger.info(
            f"Saving latest checkpoint to {self.ckpt_paths.latest_checkpoint_path}"
        )
        self.save_checkpoint(epoch, self.ckpt_paths.latest_checkpoint_path)
        if epoch > 0 and epoch % self.save_freq == 0:
            path = self.ckpt_paths.latest_checkpoint_path_with_epoch(epoch)
            logger.info(f"Saving per-epoch checkpoint to {path}")
            self.save_checkpoint(epoch, path)

        logger.info(
            f"Saving latest EMA checkpoint to {self.ckpt_paths.ema_checkpoint_path}"
        )
        self.save_checkpoint(
            epoch,
            self.ckpt_paths.ema_checkpoint_path,
            for_inference=True,
        )

    def save_checkpoint(
        self,
        epoch: int,
        checkpoint_path: Path,
        for_inference: bool = False,
        batch_in_epoch: int | None = None,
        epoch_complete: bool = True,
        save_reason: str | None = None,
    ):
        if for_inference:
            with self._ema_context():
                model_state_dict = self._model_state_dict_for_save()
        else:
            model_state_dict = self._model_state_dict_for_save()

        # Create temporary file in the same directory as the target
        temp_dir = os.path.dirname(checkpoint_path)
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as tmp:
            temporary_location = tmp.name
            optimizer_state = self.optimizer.state_dict()
            ema_state = self._ema.get_state(include_ema_params=not for_inference)
            if self.dp_ctx is not None:
                optimizer_state = self._localize_checkpoint_state(optimizer_state)
                ema_state = self._localize_checkpoint_state(ema_state)
            checkpoint = {
                "model": model_state_dict,
                "optimizer": optimizer_state,
                "epoch": epoch,
                "epoch_complete": epoch_complete,
                "best_val_loss": self.best_val_loss,
                "best_inf_loss": self.best_inf_loss,
                "ema": ema_state,
                "num_batches_seen": self.num_batches_seen,
                "wandb_id": self.wandb_id,
                "wandb_name": self.wandb_name,
            }
            if batch_in_epoch is not None:
                checkpoint["batch_in_epoch"] = batch_in_epoch
            if save_reason is not None:
                checkpoint["save_reason"] = save_reason
            loss_state: dict[str, Any] | None = None
            if state_dict_fn := getattr(self.loss_fn, "state_dict", None):
                loss_state = state_dict_fn()

            if loss_state is not None:
                checkpoint["loss_fn_state"] = loss_state
            if self.scheduler:
                checkpoint["scheduler"] = self.scheduler.state_dict()

            torch.save(checkpoint, temporary_location)
            os.replace(temporary_location, checkpoint_path)

    def _model_state_dict_for_save(self):
        """Write portable dense weights when model parameters are replicated DTensors."""
        state = self.model.state_dict()
        if self.dp_ctx is None:
            return state

        dense_state = OrderedDict()
        for name, value in state.items():
            local = self.dp_ctx.local_tensor(value)
            dense_state[name] = local.detach().cpu()
        if hasattr(state, "_metadata"):
            dense_state._metadata = state._metadata
        return dense_state

    def _localize_checkpoint_state(self, value):
        """Recursively remove DTensor wrappers from optimizer and EMA state."""
        if isinstance(value, torch.Tensor):
            assert self.dp_ctx is not None
            return self.dp_ctx.local_tensor(value).detach().cpu()
        if isinstance(value, dict):
            return value.__class__(
                (key, self._localize_checkpoint_state(item))
                for key, item in value.items()
            )
        if isinstance(value, list):
            return [self._localize_checkpoint_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._localize_checkpoint_state(item) for item in value)
        return value

    def load_checkpoint(self, checkpoint_path, finetune=False):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))

        # Remove module prefix from state dict
        def remove_module_prefix(state_dict, prefix="module."):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.removeprefix(prefix)
                new_state_dict[name] = v
            return new_state_dict

        # Load model state
        model_state_dict = checkpoint["model"]
        new_state_dict = remove_module_prefix(model_state_dict)
        self.model.load_state_dict(new_state_dict)

        # Load EMA state
        model_ema_state_dict = checkpoint["ema"]
        new_ema_state_dict = remove_module_prefix(model_ema_state_dict)
        if "ema_params" in new_ema_state_dict:
            new_ema_state_dict["ema_params"] = remove_module_prefix(
                new_ema_state_dict["ema_params"], prefix="module"
            )
        self._ema = EMATracker.from_state(new_ema_state_dict, self.model)

        if not finetune:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            if load_state_dict_fn := getattr(self.loss_fn, "load_state_dict", None):
                assert "loss_fn_state" in checkpoint, (
                    f"Expected to load loss state for {self.loss_fn} but "
                    "no state found in checkpoint"
                )
                load_state_dict_fn(checkpoint["loss_fn_state"])

            epoch_complete = checkpoint.get("epoch_complete", True)
            if epoch_complete:
                self.start_epoch = checkpoint["epoch"] + 1
                self.start_batch_in_epoch = 0
            else:
                self.start_epoch = checkpoint["epoch"]
                self.start_batch_in_epoch = checkpoint.get("batch_in_epoch", -1) + 1

            self.wandb_id = checkpoint.get("wandb_id")
            self.wandb_name = checkpoint.get("wandb_name")
            self.num_batches_seen = checkpoint.get("num_batches_seen", 0)

            logger.info(f"Start Epoch: {self.start_epoch}")
            logger.info(f"Start Batch In Epoch: {self.start_batch_in_epoch}")
            logger.info(f"Wandb id: {self.wandb_id}")
            logger.info(f"Wandb name: {self.wandb_name}")
            logger.info(f"Optimizer LR: {self.optimizer.param_groups[-1]['lr']}")

            self.best_val_loss = checkpoint["best_val_loss"]
            self.best_inf_loss = checkpoint["best_inf_loss"]

    def is_wandb_enabled(self):
        return self.wandb_logger.enabled and is_main_process()

    @contextlib.contextmanager
    def _test_context(self):
        """
        The context for running validation/inference.
        In this context, the stepper uses the EMA model if
        `self.test_using_ema` is True.
        """
        if self.test_using_ema:
            with self._ema_context():
                yield
        else:
            yield

    @contextlib.contextmanager
    def _ema_context(self):
        """
        A context where the stepper uses the EMA model.
        """
        self._ema.store(parameters=self.model.parameters())
        self._ema.copy_to(model=self.model)
        try:
            yield
        finally:
            self._ema.restore(parameters=self.model.parameters())

    def finish(self):
        if self._finished:
            return
        self._finished = True
        if self.executor is not None:
            self.executor.shutdown()
        self.wandb_logger.finish()


def main():
    # Load config from YAML
    cfg = TrainConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()  # we do this first so logging can use them

    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    trainer = Trainer(cfg)

    try:
        trainer.run()
    except Exception as e:
        logger.exception("Training failed with an exception")
        raise e


if __name__ == "__main__":
    main()
