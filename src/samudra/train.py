# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import datetime
import itertools
import logging
import multiprocessing
import os
import resource
import tempfile
import time
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, assert_never

import dask
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from samudra import config
from samudra.aggregator import (
    Aggregator,
    MultiScaleValidateAggregator,
    ValidateAggregator,
)
from samudra.aggregator.loss import (
    get_channel_loss_dict,
    get_channel_loss_scale_dict,
    get_depth_loss_dict,
    get_variable_loss_dict,
)
from samudra.aggregator.train import RouteTrainAggregator
from samudra.backend import init_train_backend
from samudra.config import TrainConfig, TrainSchedule, build_loss_fn
from samudra.constants import (
    MAX_TRAIN_MODEL_STEPS_FORWARD,
    BoundaryVarNames,
    PrognosticVarNames,
    TensorMap,
)
from samudra.datasets import (
    InferenceDataset,
    InferenceDatasets,
    TorchTrainDataset,
    TrainBatchLoader,
    TrainData,
    close_pytorch_dataloader,
)
from samudra.models.base import BaseModel
from samudra.models.modules.decoder import coordinate_bilinear_resample
from samudra.models.samudra_multi import SamudraMulti
from samudra.stepper import (
    TrainBatchOutput,
    ValBatchOutput,
    ablate_boundary_forcing,
    run_rollout,
    train_batch,
    validate_batch,
)
from samudra.train_data_loader import build_train_batch_loader
from samudra.utils.data import CanonicalDataset, Normalize, get_inference_steps
from samudra.utils.device import using_gpu
from samudra.utils.distributed import (
    all_reduce_mean,
    destroy_distributed_mode,
    get_world_size,
    is_main_process,
    set_seed,
)
from samudra.utils.ema import EMATracker
from samudra.utils.logging import (
    MetricLogger,
    SmoothedValue,
    get_model_summary,
    handle_logging,
    handle_warnings,
)
from samudra.utils.loss import DynamicLoss, LossFnWithContext
from samudra.utils.samplers import (
    DistributedEquivalenceGroupBatchSampler,
    EquivalenceGroupBatchSampler,
)
from samudra.utils.train import CheckpointPaths, collate_inference_data
from samudra.utils.wandb import WandBLogger

logger = logging.getLogger(__name__)


def load_model_state_for_finetune(
    model: nn.Module,
    state_dict: OrderedDict[str, torch.Tensor],
    allowed_missing_prefixes: list[str],
) -> bool:
    """Load model state, permitting only explicitly allowlisted missing keys.

    Returns whether the load was partial. Unexpected checkpoint keys and missing
    keys outside the allowlist remain fatal so architecture mistakes fail loudly.
    """
    if not allowed_missing_prefixes:
        model.load_state_dict(state_dict)
        return False
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Finetune checkpoint has unexpected model keys: "
            + ", ".join(incompatible.unexpected_keys)
        )
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing:
        raise RuntimeError(
            "Finetune checkpoint is missing non-allowlisted model keys: "
            + ", ".join(disallowed_missing)
        )
    logger.info(
        "Initialized %d allowlisted model tensors outside the finetune checkpoint",
        len(incompatible.missing_keys),
    )
    return True


def freeze_model_parameters(model: nn.Module, prefixes: list[str]) -> tuple[int, int]:
    """Freeze parameters selected by explicit name prefixes.

    Returns the number of frozen parameter tensors and scalar parameters. Each
    configured prefix must match so a misspelled experimental control cannot run
    silently.
    """
    if not prefixes:
        return 0, 0
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("frozen_model_prefixes must be unique.")
    if any(not prefix for prefix in prefixes):
        raise ValueError("frozen_model_prefixes cannot contain an empty prefix.")

    matches = {prefix: 0 for prefix in prefixes}
    frozen_tensors = 0
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        matching_prefixes = [prefix for prefix in prefixes if name.startswith(prefix)]
        if not matching_prefixes:
            continue
        parameter.requires_grad_(False)
        frozen_tensors += 1
        frozen_parameters += parameter.numel()
        for prefix in matching_prefixes:
            matches[prefix] += 1

    unmatched = [prefix for prefix, count in matches.items() if count == 0]
    if unmatched:
        raise ValueError(
            "frozen_model_prefixes matched no parameters: " + ", ".join(unmatched)
        )
    return frozen_tensors, frozen_parameters


def should_log_validation_images(epoch: int, frequency: int) -> bool:
    """Return whether to log validation images for a 1-based training epoch."""
    if epoch < 1:
        raise ValueError(f"Epoch must be >= 1, got {epoch}")
    if frequency < 1:
        raise ValueError(
            f"Validation image log frequency must be >= 1, got {frequency}"
        )
    return (epoch - 1) % frequency == 0


def training_processor_depth(
    depths: list[int], epoch: int, batch_index: int, batches_per_epoch: int
) -> int:
    """Cycle processor depths continuously and identically on every rank."""
    if not depths or any(depth <= 0 for depth in depths):
        raise ValueError("Training processor depths must be positive and non-empty.")
    if epoch < 1 or batch_index < 0 or batches_per_epoch < 1:
        raise ValueError("Epoch and batch coordinates must describe a training batch.")
    return depths[((epoch - 1) * batches_per_epoch + batch_index) % len(depths)]


def configure_preemptible_resume(
    cfg: TrainConfig, latest_checkpoint_path: str | Path
) -> bool:
    """Switch a requeued finetune run to an exact training-state resume."""
    if not cfg.preemptible or not os.path.isfile(latest_checkpoint_path):
        return False

    cfg.resume_ckpt_path = str(latest_checkpoint_path)
    cfg.finetune = False
    return True


class Trainer:
    """Orchestrates the full model training loop.

    Handles initialization, distributed setup, checkpointing, learning rate
    scheduling, EMA, and Weights & Biases logging.
    """

    model: BaseModel | nn.parallel.DistributedDataParallel

    def __init__(self, cfg: TrainConfig) -> None:
        if cfg.validation_only and cfg.resume_ckpt_path is None:
            raise ValueError(
                "validation_only requires resume_ckpt_path so the audit evaluates "
                "an explicit model checkpoint."
            )
        if not cfg.finetune:
            cfg.finetune_allowed_missing_prefixes = []
        cfg.prepare_output_dirs()
        cfg.save_yaml(cfg.experiment.output_dir / "config.yaml")

        # Backend
        self.device, self.distributed = init_train_backend(cfg.backend)

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.pin_mem = False
        elif cfg.disk_mode:
            cfg.pin_mem = True

        # Distributed mode
        dask.config.set(scheduler="synchronous")

        # Set seeds
        set_seed(cfg.experiment.rand_seed)
        self.rand_seed = cfg.experiment.rand_seed

        # Getting prognostic and boundary variables
        self.dataset_spec = cfg.data.dataset.build()
        self.prognostic_var_names: PrognosticVarNames = (
            self.dataset_spec.prognostic_var_names
        )
        self.boundary_var_names: BoundaryVarNames = self.dataset_spec.boundary_var_names
        self.levels = self.dataset_spec.num_prognostic_depth_levels

        str_prognostics = ", ".join([i for i in self.prognostic_var_names])
        str_boundaries = ", ".join([i for i in self.boundary_var_names])

        logger.info(f"Prognostic variables: {str_prognostics}")
        logger.info(f"Boundary variables: {str_boundaries}")
        logger.info(f"Levels: {self.levels}")

        self.N_bound = len(self.boundary_var_names)
        self.N_prog = len(self.prognostic_var_names)

        self.data_container = cfg.data.build(
            data_root=cfg.experiment.resolved_data_root,
        )
        self.train_schedule: TrainSchedule = cfg.experiment.train_schedule
        model_patch_extent = getattr(cfg.model, "patch_extent", None)
        self.model_patch_extent = (
            tuple(model_patch_extent) if model_patch_extent is not None else None
        )
        if self.train_schedule == "mix" and cfg.model.pred_residuals:
            raise ValueError(
                "Residual predictions on a mixed multiscale training schedule is not currently supported."
            )
        physical_latent_autoregression = isinstance(
            cfg.model, config.SamudraMultiConfig
        ) and bool(cfg.train_processor_depths)
        if (
            self.train_schedule == "mix"
            and any(step > 1 for step in cfg.steps)
            and not physical_latent_autoregression
        ):
            raise ValueError(
                "Step predictions on a mixed multiscale training schedule require "
                "SamudraMulti physical latent autoregression."
            )

        data_num_workers = cfg.data.loading.num_pytorch_workers()
        persistent_workers = cfg.data.loading.persistent_pytorch_workers()
        self.data_loading = cfg.data.loading

        self.mp_context: BaseContext | None = None
        if data_num_workers > 0:
            self.mp_context = multiprocessing.get_context("spawn")
        self.inference_num_workers = cfg.data.inference_loading.num_workers
        self.inference_persistent_workers = (
            cfg.data.inference_loading.persistent_workers
        )
        self.inference_mp_context: BaseContext | None = None
        if self.inference_num_workers > 0:
            self.inference_mp_context = multiprocessing.get_context("spawn")

        self.num_prog_in = int((cfg.data.hist + 1) * self.N_prog)
        self.num_boundary_in = int((cfg.data.hist + 1) * self.N_bound)
        self.num_in = self.num_prog_in + self.num_boundary_in
        self.num_out = self.num_prog_in

        self.tensor_map = TensorMap(dataset_spec=self.dataset_spec).to(self.device)

        logger.info(f"Number of inputs (prognostic + boundary): {self.num_in}")
        logger.info(f"Number of outputs (prognostic): {self.num_out}")

        assert isinstance(cfg.data_stride, list)
        assert isinstance(cfg.steps, list)
        assert isinstance(cfg.step_transition, list)
        assert len(cfg.step_transition) == len(cfg.steps) - 1
        max_steps = str(cfg.steps[-1])
        self.str_video = "steps_" + max_steps + "_" + "_Lateral_Data_025_no_smooth"

        # Dataloaders
        logger.info(f"Loading data")
        if cfg.train_time.overlaps(cfg.val_time):
            raise ValueError(
                f"Training time range {cfg.train_time} overlaps "
                f"with validation time range {cfg.val_time}"
            )

        self.concurrent_compute = cfg.data.concurrent_compute

        self.primary_src = self.data_container.primary_source

        # We use dask for inference since it has memory issues otherwise.
        # TODO(jder): Could rewrite inference dataset like we did for TorchTrainDataset
        # see https://github.com/m2lines/Samudra/issues/208
        self.inference_src = self.data_container.inference_source

        self.loader_version = self.data_container.loader_version

        # This is used by both the aggregator and corrector. It only works at a single scale.
        self.normalize = Normalize(
            self.primary_src,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
        )

        self.model = cfg.model.build(
            prog_channels=self.num_prog_in,
            boundary_channels=self.num_boundary_in,
            out_channels=self.num_out,
            hist=cfg.data.hist,
            # TODO(559): This won't work at multiple scales. Refactor as part of src.
            static_data_for_corrector=self.data_container.static_data,
            srcs=self.data_container.sources,
            tensor_map=self.tensor_map,
            normalize=self.normalize,
            dataset_spec=self.dataset_spec,
        ).to(self.device)

        frozen_tensors, frozen_parameters = freeze_model_parameters(
            self.model, cfg.frozen_model_prefixes
        )
        if frozen_tensors:
            logger.info(
                "Froze %d model parameter tensors (%d scalar parameters) selected "
                "by prefixes: %s",
                frozen_tensors,
                frozen_parameters,
                ", ".join(cfg.frozen_model_prefixes),
            )

        self.nets_dir = cfg.experiment.nets_dir
        self.network = self.model.__class__.__name__

        # Loss function
        self.loss_fn: LossFnWithContext = build_loss_fn(
            cfg.loss,
            device=self.device,
            num_channels=self.N_prog,
            pad_mode=cfg.model.pad,
        )

        # Optimizer
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("At least one model parameter must remain trainable.")
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.learning_rate)
        self.ema_decay = cfg.ema_decay
        self.faster_decay_at_start = cfg.faster_decay_at_start

        # Scheduler
        self.scheduler = None
        self.scheduler_interval = None
        if cfg.scheduler:
            self.scheduler = cfg.scheduler.build(self.optimizer, cfg.epochs)
            self.scheduler_interval = getattr(cfg.scheduler, "interval", "epoch")

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(
            cfg.experiment.wandb.mode == "online", is_main_process()
        )

        self.ckpt_paths = CheckpointPaths(self.nets_dir)

        # Check for preemption
        if configure_preemptible_resume(cfg, self.ckpt_paths.latest_checkpoint_path):
            logger.info(
                "Preempted run detected; resuming full training state from %s.",
                self.ckpt_paths.latest_checkpoint_path,
            )

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            cfg.resume_ckpt_path,
            cfg,
            data_container=self.data_container,
            finetune=cfg.finetune,
        )

        # Log effective batch size
        world_size = get_world_size()
        effective_batch_size = (
            cfg.batch_size * cfg.gradient_accumulation_steps * world_size
        )
        logger.info(
            f"Effective global batch size: {effective_batch_size} "
            f"(batch_size={cfg.batch_size} × "
            f"gradient_accumulation_steps={cfg.gradient_accumulation_steps} × "
            f"world_size={world_size})"
        )
        if self.is_wandb_enabled():
            self.wandb_logger.log(
                {
                    "config/effective_batch_size": effective_batch_size,
                    "config/world_size": world_size,
                    "config/micro_batch_size_per_rank": cfg.batch_size,
                },
                step=0,
            )

        self.num_batches_seen = 0
        self.num_optimizer_updates = 0
        self.num_samples_seen = 0
        loaded_checkpoint = False
        if cfg.resume_ckpt_path is not None:
            if cfg.finetune:
                self.load_checkpoint(
                    cfg.resume_ckpt_path,
                    finetune=True,
                    allowed_missing_prefixes=cfg.finetune_allowed_missing_prefixes,
                )
                self.start_epoch = 1
            else:
                self.load_checkpoint(cfg.resume_ckpt_path)
                if not self.wandb_logger.enabled and is_main_process():
                    warnings.warn(
                        "This checkpoint had wandb enabled, \
                            but wandb is not enabled now!"
                    )
            loaded_checkpoint = True
        else:
            self.start_epoch = 1

        # Modify DDP setup based on device
        if self.distributed is not None:
            self.model = nn.parallel.DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(self.model),
                device_ids=[self.distributed.gpu],
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
        self.save_freq = cfg.save_freq
        self.validation_image_log_freq = cfg.validation_image_log_freq
        self.output_dir = cfg.experiment.output_dir
        self.debug = cfg.debug
        self.data_stride: list[int] = cfg.data_stride
        self.batch_size: int = cfg.batch_size
        self.gradient_accumulation_steps: int = cfg.gradient_accumulation_steps
        self.train_processor_depths = cfg.train_processor_depths
        self.validation_processor_depths = cfg.validation_processor_depths
        self.validation_boundary_ablations = cfg.validation_boundary_ablations
        self.validation_only = cfg.validation_only
        if self.train_processor_depths is not None:
            unwrapped_model = getattr(self.model, "module", self.model)
            if not isinstance(unwrapped_model, SamudraMulti):
                raise TypeError("train_processor_depths requires a SamudraMulti model.")
            if cfg.target_time_mode != "forecast":
                raise ValueError(
                    "train_processor_depths requires physical forecast targets."
                )
            if unwrapped_model.pred_residuals:
                raise ValueError(
                    "Latent physical-time training requires absolute decoder outputs."
                )
            maximum_depth = max(self.train_processor_depths)
            if any(step < maximum_depth for step in self.steps):
                raise ValueError(
                    "Every configured training rollout must contain the maximum "
                    f"processor depth {maximum_depth}; got steps={self.steps}."
                )
            if unwrapped_model.boundary_encoder is None:
                raise ValueError(
                    "train_processor_depths requires a separate boundary encoder "
                    "so every physical step receives aligned forcing."
                )
        self.num_workers: int = data_num_workers
        self.persistent_workers: bool = persistent_workers
        self.pin_mem: bool = cfg.pin_mem
        self.train_time: config.TimeConfig = cfg.train_time
        self.val_time = cfg.val_time
        self.train_sample_selection = cfg.train_sample_selection
        self.target_time_mode = cfg.target_time_mode
        self.inference_times = cfg.inference_times
        self.inference_epochs = cfg.inference_epochs
        self.max_train_model_steps_forward = MAX_TRAIN_MODEL_STEPS_FORWARD // (
            self.hist + 1
        )
        self.normalize_before_mask: bool = cfg.data.normalize_before_mask
        self.normalize_fill_value: float = cfg.data.masked_fill_value
        self.delayed_loss_estimate: bool = cfg.delayed_loss_estimate
        self.wandb_watch = cfg.wandb_watch

        self.profiler = cfg.profiler.build(self.output_dir, self.device)
        self.validation_images_enabled = self._sync_flag_from_main(
            self.wandb_logger.enabled
        )

        assert self.tensor_map is not None

        if self.inference_epochs:
            self.init_inference_stores()

        # Add type annotations for samplers
        self.train_sampler: (
            EquivalenceGroupBatchSampler | DistributedEquivalenceGroupBatchSampler
        )
        self.val_sampler: (
            EquivalenceGroupBatchSampler | DistributedEquivalenceGroupBatchSampler
        )
        self.inference_sampler: DistributedSampler | RandomSampler

        # Add type annotations for loaders
        self.train_loader: TrainBatchLoader
        self.val_loader: TrainBatchLoader
        self.inference_loader: DataLoader[TrainData]

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
            sliced_src = self.inference_src.slice_time(self.inference_times[i])
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
            num_workers=self.inference_num_workers,
            persistent_workers=(
                self.inference_persistent_workers and self.inference_num_workers > 0
            ),
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_inference_data,
            multiprocessing_context=self.inference_mp_context,
        )

    def run(self) -> None:
        """Run training and deterministically release loader-owned resources."""
        try:
            self._run()
        finally:
            if hasattr(self, "train_loader"):
                self.train_loader.close()
                self.val_loader.close()
            if hasattr(self, "inference_loader"):
                close_pytorch_dataloader(self.inference_loader)

    def _run(self) -> None:
        if self.validation_only:
            self._run_validation_only()
            return

        logger.info(f"Starting training")

        self.best_val_loss = 1e8
        self.best_inf_loss = 1e8
        if self.wandb_watch is not None:
            self.wandb_logger.watch(self.model, log=self.wandb_watch)

        self.profiler.start()

        start_time = time.perf_counter()
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Iterative step training
            if epoch == self.start_epoch or epoch in self.step_transition:
                cur_step = self.get_current_step(epoch)
                self.init_data_loaders(cur_step)

            self.train_loader.set_epoch(epoch)
            self.val_loader.set_epoch(epoch)

            start_epoch_train_time = time.perf_counter()
            train_stats = self.train_one_epoch(epoch)
            end_epoch_train_time = time.perf_counter()
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
            inf_loss = inf_stats.get("inference/time_mean_norm/rmse/channel_mean", None)

            logger.info(f"Achieved Train Loss = {train_loss:.3f}")
            logger.info(f"Achieved Validation Loss = {v_loss:.3f}")
            if inf_loss is not None:
                logger.info(f"Achieved Inference Loss = {inf_loss:.3f}")

            if is_main_process():
                self.save_all_checkpoints(epoch, v_loss, inf_loss)

            time_elapsed = time.perf_counter() - start_epoch_train_time

            log_stats = {
                **train_stats,
                **val_stats,
                **inf_stats,
                "epoch": epoch,
                "progress/microbatches": self.num_batches_seen,
                "progress/optimizer_updates": self.num_optimizer_updates,
                "progress/samples": self.num_samples_seen,
                "epoch_train_seconds": end_epoch_train_time - start_epoch_train_time,
                "epoch_validation_seconds": end_epoch_val_time - end_epoch_train_time,
                "epoch_total_seconds": time_elapsed,
            }

            if end_epoch_inf_time is not None:
                log_stats["epoch_inference_seconds"] = (
                    end_epoch_inf_time - end_epoch_val_time
                )

            if is_main_process():
                self.wandb_logger.log(log_stats, step=self.num_batches_seen)

        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")
        self.finish()

    def _run_validation_only(self) -> None:
        """Evaluate one explicit checkpoint without training or checkpoint writes."""
        epoch = 1
        logger.info("Starting checkpoint validation-only audit")
        self.init_data_loaders(self.get_current_step(epoch))
        self.val_loader.set_epoch(epoch)

        start_time = time.perf_counter()
        val_stats = self.validate_one_epoch(epoch)
        elapsed = time.perf_counter() - start_time
        log_stats = {
            **val_stats,
            "epoch": epoch,
            "validation_only": 1.0,
            "epoch_validation_seconds": elapsed,
            "progress/microbatches": self.num_batches_seen,
            "progress/optimizer_updates": self.num_optimizer_updates,
            "progress/samples": self.num_samples_seen,
        }
        if is_main_process():
            self.wandb_logger.log(log_stats, step=self.num_batches_seen)

        logger.info(
            "Checkpoint validation-only audit completed in %s",
            str(datetime.timedelta(seconds=int(elapsed))),
        )
        self.finish()

    def train_one_epoch(self, epoch):
        self.model.train(True)
        train_aggregator = Aggregator.get_train_aggregator(self.tensor_map)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Training Epoch: [{epoch}]"

        total_batches = len(self.train_loader)

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

        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.train_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            in_final_cycle = (
                data_iter_step + 1 > final_cycle_start
            ) and remaining_batches > 0

            # Determine the actual number of microbatches in this accumulation cycle
            if in_final_cycle:
                r = remaining_batches
            else:
                r = self.gradient_accumulation_steps

            if self.num_batches_seen == 0:
                get_model_summary(self.model, data, self.debug)

            training_depth: int | None = None
            if self.train_processor_depths is not None:
                training_depth = training_processor_depth(
                    self.train_processor_depths,
                    epoch,
                    data_iter_step,
                    total_batches,
                )
                TO = train_batch(
                    self.model,
                    data,
                    self.loss_fn,
                    processor_depth=training_depth,
                )
            else:
                TO = train_batch(self.model, data, self.loss_fn)

            # Scale loss by the actual number of microbatches that will be accumulated
            scaled_loss = TO.loss / r
            scaled_loss.backward()

            train_aggregator.record_batch(TO)

            self.num_batches_seen += 1
            local_batch_size = data.get_label(0).shape[0]
            global_batch_size = local_batch_size * get_world_size()
            self.num_samples_seen += global_batch_size

            is_last = data_iter_step + 1 == total_batches
            should_step = (data_iter_step + 1) % self.gradient_accumulation_steps == 0
            # Step optimizer after accumulating enough batches or at the end
            optimizer_stepped = should_step or is_last
            if optimizer_stepped:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._ema(model=self.model)
                self.num_optimizer_updates += 1
                if (
                    self.scheduler is not None
                    and self.scheduler_interval == "optimizer_update"
                ):
                    self.scheduler.step()

            lr = (
                self.optimizer.param_groups[-1]["lr"]
                if self.scheduler is None
                else self.scheduler.get_last_lr()[0]
            )

            with torch.no_grad():
                # Reduce losses
                loss_value_reduce = all_reduce_mean(TO.loss.detach())
                loss_per_channel_reduce = all_reduce_mean(TO.loss_per_channel.detach())
                metrics = {
                    "train/batch/loss": loss_value_reduce,
                    "train/batch/lr": lr,
                    "train/batch/ema_cur_decay": self._ema.cur_decay.item(),
                    **get_channel_loss_dict(
                        label="train",
                        loss_per_channel=loss_per_channel_reduce,
                        tensor_map=self.tensor_map,
                    ),
                    **get_depth_loss_dict(
                        label="train",
                        loss_per_channel=loss_per_channel_reduce,
                        tensor_map=self.tensor_map,
                    ),
                    **get_variable_loss_dict(
                        label="train",
                        loss_per_channel=loss_per_channel_reduce,
                        tensor_map=self.tensor_map,
                    ),
                    "train/batch/data_load_time": metric_logger.meters[
                        "data_load_time"
                    ].value,
                    "train/batch/data_wait_time": metric_logger.meters[
                        "data_wait_time"
                    ].value,
                    "train/progress/microbatches": self.num_batches_seen,
                    "train/progress/optimizer_updates": self.num_optimizer_updates,
                    "train/progress/samples": self.num_samples_seen,
                    "train/batch/global_batch_size": global_batch_size,
                    "train/batch/optimizer_stepped": int(optimizer_stepped),
                    "train/resources/cpu_peak_memory_mib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    / 1024.0,
                }
                if training_depth is not None:
                    metrics["train/batch/processor_depth"] = training_depth

                if torch.cuda.is_available():
                    metrics.update(
                        {
                            "train/resources/gpu_allocated_memory_mib": torch.cuda.memory_allocated()
                            / (1024.0**2),
                            "train/resources/gpu_reserved_memory_mib": torch.cuda.memory_reserved()
                            / (1024.0**2),
                            "train/resources/gpu_peak_memory_mib": torch.cuda.max_memory_allocated()
                            / (1024.0**2),
                        }
                    )

                if loss_scale_per_channel_fn := getattr(
                    self.loss_fn, "loss_scale_per_channel", None
                ):
                    loss_scale_per_channel = loss_scale_per_channel_fn()
                    # Reshape from time-major channels to [hist, var] and
                    # average along the history dimension.
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
                                tensor_map=self.tensor_map,
                            ),
                            **get_channel_loss_dict(
                                label="train",
                                loss_per_channel=unscaled_loss_per_channel,
                                tensor_map=self.tensor_map,
                                loss_name="loss_unscaled",
                            ),
                            "train/batch/loss_unscaled": unscaled_loss,
                        }
                    )

            if (it_time := metric_logger.meters["iter_time"]).count > 0:
                metrics["train/batch/iter_time"] = it_time.value
                metrics["train/throughput/samples_per_second"] = (
                    global_batch_size / it_time.value
                )

            self.wandb_logger.log(metrics, step=self.num_batches_seen)

            metric_logger.update(loss=loss_value_reduce.item())
            metric_logger.update(lr=lr)

            self._maybe_update_loss(TO, data)

            self.profiler.after_batch(self.num_batches_seen)

        if self.scheduler is not None and self.scheduler_interval == "epoch":
            self.scheduler.step()

        logger.info(f"Aggregating train logs")
        return train_aggregator.get_logs()

    def _maybe_update_loss(self, output: TrainBatchOutput, data: TrainData):
        if (update := getattr(self.loss_fn, "update", None)) is None:
            return

        if self.delayed_loss_estimate:
            # Use the already-computed per-channel loss from the training
            # rollout to update DynamicLoss scales, avoiding a second forward
            # pass.  This introduces a delayed estimate but is more efficient.
            loss_per_channel = output.loss_per_channel
            # Undo the dynamic scaling to recover the raw per-channel loss.
            if get_scales := getattr(self.loss_fn, "loss_scale_per_channel", None):
                per_channel_scale = get_scales()
                raw_loss = (
                    loss_per_channel.detach().reshape(-1, per_channel_scale.shape[0])
                    / per_channel_scale
                ).reshape(-1)
            else:
                raise RuntimeError(
                    "no `loss_scale_per_channel` — cannot recover unscaled per-channel loss."
                )
            update(raw_loss)
        else:
            # Run a fresh single-step forward pass so DynamicLoss sees an
            # up-to-date, unscaled loss signal
            with torch.no_grad():
                single_step_data = TrainData(
                    data.num_prognostic_channels, data.num_boundary_channels, data.ctx
                )
                prog_input, boundary_input, label = data[0]
                single_step_data.append(prog_input, boundary_input, label)
                pred = self.model(single_step_data)
                # Compute the raw (unscaled) per-channel loss via the inner
                # loss function, bypassing DynamicLoss scaling.
                if not isinstance(self.loss_fn, DynamicLoss):
                    raise TypeError(f"Expected loss_fn to be DynamicLoss")
                raw_loss = self.loss_fn.loss_fn(pred[0], label, ctx=data.ctx)
            update(raw_loss)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        log_validation_images = (
            should_log_validation_images(epoch, self.validation_image_log_freq)
            and self.validation_images_enabled
        )

        val_aggregator: ValidateAggregator | MultiScaleValidateAggregator
        if self.train_schedule == "standard":
            # The standard val aggregator only supports a single scale.
            val_aggregator = Aggregator.get_validation_aggregator(
                self.primary_src.metadata,
                self.hist,
                self.primary_src.spherical_area_weights.to(self.device),
                self.num_out,
                self.tensor_map,
                self.normalize,
                include_image_aggregators=log_validation_images,
                patch_size=(
                    (
                        round(
                            self.model_patch_extent[0]
                            * self.primary_src.grid_size[0]
                            / 180.0
                        ),
                        round(
                            self.model_patch_extent[1]
                            * self.primary_src.grid_size[1]
                            / 360.0
                        ),
                    )
                    if self.model_patch_extent is not None
                    else None
                ),
            )
        else:
            val_aggregator = Aggregator.get_multiscale_validation_aggregator(
                self.data_container.sources,
                self.hist,
                self.num_out,
                self.tensor_map,
                self.prognostic_var_names,
                self.boundary_var_names,
                include_image_aggregators=log_validation_images,
                patch_extent=self.model_patch_extent,
            )
        metric_logger = MetricLogger(delimiter="  ")
        header = f"One-Step Validation Epoch: [{epoch}]"
        lead_aggregators = {
            depth: RouteTrainAggregator(self.tensor_map)
            for depth in self.validation_processor_depths or []
        }
        persistence_lead_aggregators = {
            depth: RouteTrainAggregator(self.tensor_map)
            for depth in self.validation_processor_depths or []
        }
        zero_depth_aggregator = (
            RouteTrainAggregator(self.tensor_map)
            if self.validation_processor_depths is not None
            else None
        )
        boundary_ablation_aggregators = {
            mode: {
                depth: RouteTrainAggregator(self.tensor_map)
                for depth in self.validation_processor_depths or []
            }
            for mode in self.validation_boundary_ablations
        }

        with torch.no_grad(), self._test_context():
            for data_iter_step, data in enumerate(
                metric_logger.log_every(self.val_loader, 1, header)
            ):
                if self.debug and (data_iter_step + 1) % 5 == 0:
                    break

                if self.validation_processor_depths is None:
                    VO = validate_batch(self.model, data, self.loss_fn)
                else:
                    unwrapped_model = getattr(self.model, "module", self.model)
                    if not isinstance(unwrapped_model, SamudraMulti):
                        raise TypeError(
                            "validation_processor_depths requires a SamudraMulti model."
                        )
                    forecasts = unwrapped_model.latent_forecast(
                        data, self.validation_processor_depths
                    )
                    lead_batch_outputs: dict[int, TrainBatchOutput] = {}
                    for depth, prediction in forecasts.items():
                        loss_per_channel = self.loss_fn(
                            prediction, data.get_label(depth - 1), data.ctx
                        )
                        lead_batch_outputs[depth] = TrainBatchOutput(
                            torch.mean(loss_per_channel), loss_per_channel
                        )
                        lead_aggregators[depth].record_batch(
                            lead_batch_outputs[depth], data.ctx
                        )

                    prognostic, boundary = data.get_initial_input()
                    if zero_depth_aggregator is None:
                        raise RuntimeError(
                            "Physical-lead validation requires a zero-depth aggregator."
                        )
                    reconstruction_ctx = unwrapped_model.reconstruction_context(
                        prognostic, data.ctx
                    )
                    reconstruction = unwrapped_model.reconstruct_once(
                        prognostic, boundary, data.ctx
                    )
                    reconstruction_loss_per_channel = self.loss_fn(
                        reconstruction, prognostic, reconstruction_ctx
                    )
                    zero_depth_aggregator.record_batch(
                        TrainBatchOutput(
                            torch.mean(reconstruction_loss_per_channel),
                            reconstruction_loss_per_channel,
                        ),
                        reconstruction_ctx,
                    )
                    persistence = coordinate_bilinear_resample(
                        prognostic,
                        data.ctx.input_resolution_cpu,
                        data.ctx.output_resolution_cpu,
                        valid_mask=data.ctx.input_mask,
                    )
                    for depth, aggregator in persistence_lead_aggregators.items():
                        persistence_loss_per_channel = self.loss_fn(
                            persistence, data.get_label(depth - 1), data.ctx
                        )
                        aggregator.record_batch(
                            TrainBatchOutput(
                                torch.mean(persistence_loss_per_channel),
                                persistence_loss_per_channel,
                            ),
                            data.ctx,
                        )
                    lead_one_prediction = forecasts[1]
                    lead_one_target = data.get_label(0)
                    lead_one_output = lead_batch_outputs[1]
                    VO = ValBatchOutput(
                        lead_one_output.loss,
                        lead_one_output.loss_per_channel,
                        torch.cat((prognostic, boundary), dim=1),
                        lead_one_target,
                        lead_one_prediction,
                        data.ctx,
                    )
                    for mode, aggregators in boundary_ablation_aggregators.items():
                        ablated_data = ablate_boundary_forcing(data, mode)
                        ablated_forecasts = unwrapped_model.latent_forecast(
                            ablated_data, self.validation_processor_depths
                        )
                        for depth, prediction in ablated_forecasts.items():
                            loss_per_channel = self.loss_fn(
                                prediction, data.get_label(depth - 1), data.ctx
                            )
                            aggregators[depth].record_batch(
                                TrainBatchOutput(
                                    torch.mean(loss_per_channel), loss_per_channel
                                ),
                                data.ctx,
                            )
                val_aggregator.record_validation_batch(VO)
                metric_logger.update(loss=VO.loss)

        logger.info(f"Aggregating validation logs")
        logs = dict(val_aggregator.get_logs(label="val"))
        for depth, lead_aggregator in lead_aggregators.items():
            logs.update(lead_aggregator.get_logs(label=f"val/physical_lead_{depth}"))
        for depth, lead_aggregator in persistence_lead_aggregators.items():
            logs.update(
                lead_aggregator.get_logs(label=f"val/physical_lead_{depth}/persistence")
            )
        if zero_depth_aggregator is not None:
            logs.update(
                zero_depth_aggregator.get_logs(label="val/zero_depth_reconstruction")
            )
        for mode, aggregators in boundary_ablation_aggregators.items():
            for depth, lead_aggregator in aggregators.items():
                logs.update(
                    lead_aggregator.get_logs(
                        label=f"val/boundary_{mode}/physical_lead_{depth}"
                    )
                )
        return logs

    def inference_one_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad(), self._test_context():
            for data_iter_step, (inference_dataset, num_steps) in enumerate(
                self.inference_loader
            ):
                # TODO(alxmrs): Aggregator only supports a single scale.
                inf_aggregator = Aggregator.get_inline_inference_aggregator(
                    num_steps,
                    self.primary_src.metadata,
                    self.hist,
                    self.primary_src.spherical_area_weights.to(self.device),
                    self.primary_src.masks.prognostic.to(self.device),
                    self.num_out,
                    self.tensor_map,
                    self.normalize,
                    self.prognostic_var_names,
                )

                # TODO(jder): we need the underlying model so we can use forward_once;
                # see https://github.com/m2lines/Samudra/issues/51
                run_rollout(
                    model=self.model.module
                    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                    else self.model,
                    dataset=inference_dataset,
                    inf_aggregator=inf_aggregator,
                    epoch=epoch,
                    num_model_steps_forward=min(
                        num_steps // 2, self.max_train_model_steps_forward
                    ),
                    tensor_map=self.tensor_map,
                    normalize=self.normalize,
                )

        logger.info(f"Aggregating inference logs")
        logs = inf_aggregator.get_summary_logs()
        return {f"inference/{k}": v for k, v in logs.items()}

    def get_current_step(self, epoch):
        """Determine the current step based on the epoch and transition points.

        Args:
            epoch (int): Current epoch number

        Returns:
            tuple: (current_step, current_step_idx)
        """
        if epoch == self.start_epoch:
            # Find initial step based on start epoch
            cur_step = None
            cur_step_idx = None
            for i, epoch_to_transition in enumerate(self.step_transition):
                if epoch <= epoch_to_transition:
                    cur_step = self.steps[i]
                    cur_step_idx = i
                    break
            if cur_step is None:
                cur_step = self.steps[-1]
                cur_step_idx = len(self.steps) - 1
            logger.info(f"Starting training at step {cur_step}")
        else:
            # Transition to next step
            cur_step_idx = next(
                i for i, e in enumerate(self.step_transition) if e == epoch
            )
            cur_step_idx += 1
            cur_step = self.steps[cur_step_idx]
            logger.info(f"Transitioning to step {cur_step}")

        return cur_step

    def init_data_loaders(self, cur_step: int) -> None:
        """Initialize training and validation data loaders.

        Args:
            cur_step: Current training step size
        """
        if hasattr(self, "train_loader"):
            self.train_loader.close()
            self.val_loader.close()
        scales = self.data_container.sources
        match self.train_schedule:
            case "standard":
                srcs: Iterable[tuple[CanonicalDataset, CanonicalDataset | None]] = [
                    (scales[0], None)
                ]
            case "match":
                srcs = [(s, s) for s in scales]
            case "mix":
                srcs = list(itertools.product(scales, repeat=2))  # type: ignore
            case _:
                assert_never(self.train_schedule)

        shard_specs = [
            (stride, src, dst) for stride in self.data_stride for src, dst in srcs
        ]
        train_datasets = [
            TorchTrainDataset(
                src=src.slice_time(self.train_time),
                dst=dst.slice_time(self.train_time) if dst else None,
                prognostic_var_names=self.prognostic_var_names,
                boundary_var_names=self.boundary_var_names,
                hist=self.hist,
                steps=cur_step,
                normalize_before_mask=self.normalize_before_mask,
                masked_fill_value=self.normalize_fill_value,
                stride=stride,
                concurrent_compute_=self.concurrent_compute,
                shard_id=f"train-{shard_index}",
                sample_num=(
                    self.train_sample_selection.num_samples
                    if self.train_sample_selection is not None
                    else None
                ),
                sample_seed=(
                    self.train_sample_selection.seed
                    if self.train_sample_selection is not None
                    else 0
                ),
                target_time_mode=self.target_time_mode,
            )
            for shard_index, (stride, src, dst) in enumerate(shard_specs)
        ]

        val_datasets = [
            TorchTrainDataset(
                src=src.slice_time(self.val_time),
                dst=dst.slice_time(self.val_time) if dst else None,
                prognostic_var_names=self.prognostic_var_names,
                boundary_var_names=self.boundary_var_names,
                hist=self.hist,
                steps=(
                    max(self.validation_processor_depths)
                    if self.validation_processor_depths is not None
                    else 1
                ),
                normalize_before_mask=self.normalize_before_mask,
                masked_fill_value=self.normalize_fill_value,
                stride=stride,
                concurrent_compute_=self.concurrent_compute,
                shard_id=f"val-{shard_index}",
                target_time_mode=self.target_time_mode,
            )
            for shard_index, (stride, src, dst) in enumerate(shard_specs)
        ]

        if self.loader_version != TorchTrainDataset.FLAG:
            raise NotImplementedError(
                f"Loader version {self.loader_version} not supported."
            )

        # Create batch samplers - branch on distributed vs non-distributed
        if self.distributed is not None:
            # Distributed training
            assert self.distributed.world_size is not None
            assert self.distributed.rank is not None
            train_batch_sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=train_datasets,
                batch_size=self.batch_size,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=True,
                drop_last=True,
                seed=self.rand_seed,
            )

            val_batch_sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=val_datasets,
                batch_size=self.batch_size,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=False,
                drop_last=False,
                seed=self.rand_seed,
            )
        else:
            # Non-distributed training
            train_batch_sampler = EquivalenceGroupBatchSampler.from_datasets(  # type: ignore
                datasets=train_datasets,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                seed=self.rand_seed,
            )

            val_batch_sampler = EquivalenceGroupBatchSampler.from_datasets(  # type: ignore
                datasets=val_datasets,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                seed=self.rand_seed,
            )

        # Store samplers for set_epoch calls
        self.train_sampler = train_batch_sampler
        self.val_sampler = val_batch_sampler

        worker_seed = self.rand_seed
        if self.distributed is not None:
            assert self.distributed.rank is not None
            worker_seed += 2 * self.distributed.rank
        self.train_loader = build_train_batch_loader(
            train_datasets,
            train_batch_sampler,
            self.device,
            self.data_loading,
            pin_memory=self.pin_mem,
            multiprocessing_context=self.mp_context,
            worker_seed=worker_seed,
        )
        self.val_loader = build_train_batch_loader(
            val_datasets,
            val_batch_sampler,
            self.device,
            self.data_loading,
            pin_memory=self.pin_mem,
            multiprocessing_context=self.mp_context,
            worker_seed=worker_seed + 1,
        )

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
    ):
        if for_inference:
            with self._ema_context():
                model_state_dict = self.model.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # Create temporary file in the same directory as the target
        temp_dir = os.path.dirname(checkpoint_path)
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as tmp:
            temporary_location = tmp.name
            checkpoint = {
                "model": model_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "best_inf_loss": self.best_inf_loss,
                "ema": self._ema.get_state(include_ema_params=not for_inference),
                "num_batches_seen": self.num_batches_seen,
                "num_optimizer_updates": self.num_optimizer_updates,
                "num_samples_seen": self.num_samples_seen,
                "wandb_id": self.wandb_id,
                "wandb_name": self.wandb_name,
            }
            loss_state: dict[str, Any] | None = None
            if state_dict_fn := getattr(self.loss_fn, "state_dict", None):
                loss_state = state_dict_fn()

            if loss_state is not None:
                checkpoint["loss_fn_state"] = loss_state
            if self.scheduler:
                checkpoint["scheduler"] = self.scheduler.state_dict()

            torch.save(checkpoint, temporary_location)
            os.replace(temporary_location, checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path,
        finetune=False,
        allowed_missing_prefixes: list[str] | None = None,
    ):
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
        partial_finetune = load_model_state_for_finetune(
            self.model, new_state_dict, allowed_missing_prefixes or []
        )

        # Load EMA state
        if partial_finetune:
            self._ema = EMATracker(
                self.model,
                decay=self.ema_decay,
                faster_decay_at_start=self.faster_decay_at_start,
            )
        else:
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

            self.start_epoch = checkpoint["epoch"] + 1
            self.wandb_id = checkpoint.get("wandb_id")
            self.wandb_name = checkpoint.get("wandb_name")
            self.num_batches_seen = checkpoint.get("num_batches_seen", 0)
            self.num_optimizer_updates = checkpoint.get("num_optimizer_updates", 0)
            self.num_samples_seen = checkpoint.get("num_samples_seen", 0)

            logger.info(f"Start Epoch: {self.start_epoch}")
            logger.info(f"Wandb id: {self.wandb_id}")
            logger.info(f"Wandb name: {self.wandb_name}")
            logger.info(f"Optimizer LR: {self.optimizer.param_groups[-1]['lr']}")

            self.best_val_loss = checkpoint["best_val_loss"]
            self.best_inf_loss = checkpoint["best_inf_loss"]

    def is_wandb_enabled(self):
        return self.wandb_logger.enabled and is_main_process()

    def _sync_flag_from_main(self, flag: bool) -> bool:
        """Broadcast a boolean decision from rank 0 to every process."""
        if self.distributed is None:
            return flag

        synced = torch.tensor([int(flag)], device=self.device)
        torch.distributed.broadcast(synced, src=0)
        return bool(synced.item())

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
        self.wandb_logger.finish()


def main():
    # Load config from YAML
    cfg = TrainConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()  # we do this first so logging can use them

    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    trainer = None
    try:
        trainer = Trainer(cfg)
        trainer.run()
    except Exception:
        logger.exception("Training failed with an exception")
        raise
    finally:
        del trainer
        destroy_distributed_mode()


if __name__ == "__main__":
    main()
