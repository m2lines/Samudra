# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import datetime
import itertools
import logging
import multiprocessing
import os
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
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    DistributedSampler,
    RandomSampler,
)

from samudra import config
from samudra.aggregator import Aggregator, ValidateAggregator
from samudra.aggregator.loss import (
    get_channel_loss_dict,
    get_channel_loss_scale_dict,
    get_depth_loss_dict,
    get_variable_loss_dict,
)
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
    RawTrainData,
    TorchTrainDataset,
    TrainData,
    TrainDataLoader,
)
from samudra.models.base import BaseModel
from samudra.rust_data import RustTrainDataLoader
from samudra.stepper import (
    TrainBatchOutput,
    ValBatchOutput,
    run_rollout,
    train_batch,
    validate_batch,
)
from samudra.utils.data import DataSource, Normalize, get_inference_steps
from samudra.utils.device import using_gpu
from samudra.utils.distributed import (
    all_reduce_mean,
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
from samudra.utils.train import (
    CheckpointPaths,
    collate_inference_data,
    collate_raw_train_data,
)
from samudra.utils.wandb import WandBLogger

logger = logging.getLogger(__name__)


def should_log_validation_images(epoch: int, frequency: int) -> bool:
    """Return whether to log validation images for a 1-based training epoch."""
    if epoch < 1:
        raise ValueError(f"Epoch must be >= 1, got {epoch}")
    if frequency < 1:
        raise ValueError(
            f"Validation image log frequency must be >= 1, got {frequency}"
        )
    return (epoch - 1) % frequency == 0


class Trainer:
    """Orchestrates the full model training loop.

    Handles initialization, distributed setup, checkpointing, learning rate
    scheduling, EMA, and Weights & Biases logging.
    """

    model: BaseModel | nn.parallel.DistributedDataParallel

    def __init__(self, cfg: TrainConfig) -> None:
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
        if self.train_schedule == "mix" and cfg.model.pred_residuals:
            raise ValueError(
                "Residual predictions on a mixed multiscale training schedule is not currently supported."
            )
        if self.train_schedule == "mix" and any(step > 1 for step in cfg.steps):
            raise ValueError(
                "Step predictions on a mixed multiscale training schedule is not currently supported."
            )

        data_num_workers = cfg.data.loading.num_pytorch_workers()
        persistent_workers = cfg.data.loading.persistent_pytorch_workers()
        self.data_loading = cfg.data.loading

        self.mp_context: BaseContext | None = None
        if data_num_workers > 0:
            if self.data_container.supports_fork:
                self.mp_context = multiprocessing.get_context("fork")
            else:
                self.mp_context = multiprocessing.get_context("spawn")

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # Scheduler
        self.scheduler = None
        if cfg.scheduler:
            self.scheduler = cfg.scheduler.build(self.optimizer, cfg.epochs)

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(
            cfg.experiment.wandb.mode == "online", is_main_process()
        )

        self.ckpt_paths = CheckpointPaths(self.nets_dir)

        # Check for preemption
        if cfg.preemptible:
            assert not cfg.finetune, "Finetune is not supported with preemptible"
            preempted = os.path.isfile(self.ckpt_paths.latest_checkpoint_path)
            if preempted:
                cfg.resume_ckpt_path = str(self.ckpt_paths.latest_checkpoint_path)

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            cfg.resume_ckpt_path,
            cfg,
            data_container=self.data_container,
            finetune=cfg.finetune,
        )

        # Log effective batch size
        effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
        logger.info(
            f"Effective batch size: {effective_batch_size} "
            f"(batch_size={cfg.batch_size} × "
            f"gradient_accumulation_steps={cfg.gradient_accumulation_steps})"
        )
        if self.is_wandb_enabled():
            self.wandb_logger.log(
                {
                    "config/effective_batch_size": effective_batch_size,
                },
                step=0,
            )

        self.num_batches_seen = 0
        loaded_checkpoint = False
        if cfg.resume_ckpt_path is not None:
            if cfg.finetune:
                self.load_checkpoint(cfg.resume_ckpt_path, finetune=True)
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
        self.num_workers: int = data_num_workers
        self.persistent_workers: bool = persistent_workers
        self.pin_mem: bool = cfg.pin_mem
        self.train_time: config.TimeConfig = cfg.train_time
        self.val_time = cfg.val_time
        self.inference_times = cfg.inference_times
        self.inference_epochs = cfg.inference_epochs
        self.max_train_model_steps_forward = MAX_TRAIN_MODEL_STEPS_FORWARD // (
            self.hist + 1
        )
        self.normalize_before_mask: bool = cfg.data.normalize_before_mask
        self.normalize_fill_value: float = cfg.data.masked_fill_value
        self.delayed_loss_estimate: bool = cfg.delayed_loss_estimate

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
        self.train_loader: TrainDataLoader | RustTrainDataLoader
        self.val_loader: TrainDataLoader | RustTrainDataLoader
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
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Iterative step training
            if epoch == self.start_epoch or epoch in self.step_transition:
                cur_step = self.get_current_step(epoch)
                self.init_data_loaders(cur_step)

            if hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)
            if hasattr(self.val_sampler, "set_epoch"):
                self.val_sampler.set_epoch(epoch)

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

            TO: TrainBatchOutput = train_batch(self.model, data, self.loss_fn)

            # Scale loss by the actual number of microbatches that will be accumulated
            scaled_loss = TO.loss / r
            scaled_loss.backward()

            train_aggregator.record_batch(TO)

            self.num_batches_seen += 1

            is_last = data_iter_step + 1 == total_batches
            should_step = (data_iter_step + 1) % self.gradient_accumulation_steps == 0
            # Step optimizer after accumulating enough batches or at the end
            if should_step or is_last:
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
                }

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

            self.wandb_logger.log(metrics, step=self.num_batches_seen)

            metric_logger.update(loss=loss_value_reduce.item())
            metric_logger.update(lr=lr)

            self._maybe_update_loss(TO, data)

            self.profiler.after_batch(self.num_batches_seen)

        if self.scheduler is not None:
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
            )
        else:
            # Create a validation aggregator that handles multiple scales.
            val_aggregator = ValidateAggregator(
                {},  # Currently, don't do anything else besides record the training loss.
                self.hist,
                self.num_out,
                tensor_map=self.tensor_map,
                normalize=self.normalize,
            )
        metric_logger = MetricLogger(delimiter="  ")
        header = f"One-Step Validation Epoch: [{epoch}]"

        with torch.no_grad(), self._test_context():
            for data_iter_step, data in enumerate(
                metric_logger.log_every(self.val_loader, 1, header)
            ):
                if self.debug and (data_iter_step + 1) % 5 == 0:
                    break

                VO: ValBatchOutput = validate_batch(self.model, data, self.loss_fn)
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
        scales = self.data_container.sources
        match self.train_schedule:
            case "standard":
                srcs: Iterable[tuple[DataSource, DataSource | None]] = [
                    (scales[0], None)
                ]
            case "match":
                srcs = [(s, s) for s in scales]
            case "mix":
                srcs = list(itertools.product(scales, repeat=2))  # type: ignore
            case _:
                assert_never(self.train_schedule)

        train_datasets = [
            TorchTrainDataset(
                src=src.slice(self.train_time),
                dst=dst.slice(self.train_time) if dst else None,
                prognostic_var_names=self.prognostic_var_names,
                boundary_var_names=self.boundary_var_names,
                hist=self.hist,
                steps=cur_step,
                normalize_before_mask=self.normalize_before_mask,
                masked_fill_value=self.normalize_fill_value,
                stride=stride,
                concurrent_compute_=self.concurrent_compute,
            )
            for stride in self.data_stride
            for src, dst in srcs
        ]

        val_datasets = [
            TorchTrainDataset(
                src=src.slice(self.val_time),
                dst=dst.slice(self.val_time) if dst else None,
                prognostic_var_names=self.prognostic_var_names,
                boundary_var_names=self.boundary_var_names,
                hist=self.hist,
                steps=1,  # current_step set to 1 for validation
                normalize_before_mask=self.normalize_before_mask,
                masked_fill_value=self.normalize_fill_value,
                stride=stride,
                concurrent_compute_=self.concurrent_compute,
            )
            for stride in self.data_stride
            for src, dst in srcs
        ]

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

        match self.loader_version:
            case TorchTrainDataset.FLAG:
                collate_fn = collate_raw_train_data
            case _:
                raise NotImplementedError(
                    f"Collate function not defined for loader version "
                    f"{self.loader_version}"
                )

        # Create batch samplers - branch on distributed vs non-distributed
        # Group by input AND label resolution to handle all training schedules
        def group_key(ds):
            return tuple(prog.grid_size for prog in ds.prognostic_srcs)

        if self.distributed is not None:
            # Distributed training
            assert self.distributed.world_size is not None
            assert self.distributed.rank is not None
            train_batch_sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=train_datasets,
                group_key=group_key,
                batch_size=self.batch_size,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=True,
                drop_last=True,
            )

            val_batch_sampler = DistributedEquivalenceGroupBatchSampler(
                datasets=val_datasets,
                group_key=group_key,
                batch_size=self.batch_size,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            # Non-distributed training
            train_batch_sampler = EquivalenceGroupBatchSampler.from_datasets(  # type: ignore
                datasets=train_datasets,
                group_key=group_key,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )

            val_batch_sampler = EquivalenceGroupBatchSampler.from_datasets(  # type: ignore
                datasets=val_datasets,
                group_key=group_key,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )

        # Store samplers for set_epoch calls
        self.train_sampler = train_batch_sampler
        self.val_sampler = val_batch_sampler

        if isinstance(self.data_loading, config.RustDataLoadingConfig):
            rust_pin_memory = self.pin_mem or (
                self.device.type == "cuda" and self.data_loading.prefetch_to_device
            )
            logger.info(
                "Instantiating Rust loaders with %d prefetched batches and no "
                "PyTorch worker processes",
                self.data_loading.prefetch_batches,
            )
            self.train_loader = RustTrainDataLoader(
                train_datasets,
                train_batch_sampler,
                self.device,
                max_concurrent_reads=self.data_loading.max_concurrent_reads,
                prefetch_batches=self.data_loading.prefetch_batches,
                pin_memory=rust_pin_memory,
                prefetch_to_device=self.data_loading.prefetch_to_device,
            )
            self.val_loader = RustTrainDataLoader(
                val_datasets,
                val_batch_sampler,
                self.device,
                max_concurrent_reads=self.data_loading.max_concurrent_reads,
                prefetch_batches=self.data_loading.prefetch_batches,
                pin_memory=rust_pin_memory,
                prefetch_to_device=self.data_loading.prefetch_to_device,
            )
            return

        # Create data loaders (same for both distributed and non-distributed)
        # When using batch_sampler, don't specify batch_size or sampler
        train_dataloader = DataLoader(
            train_data,
            batch_sampler=train_batch_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_mem,
            collate_fn=collate_fn,
            multiprocessing_context=self.mp_context,
        )

        val_dataloader = DataLoader(
            val_data,
            batch_sampler=val_batch_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_mem,
            collate_fn=collate_fn,
            multiprocessing_context=self.mp_context,
        )

        # Wrap dataloaders to handle GPU post-processing
        self.train_loader = TrainDataLoader(
            train_dataloader, train_datasets, self.device
        )
        self.val_loader = TrainDataLoader(val_dataloader, val_datasets, self.device)

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

            self.start_epoch = checkpoint["epoch"] + 1
            self.wandb_id = checkpoint.get("wandb_id")
            self.wandb_name = checkpoint.get("wandb_name")
            self.num_batches_seen = checkpoint.get("num_batches_seen", 0)

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

    trainer = Trainer(cfg)

    try:
        trainer.run()
    except Exception as e:
        logger.exception("Training failed with an exception")
        raise e


if __name__ == "__main__":
    main()
