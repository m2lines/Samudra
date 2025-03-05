# TODO:
# - resubmit jobs / preempted job safety
# - better stepper module and a cleaner model module
# - cleaner dataset modules
import argparse
import datetime
import logging
import os
import time
import traceback
import warnings
from functools import partial
from pathlib import Path
from typing import Union

import dask
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
)

import config
from aggregator import Aggregator, LossAggregator
from backend import init_train_backend
from config import TrainConfig
from constants import (
    EXTRA_VARS,
    INPT_VARS,
    OUT_VARS,
    ExtraVars,
    Grid,
    InputVars,
    OutputVars,
    TensorMap,
    construct_metadata,
)
from datasets import InferenceDataset, InferenceDatasets, TrainDataset
from models.unet import UNet
from stepper import Stepper, TrainOutput, ValOutput
from utils.data import (
    Normalize,
    extract_wet_mask,
    get_time_slice,
    spherical_area_weights,
)
from utils.device import using_gpu
from utils.distributed import all_reduce_mean, get_world_size, is_main_process, set_seed
from utils.logging import MetricLogger, SmoothedValue, handle_logging, handle_warnings
from utils.loss import (
    decomposed_mse,
    decomposed_mse_cos_weighted,
    decomposed_mse_diff_weighted,
    decomposed_mse_mae,
    decomposed_mse_scaled,
)
from utils.model import get_model_summary
from utils.train import collate_inference_data, collate_train_data
from utils.wandb import WandBLogger


class Trainer:
    model: UNet | nn.parallel.DistributedDataParallel

    def __init__(self, cfg: TrainConfig) -> None:
        # Prep directory structure -- we do this first so it's set up before
        # we try to initialize distributed training.
        cfg.experiment.nets_dir.mkdir(parents=True, exist_ok=True)
        cfg.experiment.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_yaml(str(cfg.experiment.output_dir / "config.yaml"))

        # Backend
        self.device, self.distributed = init_train_backend(cfg.backend)

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.data.num_workers = 0  # Disable multi-processing on CPU
            cfg.pin_mem = False
        elif cfg.disk_mode:
            cfg.data.num_workers = torch.cuda.device_count() * cfg.data.num_workers
            cfg.pin_mem = True

        # Distributed mode
        dask.config.set(scheduler="synchronous")

        # Set seeds
        set_seed(cfg.experiment.rand_seed)

        # Getting input, extra input and output
        self.inputs: InputVars = INPT_VARS[cfg.experiment.exp_num_in]
        self.extra_in: ExtraVars = EXTRA_VARS[cfg.experiment.exp_num_extra]
        self.outputs: OutputVars = OUT_VARS[cfg.experiment.exp_num_out]

        # TODO: The codebase currently contains code that depends on this
        assert (
            self.inputs == self.outputs
        ), "Input and output variables must be the same"

        levels = cfg.experiment.exp_num_in.split("_")[-1]
        if "all" in levels:
            self.levels = 19
        elif "2D" in levels:
            self.levels = 1
        else:
            self.levels = int(levels)

        str_in = ", ".join([i for i in self.inputs])
        str_ext = ", ".join([i for i in self.extra_in])
        str_out = ", ".join([i for i in self.outputs])

        logging.info(f"inputs: {str_in}")
        logging.info(f"extra inputs: {str_ext}")
        logging.info(f"outputs: {str_out}")
        logging.info(f"levels: {self.levels}")

        self.N_atm = len(self.extra_in)
        self.N_in = len(self.inputs)
        self.N_extra = self.N_atm  # Number of atmosphere variables
        self.N_out = len(self.outputs)

        self.num_in = int((cfg.data.hist + 1) * self.N_in + self.N_extra)
        self.num_out = int((cfg.data.hist + 1) * len(self.outputs))

        self.tensor_map = TensorMap.init_instance(cfg.experiment.exp_num_out)

        logging.info(f"Number of inputs: {self.num_in}")
        logging.info(f"Number of outputs: {self.num_out}")

        assert isinstance(cfg.data_stride, list)
        assert isinstance(cfg.steps, list)
        assert isinstance(cfg.step_transition, list)
        assert len(cfg.step_transition) == len(cfg.steps) - 1
        max_steps = str(cfg.steps[-1])
        self.str_video = (
            "steps_"
            + max_steps
            + "_"
            + cfg.data.depth_mode
            + "_Lateral_Data_025_no_smooth"
        )

        # Dataloaders
        logging.info(f"Loading data")
        assert cfg.data.depth_mode == "surface" or cfg.data.depth_mode == "all"
        self.data_dir = cfg.experiment.data_dir
        self.data_path = cfg.data.data_path
        self.data_means_path = cfg.data.data_means_path
        self.data_stds_path = cfg.data.data_stds_path
        self.scaling_residuals_file = cfg.data.scaling_residuals_file

        if "*" in self.data_path:
            self.data = xr.open_mfdataset(
                os.path.join(self.data_dir, self.data_path),
                engine="netcdf4",
                chunks={"time": 1, "lat": 180, "lon": 360},
            )
        else:
            self.data = xr.open_zarr(
                os.path.join(self.data_dir, self.data_path),
                chunks={},
            )
        self.data_mean = xr.open_dataset(
            os.path.join(self.data_dir, self.data_means_path),
            engine="netcdf4",
            chunks={},
        )
        self.data_std = xr.open_dataset(
            os.path.join(self.data_dir, self.data_stds_path),
            engine="netcdf4",
            chunks={},
        )

        self.metadata = construct_metadata(self.data)
        self.wet, self.wet_surface = extract_wet_mask(
            self.data, self.outputs, cfg.data.hist
        )
        wet_without_hist, _ = extract_wet_mask(self.data, self.outputs, 0)
        self.area_weights: Grid = spherical_area_weights(self.data)

        self.area_weights = self.area_weights.to(self.device)

        self.normalize = Normalize.init_instance(
            self.data_mean,
            self.data_std,
            self.inputs,
            self.extra_in,
            self.outputs,
            wet_without_hist,
        )

        # Model
        logging.info(f"Getting model {cfg.experiment.network}")
        if "convnextunet" == cfg.experiment.network:
            if cfg.unet.ch_width[0] != self.num_in:
                logging.info(
                    f"NOTE: Changing input channels to match data "
                    f"{cfg.unet.ch_width[0]}->{self.num_in}"
                )
                cfg.unet.ch_width[0] = self.num_in
            if cfg.unet.n_out != self.num_out:
                logging.info(
                    f"NOTE: Changing output channels to match data "
                    f"{cfg.unet.n_out}->{self.num_out}"
                )
                cfg.unet.n_out = self.num_out
            model = UNet(cfg.unet, hist=cfg.data.hist, wet=self.wet.to(self.device)).to(
                self.device
            )
        else:
            raise NotImplementedError

        get_model_summary(model, self.num_in)

        self.model = model
        self.nets_dir = cfg.experiment.nets_dir
        self.network = cfg.experiment.network

        # Loss function
        if cfg.loss == "mse":
            logging.info("Using decomposed mse loss")
            self.loss_fn = decomposed_mse
        elif cfg.loss == "mse_diff_weighted":
            assert cfg.data.hist == 1  # TEMP
            logging.info("Using decomposed mse loss with weighted diff")
            self.loss_fn = decomposed_mse_diff_weighted
        elif cfg.loss == "mse_cos_weighted":
            logging.info("Using decomposed mse loss with weighted cos")
            area_weights = np.sqrt(np.cos(np.deg2rad(self.data.y))).to_numpy()
            area_weights = torch.from_numpy(area_weights).to(device=self.device)
            self.loss_fn = partial(decomposed_mse_cos_weighted, cos=area_weights)
        elif cfg.loss == "mse_residual_scaled":
            logging.info("Using decomposed mse loss with scaled residuals")
            scaling_residuals = xr.open_zarr(
                os.path.join(self.data_dir, self.scaling_residuals_file)
            )
            scale = torch.from_numpy(
                (self.data_std[self.outputs] / scaling_residuals[self.outputs])
                .compute()
                .to_array()
                .to_numpy()
            ).to(device=self.device)
            scale = torch.concat([scale] * (cfg.data.hist + 1), dim=0)
            self.loss_fn = partial(decomposed_mse_scaled, scaling=scale)
        elif cfg.loss == "mse_mae":
            logging.info("Using decomposed mse loss with mae")
            self.loss_fn = decomposed_mse_mae
        else:
            raise NotImplementedError

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(), lr=cfg.learning_rate, fused=True
        # )

        # Scheduler
        self.scheduler = None
        if cfg.scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs
            )

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(
            cfg.experiment.wandb.mode == "online", is_main_process()
        )

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            cfg.resume_ckpt_path, cfg, finetune=cfg.finetune
        )

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
        else:
            self.start_epoch = 1

        # Modify DDP setup based on device
        if self.distributed is not None:
            self.model = nn.parallel.DistributedDataParallel(
                nn.SyncBatchNorm.convert_sync_batchnorm(self.model),
                device_ids=[self.distributed.gpu],
            )

        # Training
        self.epochs = cfg.epochs
        self.hist: int = cfg.data.hist
        self.steps = cfg.steps
        self.step_transition = cfg.step_transition
        self.save_freq = cfg.save_freq
        self.output_dir = cfg.experiment.output_dir
        self.network = cfg.experiment.network
        self.debug = cfg.debug
        self.data_stride: list[int] = cfg.data_stride
        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.data.num_workers
        self.pin_mem: bool = cfg.pin_mem
        self.train_times: config.TimeConfig = cfg.train
        self.val_times = cfg.val
        self.inference_times = cfg.inference
        self.inference_epochs = cfg.inference_epochs
        self.time_delta: int = cfg.data.time_delta
        self.num_batches_seen = 0

        assert self.tensor_map is not None
        self.loss_aggregator = LossAggregator.init_instance()

        self.init_inference_stores()

        # Add type annotations for samplers
        self.train_sampler: Union[DistributedSampler, RandomSampler]
        self.val_sampler: Union[DistributedSampler, RandomSampler]
        self.inference_sampler: Union[DistributedSampler, RandomSampler]

        # Add type annotations for loaders
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.inference_loader: DataLoader

    def init_inference_stores(self):
        # Determine number of processes based on device
        if using_gpu():
            num_splits = get_world_size()
            logging.info(f"Number of processes: {num_splits}, preferably use 8")
        else:
            num_splits = 1

        # Create datasets
        inference_datasets = []
        num_steps_inf_set = []
        for i in range(num_splits):
            time_slice_with_initial_condition, num_time_steps = get_time_slice(
                self.inference_times[i],
                time_delta=self.time_delta,
                hist=self.hist,
            )
            inference_data = self.data.sel(time=time_slice_with_initial_condition)
            inference_dataset = InferenceDataset(
                inference_data,
                self.inputs,
                self.extra_in,
                self.outputs,
                self.wet,
                self.wet_surface,
                self.hist,
                long_rollout=True,
            )

            inference_datasets.append(inference_dataset)
            num_steps_inf_set.append(num_time_steps)

        inference_data_combined: Dataset = InferenceDatasets(
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
        )

    def run(self) -> None:
        self.best_val_loss = torch.tensor(1e8)
        self.best_inf_loss = torch.tensor(1e8)
        self.wandb_logger.watch(self.model, log="all")

        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Iterative step training
            if epoch == self.start_epoch or epoch in self.step_transition:
                cur_step = self.get_current_step(epoch)
                self.init_data_loaders(cur_step)

            if isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
            if isinstance(self.val_sampler, DistributedSampler):
                self.val_sampler.set_epoch(epoch)

            start_epoch_train_time = time.time()
            train_stats = self.train_one_epoch(epoch)
            end_epoch_train_time = time.time()
            val_stats = self.validate_one_epoch(epoch)
            end_epoch_val_time = time.time()

            if -1 in self.inference_epochs or epoch in self.inference_epochs:
                inf_stats = self.inference_one_epoch(epoch)
                end_epoch_inf_time = time.time()
            else:
                inf_stats = {}
                end_epoch_inf_time = None

            train_loss = train_stats["train/mean/loss"]
            v_loss = val_stats["val/mean/loss"]
            inf_loss = inf_stats.get("inference/time_mean_norm/rmse/channel_mean", None)

            logging.info(f"Achieved Train Loss = {train_loss:.3f}")
            logging.info(f"Achieved Validation Loss = {v_loss:.3f}")
            if inf_loss is not None:
                logging.info(f"Achieved Inference Loss = {inf_loss:.3f}")

            if is_main_process():
                self.save_checkpoint(epoch, v_loss, inf_loss)

            time_elapsed = time.time() - start_epoch_train_time

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

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Training time {total_time_str}")
        self.finish()

    def train_one_epoch(self, epoch):
        self.model.train(True)
        train_aggregator = Aggregator.get_train_aggregator()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Training Epoch: [{}]".format(epoch)
        # iters = len(self.train_loader)
        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.train_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            self.optimizer.zero_grad()
            data.to(self.device)
            TO: TrainOutput = Stepper.train_step(self.model, data, self.loss_fn)
            TO.loss.backward()
            train_aggregator.record_batch(TO)

            self.num_batches_seen += 1

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

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
                    **self.loss_aggregator.get_channel_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                    **self.loss_aggregator.get_depth_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                    **self.loss_aggregator.get_variable_loss_dict(
                        label="train", loss_per_channel=loss_per_channel_reduce
                    ),
                }

            self.wandb_logger.log(metrics, step=self.num_batches_seen)

            metric_logger.update(loss=loss_value_reduce.item())
            metric_logger.update(lr=lr)

        if self.scheduler is not None:
            self.scheduler.step()

        return train_aggregator.get_logs()

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.model.eval()

        val_aggregator = Aggregator.get_validation_aggregator(
            self.metadata, self.hist, self.area_weights, self.num_out
        )
        metric_logger = MetricLogger(delimiter="  ")
        header = "One-Step Validation Epoch: [{}]".format(epoch)

        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.val_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            data.to(self.device)
            VO: ValOutput = Stepper.validate_step(self.model, data, self.loss_fn)
            val_aggregator.record_batch(VO)
            metric_logger.update(loss=VO.loss)

        return val_aggregator.get_logs(label="val")

    @torch.no_grad()
    def inference_one_epoch(self, epoch):
        self.model.eval()

        for data_iter_step, (inference_dataset, num_steps) in enumerate(
            self.inference_loader
        ):
            inf_aggregator = Aggregator.get_inline_inference_aggregator(
                num_steps,
                self.metadata,
                self.hist,
                self.area_weights,
                self.num_out,
            )

            Stepper.inference(
                # TODO(jder): we need the underlying model so we can use forward_once;
                # see https://github.com/suryadheeshjith/Ocean_Emulator/issues/51
                model=self.model.module
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                else self.model,
                dataset=inference_dataset,
                inf_aggregator=inf_aggregator,
                epoch=epoch,
                num_model_steps_forward=num_steps,
            )
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
            logging.info(f"Starting training at step {cur_step}")
        else:
            # Transition to next step
            cur_step_idx = next(
                i for i, e in enumerate(self.step_transition) if e == epoch
            )
            cur_step_idx += 1
            cur_step = self.steps[cur_step_idx]
            logging.info(f"Transitioning to step {cur_step}")

        return cur_step

    def init_data_loaders(self, cur_step: int) -> None:
        """Initialize training and validation data loaders.

        Args:
            cur_step: Current training step size
        """
        # Create datasets
        train_data: Dataset = ConcatDataset(
            [
                TrainDataset(
                    self.data.sel(
                        time=get_time_slice(
                            self.train_times,
                            time_delta=self.time_delta,
                            hist=self.hist,
                        )[0]
                    ),
                    self.inputs,
                    self.extra_in,
                    self.outputs,
                    self.wet,
                    self.wet_surface,
                    self.hist,
                    cur_step,
                    stride,
                )
                for stride in self.data_stride
            ]
        )

        val_data: Dataset = ConcatDataset(
            [
                TrainDataset(
                    self.data.sel(
                        time=get_time_slice(
                            self.val_times,
                            time_delta=self.time_delta,
                            hist=self.hist,
                        )[0]
                    ),
                    self.inputs,
                    self.extra_in,
                    self.outputs,
                    self.wet,
                    self.wet_surface,
                    self.hist,
                    1,  # current_step set to 1 for validation
                    stride,
                )
                for stride in self.data_stride
            ]
        )

        logging.info("Instantiating torch loaders")

        if self.distributed is not None:
            self.train_sampler = DistributedSampler(train_data, shuffle=True)
            self.val_sampler = DistributedSampler(val_data, shuffle=False)
        else:
            self.train_sampler = RandomSampler(train_data)
            self.val_sampler = RandomSampler(val_data)

        # Create data loaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
            collate_fn=collate_train_data,
        )

        self.val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=False,
            collate_fn=collate_train_data,
        )

    def save_checkpoint(self, epoch, v_loss, inf_loss):
        save_best_val = False
        # Check for best validation loss
        if v_loss < self.best_val_loss:
            self.best_val_loss = v_loss
            logging.info(f"New best validation loss achieved at epoch {epoch}")
            save_best_val = True  # Wait to save until we check inference loss

        # Check for best inference loss if available
        if inf_loss is not None:
            if inf_loss < self.best_inf_loss:
                self.best_inf_loss = inf_loss
                logging.info(f"New best inference loss achieved at epoch {epoch}")
                logging.info("Saving best inference model")
                self._save_checkpoint(epoch, best=True, best_type="inference")

        # Save best validation checkpoint if needed
        if save_best_val:
            logging.info("Saving best validation model")
            self._save_checkpoint(epoch, best=True, best_type="validation")

        # Regular checkpoint saving
        elif (epoch) % self.save_freq == 0:
            logging.info(f"Saving model at epoch {epoch}")
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch, best=False, best_type=None):
        checkpoint = {
            # TODO(jder): we need the underlying model so we can use forward_once;
            # see https://github.com/suryadheeshjith/Ocean_Emulator/issues/51
            "model": self.model.module.state_dict()
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "wandb_id": self.wandb_id,
            "wandb_name": self.wandb_name,
        }
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        # Determine filename suffix based on checkpoint type
        if best:
            suffix = f"best_{best_type}" if best_type else "best"
        else:
            suffix = ""

        save_path = Path(self.nets_dir) / "{0}_epoch_{1}_{2}.pt".format(
            self.network, epoch, (suffix + "_" if suffix else "") + self.str_video
        )

        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path, finetune=False):
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        if finetune:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.wandb_id = checkpoint["wandb_id"]
            self.wandb_name = checkpoint["wandb_name"]

            logging.info(f"Start Epoch: {self.start_epoch}")
            logging.info(f"Wandb id: {self.wandb_id}")
            logging.info(f"Wandb name: {self.wandb_name}")
            logging.info(f"Optimizer LR: {self.optimizer.param_groups[-1]['lr']}")

    def is_wandb_enabled(self):
        return self.wandb_logger.enabled and is_main_process()

    def finish(self):
        self.wandb_logger.finish()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--subname", type=str, required=False, help="Subname for the run", default=""
    )
    args = parser.parse_args()

    overrides = {}
    if args.subname:
        overrides["sub_name"] = args.subname

    # Load config from YAML
    cfg = TrainConfig.from_yaml(args.config, overrides)

    handle_logging(cfg)
    handle_warnings()

    trainer = Trainer(cfg)

    try:
        trainer.run()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
