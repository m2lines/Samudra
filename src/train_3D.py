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

import dask
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import ConcatDataset

from aggregator import Aggregator, LossAggregator
from config import Config
from constants import EXTRA_VARS, INPT_VARS, OUT_VARS, TensorMap, construct_metadata
from datasets import data_CNN_Disk, data_CNN_Disk_steps
from models.base import get_model_summary
from models.rollout import generate_model_rollout
from models.unet import UNet
from stepper import Stepper, TrainOutput, ValOutput
from utils.data import Normalize, extract_wet_mask, get_time_slice
from utils.device import get_device, using_gpu
from utils.distributed import (
    all_reduce_mean,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    set_seed,
)
from utils.logging import MetricLogger, SmoothedValue, handle_logging, handle_warnings
from utils.loss import (
    decomposed_mse,
    decomposed_mse_cos_weighted,
    decomposed_mse_diff_weighted,
    decomposed_mse_mae,
    decomposed_mse_scaled,
)
from utils.wandb import WandBLogger


class Trainer:
    def __init__(self, cfg) -> None:
        if not using_gpu():
            logging.info("No GPU available, using CPU")
            cfg.training.distributed = False

        self.device = get_device()

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.training.num_workers = 0  # Disable multi-processing on CPU
            cfg.training.pin_mem = False
        elif cfg.training.disk_mode:
            cfg.training.num_workers = (
                torch.cuda.device_count() * cfg.training.num_workers
            )
            cfg.training.pin_mem = True

        # Distributed mode
        init_distributed_mode(cfg.training)
        dask.config.set(scheduler="synchronous")

        # Set seeds
        set_seed(cfg.rand_seed)

        # Getting input, extra input and output
        self.inputs = INPT_VARS[cfg.training.exp_num_in]
        self.extra_in = EXTRA_VARS[cfg.training.exp_num_extra]
        self.outputs = OUT_VARS[cfg.training.exp_num_out]

        levels = cfg.training.exp_num_in.split("_")[-1]
        if "all" in levels:
            self.levels = 19
        elif "2D" in levels:
            self.levels = 1
        else:
            self.levels = int(levels)

        self.str_in = "".join([i + "_" for i in self.inputs])
        self.str_ext = "".join([i + "_" for i in self.extra_in])
        self.str_out = "".join([i + "_" for i in self.outputs])

        logging.info(f"inputs: {self.str_in}")
        logging.info(f"extra inputs: {self.str_ext}")
        logging.info(f"outputs: {self.str_out}")
        logging.info(f"levels: {self.levels}")

        self.N_atm = len(self.extra_in)
        self.N_in = len(self.inputs)
        self.N_extra = self.N_atm  # Number of atmosphere variables
        self.N_out = len(self.outputs)

        self.num_in = int((cfg.data.hist + 1) * self.N_in + self.N_extra)
        self.num_out = int((cfg.data.hist + 1) * len(self.outputs))

        self.tensor_map = TensorMap.init_instance(cfg.training.exp_num_out)

        logging.info(f"Number of inputs: {self.num_in}")
        logging.info(f"Number of outputs: {self.num_out}")

        assert isinstance(cfg.data.data_stride, list)
        assert isinstance(cfg.data.steps, list)
        assert isinstance(cfg.data.step_transition, list)
        assert len(cfg.data.step_transition) == len(cfg.data.steps) - 1
        max_steps = str(cfg.data.steps[-1])
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
        self.data_dir = cfg.data_dir
        self.wet_file = cfg.data.wet_file
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
            self.data = xr.open_dataset(os.path.join(self.data_dir, self.data_path))
        self.data_mean = xr.open_dataset(
            os.path.join(self.data_dir, self.data_means_path)
        )
        self.data_std = xr.open_dataset(
            os.path.join(self.data_dir, self.data_stds_path)
        )

        self.metadata = construct_metadata(self.data)
        wet_zarr = xr.open_zarr(os.path.join(self.data_dir, self.wet_file))
        self.wet = extract_wet_mask(wet_zarr, self.outputs, cfg.data.hist)
        areacello = wet_zarr.areacello.to_numpy()
        self.area_weights = areacello / areacello.sum()
        self.area_weights = torch.from_numpy(self.area_weights).to(self.device)

        # Model
        logging.info(f"Getting model {cfg.training.network}")
        if "convnextunet" == cfg.training.network or "adamunet" == cfg.training.network:
            if cfg.unet.ch_width[0] != self.num_in:
                logging.info(
                    f"NOTE: Changing input channels to match data"
                    f"{cfg.unet.ch_width[0]}->{self.num_in}"
                )
                cfg.unet.ch_width[0] = self.num_in
            if cfg.unet.n_out != self.num_out:
                logging.info(
                    f"NOTE: Changing output channels to match data"
                    f"{cfg.unet.n_out}->{self.num_out}"
                )
                cfg.unet.n_out = self.num_out
            model = UNet(cfg.unet, wet=self.wet.to(self.device)).to(self.device)
        else:
            raise NotImplementedError

        get_model_summary(model, self.num_in)

        self.model = model
        self.nets_dir = cfg.nets_dir
        self.network = cfg.training.network

        # Loss function
        if cfg.training.loss == "mse":
            logging.info("Using decomposed mse loss")
            self.loss_fn = decomposed_mse
        elif cfg.training.loss == "mse_diff_weighted":
            assert cfg.data.hist == 1  # TEMP
            logging.info("Using decomposed mse loss with weighted diff")
            self.loss_fn = decomposed_mse_diff_weighted
        elif cfg.training.loss == "mse_cos_weighted":
            logging.info("Using decomposed mse loss with weighted cos")
            area_weights = np.sqrt(np.cos(np.deg2rad(self.data.y))).to_numpy()
            area_weights = torch.from_numpy(area_weights).to(device=self.device)
            self.loss_fn = partial(decomposed_mse_cos_weighted, cos=area_weights)
        elif cfg.training.loss == "mse_residual_scaled":
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
        elif cfg.training.loss == "mse_mae":
            logging.info("Using decomposed mse loss with mae")
            self.loss_fn = decomposed_mse_mae
        else:
            raise NotImplementedError

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.training.learning_rate
        )

        # Scheduler
        self.scheduler = None
        if cfg.training.scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.training.epochs
            )

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(cfg.wandb.mode == "online", is_main_process())

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            cfg.training.resume_ckpt_path, cfg, finetune=cfg.training.finetune
        )

        if cfg.training.resume_ckpt_path is not None:
            if cfg.training.finetune:
                self.load_checkpoint(cfg.training.resume_ckpt_path, finetune=True)
                self.start_epoch = 1
            else:
                self.load_checkpoint(cfg.training.resume_ckpt_path)
                if not self.wandb_logger.enabled and is_main_process():
                    warnings.warn(
                        "This checkpoint had wandb enabled, \
                            but wandb is not enabled now!"
                    )
        else:
            self.start_epoch = 1

        # Modify DDP setup based on device
        if using_gpu():
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[cfg.training.gpu]
            )

        # Training
        self.epochs = cfg.training.epochs
        self.hist = cfg.data.hist
        self.steps = cfg.data.steps
        self.step_transition = cfg.data.step_transition
        self.save_freq = cfg.training.save_freq
        self.output_dir = cfg.output_dir
        self.network = cfg.training.network
        self.debug = cfg.debug
        self.data_stride = cfg.data.data_stride
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        self.pin_mem = cfg.training.pin_mem
        self.train_times = cfg.data.train
        self.val_times = cfg.data.val
        self.inference_times = cfg.data.inference
        self.inference_epochs = cfg.data.inference_epochs
        self.time_delta = cfg.data.time_delta
        self.num_batches_seen = 0

        assert self.tensor_map is not None
        self.loss_aggregator = LossAggregator.init_instance()
        self.normalize = Normalize.init_instance(
            self.data_mean,
            self.data_std,
            self.inputs,
            self.extra_in,
            self.outputs,
        )

        self.init_inference_stores()

    def init_inference_stores(self):
        # Determine number of processes based on device
        if using_gpu():
            num_splits = get_world_size()
            logging.info(f"Number of processes: {num_splits}, preferably use 8")
        else:
            num_splits = 1

        self.inference_data_loader_set = []
        self.inference_target_set = []
        self.num_steps_inf_set = []
        for i in range(num_splits):
            time_slice_with_initial_condition, num_steps = get_time_slice(
                self.inference_times[i],
                initial_cond=True,
                time_delta=self.time_delta,
                hist=self.hist,
            )
            time_slice_target, _ = get_time_slice(
                self.inference_times[i],
                initial_cond=False,
                time_delta=self.time_delta,
                hist=self.hist,
            )
            inference_data = self.data.sel(time=time_slice_with_initial_condition)
            inference_data_loader = data_CNN_Disk(
                inference_data,
                self.inputs,
                self.extra_in,
                self.outputs,
                self.wet,
                self.hist,
                long_rollout=True,
            )

            inference_target = (
                inference_data_loader[:][1]
                .reshape((num_steps, -1, *self.wet.shape[1:]))
                .numpy()
            )

            # Check if the inference target is correct
            if i == 0:
                inf_data = (
                    self.data[self.outputs]
                    .sel(time=time_slice_target)
                    .to_array()
                    .transpose("time", "variable", "lat", "lon")
                )
                inf_data = self.normalize.normalize_numpy_outputs(inf_data)
                inf_data = inf_data.to_numpy()[:num_steps]
                assert np.equal(inference_target, inf_data).all()

            self.inference_data_loader_set.append(inference_data_loader)
            self.inference_target_set.append(inference_target)
            self.num_steps_inf_set.append(num_steps)

    def run(self) -> None:
        self.best_val_loss = torch.tensor(1e8)
        self.best_inf_loss = torch.tensor(1e8)
        self.wandb_logger.watch(self.model, log="all")
        cur_step_idx = None

        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Iterative step training
            if epoch == self.start_epoch or epoch in self.step_transition:
                cur_step, cur_step_idx = self.get_current_step(epoch)
                self.init_data_loaders(cur_step)

            if using_gpu():
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)

            start_epoch_train_time = time.time()
            train_stats = self.train_one_epoch(epoch)
            end_epoch_train_time = time.time()
            val_stats = self.validate_one_epoch(epoch)
            end_epoch_val_time = time.time()

            if -1 in self.inference_epochs or epoch in self.inference_epochs:
                inf_stats = self.inference_one_epoch()
                end_epoch_inf_time = time.time()
            else:
                inf_stats = {}
                end_epoch_inf_time = None

            train_loss = train_stats["train/mean/loss"]
            v_loss = val_stats["val/mean/loss"]
            inf_loss = inf_stats.get("inference/mean/loss", None)

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

        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.train_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            self.optimizer.zero_grad()
            data = [d.to(self.device) for d in data]
            TO: TrainOutput = Stepper.train_step(self.model, data, self.loss_fn)
            TO.loss.backward()
            train_aggregator.record_batch(TO)

            self.num_batches_seen += 1

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            if self.scheduler is not None:
                # self.scheduler.step()
                # self.scheduler.step(epoch - 1 + data_iter_step / iters)
                self.scheduler.step()
            # Only synchronize if using CUDA
            if using_gpu():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

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

        return train_aggregator.get_logs()

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.model.eval()

        val_aggregator = Aggregator.get_validation_aggregator(
            self.metadata, self.hist, self.area_weights
        )
        metric_logger = MetricLogger(delimiter="  ")
        header = "One-Step Validation Epoch: [{}]".format(epoch)

        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.val_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            data = [d.to(self.device) for d in data]
            VO: ValOutput = Stepper.validate_step(self.model, data, self.loss_fn)
            val_aggregator.record_batch(VO)
            metric_logger.update(loss=VO.loss)

        return val_aggregator.get_logs(label="val")

    @torch.no_grad()
    def inference_one_epoch(self):
        self.model.eval()

        # Determine rank based on device
        if using_gpu():
            rank = get_rank()
        else:
            rank = 0

        model_pred = generate_model_rollout(
            self.num_steps_inf_set[rank],
            self.inference_data_loader_set[rank],
            self.model.module if using_gpu() else self.model,
            self.hist,
            self.N_out,
            self.N_extra,
            initial_input=None,
            train=True,
        )

        predictions = model_pred.transpose(0, 3, 1, 2)
        targets = self.inference_target_set[rank]
        # targets_transposed = targets.transpose(0, 2, 3, 1)

        predictions = torch.from_numpy(predictions)
        targets = torch.from_numpy(targets)

        full_mse = nn.functional.mse_loss(predictions, targets, reduction="none")
        loss_per_channel = torch.mean(full_mse, dim=(0, 2, 3))
        loss_value = torch.mean(loss_per_channel)

        # model_pred_unnormalized = (
        #     model_pred * self.inference_data_loader_set[rank].norm_vals["s_out"]
        #     + self.inference_data_loader_set[rank].norm_vals["m_out"]
        # )
        # targets_unnormalized = (
        #     targets_transposed *
        #  self.inference_data_loader_set[rank].norm_vals["s_out"]
        #     + self.inference_data_loader_set[rank].norm_vals["m_out"]
        # )

        loss_value = all_reduce_mean(loss_value)
        loss_per_channel = all_reduce_mean(loss_per_channel)

        return {"loss": loss_value.item()}

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

        return cur_step, cur_step_idx

    def init_data_loaders(self, cur_step):
        """Initialize training and validation data loaders.

        Args:
            cur_step (int): Current training step size
        """
        train_data = [
            data_CNN_Disk_steps(
                self.data.sel(
                    time=get_time_slice(
                        self.train_times,
                        initial_cond=False,
                        time_delta=self.time_delta,
                        hist=self.hist,
                    )[0]
                ),
                self.inputs,
                self.extra_in,
                self.outputs,
                self.wet,
                self.hist,
                cur_step,
                stride,
            )
            for stride in self.data_stride
        ]
        train_data = ConcatDataset(train_data)

        val_data = [
            data_CNN_Disk_steps(
                self.data.sel(
                    time=get_time_slice(
                        self.val_times,
                        initial_cond=False,
                        time_delta=self.time_delta,
                        hist=self.hist,
                    )[0]
                ),
                self.inputs,
                self.extra_in,
                self.outputs,
                self.wet,
                self.hist,
                1,  # current_step set to 1 for validation
                stride,
            )
            for stride in self.data_stride
        ]
        val_data = ConcatDataset(val_data)

        logging.info("Instantiating torch loaders")

        if using_gpu():
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data, shuffle=True
            )
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_data, shuffle=False
            )
        else:
            self.train_sampler = torch.utils.data.RandomSampler(train_data)
            self.val_sampler = torch.utils.data.RandomSampler(val_data)

        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=False,
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
            "model": self.model.module.state_dict()
            if using_gpu()
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
        if using_gpu():
            return self.wandb_logger.enabled and is_main_process()
        else:
            return self.wandb_logger.enabled

    def finish(self):
        self.wandb_logger.finish()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    # Load config from YAML
    cfg = Config.from_yaml(args.config)

    # Check dirs
    if not os.path.exists(cfg.nets_dir):
        os.makedirs(cfg.nets_dir, exist_ok=True)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg.save_yaml(cfg.output_dir / "config.yaml")

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
