import argparse
import os
import copy
import wandb
import time
import datetime
import json
from pathlib import Path
from functools import partial
import logging
import traceback
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
import numpy as np
from torch.cuda import amp
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

from constants import INPT_VARS, EXTRA_VARS, OUT_VARS, DEPTH_LEVELS, get_eval_maps
from config import Config
from utils.train_utils import (
    decomposed_mse,
    decomposed_mse_diff_weighted,
    decomposed_mse_cos_weighted,
    decomposed_mse_scaled,
    decomposed_mse_mae,
    SmoothedValue,
    MetricLogger,
    extract_wet,
    extract_surface_wet,
)
from utils.dist_utils import (
    set_seed,
    init_distributed_mode,
    get_world_size,
    get_rank,
    is_main_process,
    all_reduce_mean,
)
from utils.eval_utils import generate_model_rollout, get_corr_rmse
from utils.data_utils import (
    data_CNN_Disk,
    data_CNN_Disk_steps,
    get_time_slice
)

import xarray as xr
import dask

from models.unet import UNet
from utils.wandb_utils import WandBLogger

class Trainer:
    def __init__(self, cfg) -> None:
        
        if not torch.cuda.is_available():
            cfg.training.device = "cpu"
            cfg.training.distributed = False
            logging.info("No GPU available, using CPU")
        
        self.device = torch.device(cfg.training.device)
        
        # Adjust workers and memory pinning based on device
        if self.device.type == "cpu":
            cfg.training.num_workers = 0  # Disable multi-processing on CPU
            cfg.training.pin_mem = False
        elif cfg.training.disk_mode:
            cfg.training.num_workers = torch.cuda.device_count() * cfg.training.num_workers
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
        self.CH_3D_IDX, self.DP_3D_IDX, self.VAR_SET, self.DEPTH_SET = get_eval_maps(
            cfg.training.exp_num_out
        )
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

        if '*' in self.data_path:
            self.data = xr.open_mfdataset(os.path.join(self.data_dir, self.data_path), engine="netcdf4", chunks={"time": 1, "lat": 180, "lon": 360})
        else:
            self.data = xr.open_dataset(os.path.join(self.data_dir, self.data_path))
        self.data_mean = xr.open_dataset(os.path.join(self.data_dir, self.data_means_path))
        self.data_std = xr.open_dataset(os.path.join(self.data_dir, self.data_stds_path))

        wet_zarr = xr.open_zarr(os.path.join(self.data_dir, self.wet_file))
        self.wet = extract_wet(wet_zarr, self.outputs, cfg.data.hist)
        self.area = torch.from_numpy(wet_zarr['areacello'].to_numpy()).to(device="cpu")
        self.surface_wet = extract_surface_wet(wet_zarr)

        # Model
        logging.info(f"Getting model {cfg.training.network}")
        if "convnextunet" == cfg.training.network or "adamunet" == cfg.training.network:
            if cfg.unet.ch_width[0] != self.num_in:
                logging.info(
                    f"NOTE: Changing input channels to match data {cfg.unet.ch_width[0]}->{self.num_in}"
                )
                cfg.unet.ch_width[0] = self.num_in
            if cfg.unet.n_out != self.num_out:
                logging.info(
                    f"NOTE: Changing output channels to match data {cfg.unet.n_out}->{self.num_out}"
                )
                cfg.unet.n_out = self.num_out
            model = UNet(cfg.unet, wet=self.wet.to(self.device)).to(self.device)
        else:
            raise NotImplementedError

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f"Number of parameters: {params}")
        # summary(model)

        # Model summary with proper device tensors
        # input_tensor = torch.zeros(1, self.num_in, 180, 360, device=self.device)
        # logging.info(
        #     summary(
        #         model,
        #         input_data=[[input_tensor] * 2],
        #         col_names=["kernel_size", "output_size", "num_params"],
        #         depth=10,
        #     )
        # )

        # input_tensor = torch.zeros(1, self.num_in, 180, 360, device=self.device)
        # logging.info(
        #     summary(
        #         model,
        #         input_data=[[input_tensor] * 8],
        #         col_names=[],
        #         depth=10
        #     )
        # )

        self.model = model
        self.nets_dir = cfg.nets_dir
        self.network = cfg.training.network

        # Loss function
        if cfg.training.loss == "mse":
            logging.info("Using decomposed mse loss")
            self.loss = decomposed_mse
        elif cfg.training.loss == "mse_diff_weighted":
            assert cfg.data.hist == 1  # TEMP
            logging.info("Using decomposed mse loss with weighted diff")
            self.loss = decomposed_mse_diff_weighted
        elif cfg.training.loss == "mse_cos_weighted":
            logging.info("Using decomposed mse loss with weighted cos")
            area_weights = np.sqrt(np.cos(np.deg2rad(self.data.y))).to_numpy()
            area_weights = torch.from_numpy(area_weights).to(device="cuda")
            self.loss = partial(decomposed_mse_cos_weighted, cos=area_weights)
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
            ).to(device="cuda")
            scale = torch.concat([scale] * (cfg.data.hist + 1), dim=0)
            self.loss = partial(decomposed_mse_scaled, scaling=scale)
        elif cfg.training.loss == "mse_mae":
            logging.info("Using decomposed mse loss with mae")
            self.loss = decomposed_mse_mae
        else:
            raise NotImplementedError

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.training.learning_rate)

        # Scheduler
        self.scheduler = None
        if cfg.training.scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=cfg.training.epochs
            )

        # Initialize WandB
        self.wandb_logger = WandBLogger.get_instance()
        self.wandb_logger.configure(cfg.wandb.mode == "online", is_main_process())
        
        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            cfg.training.resume_ckpt_path,
            cfg,
            finetune=cfg.training.finetune
        )

        if cfg.training.resume_ckpt_path is not None:
            if cfg.training.finetune:
                self.load_checkpoint(cfg.training.resume_ckpt_path, finetune=True)
                self.start_epoch = 1
            else:
                self.load_checkpoint(cfg.training.resume_ckpt_path)
                if not self.wandb_logger.enabled and is_main_process():
                    warnings.warn(
                        "This checkpoint had wandb enabled, but wandb is not enabled now!"
                    )
        else:
            self.start_epoch = 1

        # Modify DDP setup based on device
        if self.device.type == "cuda":
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
        
        self.init_inference_stores()

    def init_inference_stores(self):
        # Determine number of processes based on device
        if self.device.type == "cuda":
            num_splits = get_world_size()
            logging.info(f"Number of processes: {num_splits}, preferably use 8")
        else:
            num_splits = 1
        
        self.inference_data_loader_set = []
        self.inference_target_set = []
        self.num_steps_inf_set = []
        for i in range(num_splits):
            time_slice_with_initial_condition, num_steps = get_time_slice(self.inference_times[i], initial_cond=True, time_delta=self.time_delta, hist=self.hist)
            time_slice_target, _ = get_time_slice(self.inference_times[i], initial_cond=False, time_delta=self.time_delta, hist=self.hist)
            inference_data = self.data.sel(time=time_slice_with_initial_condition)
            inference_data_loader = data_CNN_Disk(
                inference_data,
                self.inputs,
                self.extra_in,
                self.outputs,
                self.wet,
                self.data_mean,
                self.data_std,
                self.hist,
                long_rollout=True,
                device="cuda",
            )

            mean_in = inference_data_loader.in_mean.to_array().to_numpy().reshape(-1)
            std_in = inference_data_loader.in_std.to_array().to_numpy().reshape(-1)
            mean_out = inference_data_loader.out_mean.to_array().to_numpy().reshape(-1)
            std_out = inference_data_loader.out_std.to_array().to_numpy().reshape(-1)

            inference_data_loader.norm_vals = {
                "s_out": std_out,
                "s_in": std_in,
                "m_out": mean_out,
                "m_in": mean_in,
            }

            inference_target = inference_data_loader[:][1].reshape((num_steps, -1, *self.surface_wet.shape)).numpy()
            
            # Check if the inference target is correct
            if i == 0:
                inf_data = self.data[self.outputs].sel(time=time_slice_target).to_array().transpose("time", "variable", "lat", "lon") 
                inf_data = (inf_data - inference_data_loader.out_mean.to_array().to_numpy().reshape([1, -1, 1, 1])) / inference_data_loader.out_std.to_array().to_numpy().reshape([1, -1, 1, 1])
                inf_data = inf_data.fillna(0)
                assert np.equal(inference_target, inf_data.to_numpy()).all()

            self.inference_data_loader_set.append(inference_data_loader)
            self.inference_target_set.append(inference_target)
            self.num_steps_inf_set.append(num_steps)

    def run(self) -> None:
        best_val_loss = torch.tensor(1e8)
        best_inf_loss = torch.tensor(1e8)
        self.wandb_logger.watch(self.model, log="all")
        cur_step_idx = None

        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Iterative step training
            if epoch == self.start_epoch or epoch in self.step_transition:
                cur_step, cur_step_idx = self.get_current_step(epoch)
                self.init_data_loaders(cur_step)

            if self.device.type == "cuda":
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
                
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.validate_one_epoch()
            v_loss = val_stats["loss"]
            
            if epoch in self.inference_epochs:
                inf_stats = self.inference_one_epoch()
                i_loss = inf_stats["loss"]
            else:
                inf_stats = {}
                i_loss = None

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"eval_{k}": v for k, v in val_stats.items()},
                **{f"inf_{k}": v for k, v in inf_stats.items()},
                "epoch": epoch,
            }

            if is_main_process():
                with open(
                    Path(self.output_dir) / "log.txt", mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

                logging.info(f"Achieved Validation Loss = {v_loss:.3f}")
                save_best_val = False
                
                # Check for best validation loss
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    logging.info(f"New best validation loss achieved at epoch {epoch}")
                    save_best_val = True  # Wait to save until we check inference loss
                
                # Check for best inference loss if available
                if i_loss is not None:
                    logging.info(f"Achieved Inference Loss = {i_loss:.3f}")
                    if i_loss < best_inf_loss:
                        best_inf_loss = i_loss
                        logging.info(f"New best inference loss achieved at epoch {epoch}")
                        logging.info("Saving best inference model")
                        self.save_checkpoint(epoch, best=True, best_type="inference")
                
                # Save best validation checkpoint if needed
                if save_best_val:
                    logging.info("Saving best validation model")
                    self.save_checkpoint(epoch, best=True, best_type="validation")

                # Regular checkpoint saving
                elif (epoch) % self.save_freq == 0:
                    logging.info(f"Saving model at epoch {epoch}")
                    self.save_checkpoint(epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Training time {total_time_str}")
        self.finish()

    def train_one_epoch(self, epoch):
        self.model.train(True)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Epoch: [{}]".format(epoch)
        iters = len(self.train_loader)

        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.train_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            self.optimizer.zero_grad()
            
            data = [d.to(self.device) for d in data]

            loss_per_channel = self.model(data, loss_fn=self.loss)
            loss = torch.mean(loss_per_channel)
            loss.backward()
            loss_value = loss.item()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            if self.scheduler is not None:
                # self.scheduler.step()
                # self.scheduler.step(epoch - 1 + data_iter_step / iters)
                self.scheduler.step()
            # Only synchronize if using CUDA
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            metric_logger.update(loss=loss_value)

            lr = (
                self.optimizer.param_groups[-1]["lr"]
                if self.scheduler is None
                else self.scheduler.get_last_lr()[0]
            )
            metric_logger.update(lr=lr)

            loss_value_reduce = all_reduce_mean(loss_value)

            self.wandb_logger.log_training_metrics(
                epoch=epoch,
                loss_value=loss_value_reduce,
                lr=lr,
                loss_per_channel=loss_per_channel,
                outputs=self.outputs,
                depth_indices=self.DP_3D_IDX,
                var_indices=self.CH_3D_IDX,
                depth_set=self.DEPTH_SET,
                var_set=self.VAR_SET
            )

        metric_logger.synchronize_between_processes()
        logging.info("Averaged train stats: " + str(metric_logger))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        
        metric_logger = MetricLogger(delimiter="  ")
        header = "Epoch: [{}]".format(epoch)
        iters = len(self.val_loader)
        
        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.val_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break
            
            data = [d.to(self.device) for d in data]
            loss_per_channel = self.model(data, loss_fn=self.loss)
            loss = torch.mean(loss_per_channel)
            loss_value = loss.item()
            
            metric_logger.update(loss=loss_value)
            
        metric_logger.synchronize_between_processes()
        logging.info("Averaged train stats: " + str(metric_logger))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    @torch.no_grad()
    def inference_one_epoch(self):
        self.model.eval()
        
        # Determine rank based on device
        if self.device.type == "cuda":
            rank = get_rank()
        else:
            rank = 0

        model_pred = generate_model_rollout(
            self.num_steps_inf_set[rank],
            self.inference_data_loader_set[rank],
            self.model.module if self.device.type == "cuda" else self.model,
            self.hist,
            self.N_out,
            self.N_extra,
            initial_input=None,
            train=True,
            device=self.device.type
        )

        predictions = model_pred.transpose(0, 3, 1, 2)
        targets = self.inference_target_set[rank]
        targets_transposed = targets.transpose(0, 2, 3, 1)

        predictions = torch.from_numpy(predictions)
        targets = torch.from_numpy(targets)

        full_mse = nn.functional.mse_loss(predictions, targets, reduction="none")
        loss_per_channel = torch.mean(full_mse, dim=(0, 2, 3))
        loss_value = torch.mean(loss_per_channel)

        model_pred_unnormalized = (
            model_pred * self.inference_data_loader_set[rank].norm_vals["s_out"]
            + self.inference_data_loader_set[rank].norm_vals["m_out"]
        )
        targets_unnormalized = (
            targets_transposed * self.inference_data_loader_set[rank].norm_vals["s_out"]
            + self.inference_data_loader_set[rank].norm_vals["m_out"]
        )
        
        loss_value = all_reduce_mean(loss_value)
        loss_per_channel = all_reduce_mean(loss_per_channel)
        
        self.wandb_logger.log_inference_metrics(
                loss_value=loss_value.item(),
                loss_per_channel=loss_per_channel,
                outputs=self.outputs,
                predictions=predictions,
                targets=targets,
                targets_unnormalized=targets_unnormalized,
                model_pred_unnormalized=model_pred_unnormalized,
                depth_indices=self.DP_3D_IDX,
                var_indices=self.CH_3D_IDX,
                depth_set=self.DEPTH_SET,
                var_set=self.VAR_SET
            )  
        

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
            cur_step_idx = next(i for i, e in enumerate(self.step_transition) if e == epoch)
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
                self.data.sel(time=get_time_slice(self.train_times, initial_cond=False, time_delta=self.time_delta, hist=self.hist)[0]),
                self.inputs,
                self.extra_in,
                self.outputs,
                self.wet,
                self.data_mean,
                self.data_std,
                self.hist,
                cur_step,
                stride,
                device="cuda",
            )
            for stride in self.data_stride
        ]
        train_data = ConcatDataset(train_data)
        
        # TODO: data_CNN_Disk currently does not support strides
        val_data = data_CNN_Disk(
                self.data.sel(time=get_time_slice(self.val_times, initial_cond=False, time_delta=self.time_delta, hist=self.hist)[0]),
                self.inputs,
                self.extra_in,
                self.outputs,
                self.wet,
                self.data_mean,
                self.data_std,
                self.hist,
                long_rollout=False,
                device="cuda",
        )

        logging.info("Instantiating torch loaders")

        if self.device.type == "cuda":
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

    def save_checkpoint(self, epoch, best=False, best_type=None):
        checkpoint = {
            "model": self.model.module.state_dict() if self.device.type == "cuda" else self.model.state_dict(),
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
        if self.device.type == "cuda":
            return self.wandb_logger.enabled and is_main_process()
        else:
            return self.wandb_logger.enabled

    def finish(self):
        self.wandb_logger.finish()

def handle_logging(cfg):
    # Set up logging
    if cfg.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(cfg.output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

    # Add separate error log file handler
    error_handler = logging.FileHandler(cfg.output_dir / 'error.log')
    error_handler.setLevel(logging.WARNING)  # Capture warnings and errors
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(error_handler)

def handle_warnings():
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.info('\n=== Warning Details ===')
        logging.info(f'Message: {message}')
        logging.info(f'Category: {category}')
        logging.info(f'File: {filename}')
        logging.info(f'Line: {lineno}')
        logging.info('\nFull stack trace:')
        stack = traceback.extract_stack()[:-1]  # Remove current frame
        for frame in stack:
            logging.info(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
            if frame.line:
                logging.info(f'    {frame.line}')
        logging.info('=====================\n')

    warnings.showwarning = warning_handler

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config from YAML
    cfg = Config.from_yaml(args.config)
    
    # Check dirs
    if not os.path.exists(cfg.nets_dir):
        os.makedirs(cfg.nets_dir, exist_ok=True)
    
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg.save_yaml(cfg.output_dir / 'config.yaml')
    
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
