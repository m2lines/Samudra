import os
import copy
import wandb
import warnings
import time
import datetime
import json
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

from constants import INPT_VARS, EXTRA_VARS, OUT_VARS, DEPTH_LEVELS, get_eval_maps
from utils.train_utils import decomposed_mse, SmoothedValue, MetricLogger, extract_wet, extract_surface_wet, set_debug_apis
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
)

import xarray as xr
import dask


class Trainer:
    def __init__(self, args) -> None:

        # Distributed mode
        init_distributed_mode(args)
        dask.config.set(scheduler="synchronous")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        # disable pytorch debugging apis
        set_debug_apis(state=False)
        
        # Set precision
        amp_dtype = torch.float32
        self.enable_amp = False
        if args.amp_mode is not None:
            if args.amp_mode == "fp16":
                amp_dtype = torch.float16
            elif args.amp_mode == "bf16":
                amp_dtype = torch.bfloat16 
            self.enable_amp = True
        self.amp_dtype = amp_dtype

        if self.amp_dtype == torch.float16: 
            self.scaler = GradScaler()

        if not args.disk_mode:
            assert args.num_workers == 0 and args.pin_mem == False
        else:
            args.num_workers = torch.cuda.device_count() * args.num_workers
            args.pin_mem = True

        # Set seeds
        set_seed(args.rand_seed)

        # Check dirs
        if not os.path.exists(args.nets_dir):
            os.makedirs(args.nets_dir, exist_ok=True)

        # Getting input, extra input and output
        self.inputs = INPT_VARS[args.exp_num_in]
        self.extra_in = EXTRA_VARS[args.exp_num_extra]
        self.outputs = OUT_VARS[args.exp_num_out]
        self.CH_3D_IDX, self.DP_3D_IDX, self.VAR_SET, self.DEPTH_SET = get_eval_maps(args.exp_num_out)
        levels = args.exp_num_in.split("_")[-1]
        if "all" in levels:
            self.levels = 19
        elif "2D" in levels:
            self.levels = 1
        else:
            self.levels = int(levels)

        self.str_in = "".join([i + "_" for i in self.inputs])
        self.str_ext = "".join([i + "_" for i in self.extra_in])
        self.str_out = "".join([i + "_" for i in self.outputs])

        print("inputs: " + self.str_in)
        print("extra inputs: " + self.str_ext)
        print("outputs: " + self.str_out)
        print("levels: " + str(self.levels))

        s_train = args.lag * args.hist
        e_train = s_train + args.N_samples * args.interval
        e_test = e_train + args.interval * args.N_val

        self.N_atm = len(self.extra_in)
        self.N_in = len(self.inputs)
        if args.lateral:
            self.N_extra = (
                self.N_atm + self.N_in
            )  # Number of atmosphere variables + Lateral boundary variables
        else:
            self.N_extra = self.N_atm  # Number of atmosphere variables
        self.N_out = len(self.outputs)

        self.num_in = int((args.hist + 1) * self.N_in + self.N_extra)
        self.num_out = int((args.hist + 1) * len(self.outputs))

        print("Number of inputs: ", self.num_in)
        print("Number of outputs: ", self.num_out)

        assert args.region == "global_3D"
        self.region = args.region

        self.str_video = (
            "steps_"
            + str(args.steps)
            + "_"
            + args.region
            + "_"
            + args.depth_mode
            + "_"
            + "N_train_"
            + str(args.N_samples)
            + "_Lateral_Data_025_no_smooth"
        )

        # Dataloaders
        print("Loading data")
        assert args.depth_mode == "surface" or args.depth_mode == "all"
        self.data_dir = args.data_dir
        self.wet_file = args.wet_file
        self.data_zarr = args.data_zarr
        self.data_means_zarr = args.data_means_zarr
        self.data_stds_zarr = args.data_stds_zarr
        self.grid_file = args.grid_file

        self.data = xr.open_zarr(os.path.join(self.data_dir, self.data_zarr))
        self.data_mean = xr.open_zarr(os.path.join(self.data_dir, self.data_means_zarr))
        self.data_std = xr.open_zarr(os.path.join(self.data_dir, self.data_stds_zarr))
        wet_zarr = xr.open_zarr(os.path.join(self.data_dir, self.wet_file))
        self.wet = extract_wet(wet_zarr, self.outputs, args.hist)
        self.surface_wet = extract_surface_wet(wet_zarr)

        train_data = data_CNN_Disk_steps(
            self.data,
            self.inputs,
            self.extra_in,
            self.outputs,
            self.wet,
            self.data_mean,
            self.data_std,
            args.N_samples,
            args.lag,
            args.interval,
            args.hist,
            args.steps,
            device="cuda",
        )

        print("Instantiating torch loaders")

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, shuffle=True
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=self.train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=True
        )

        # Model
        print("Getting model " + args.network)
        if "swin" == args.network:
            model = instantiate(
                args.swin,
                in_channels=self.num_in,
                output_channels=self.num_out,
                pretrain_img_size=[180, 360],
                wet=self.wet.cuda(non_blocking=True),
                hist=args.hist,
            )
        elif "convnextunet" == args.network or "adamunet" == args.network:
            if args.unet.ch_width[0] != self.num_in:
                print(
                    "NOTE: Changing input channels to match data {}->{}".format(
                        args.unet.ch_width[0], self.num_in
                    )
                )
                args.unet.ch_width[0] = self.num_in
            if args.unet.n_out != self.num_out:
                print(
                    "NOTE: Changing output channels to match data {}->{}".format(
                        args.unet.n_out, self.num_out
                    )
                )
                args.unet.n_out = self.num_out
            model = instantiate(
                args.unet, n_out=self.num_out, wet=self.wet.cuda(non_blocking=True), hist=args.hist
            )
        else:
            raise NotImplementedError

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters: ", params)
        # summary(model)

        model = model.to(args.device, non_blocking=True)

        # Summary
        i = [torch.zeros(1, *self.train_loader.dataset[0][0].shape).cuda(non_blocking=True)] * 2
        summary(
            model,
            input_data=[i],
            col_names=["kernel_size", "output_size", "num_params"],
            depth=10,
        )

        i = [torch.zeros(1, *self.train_loader.dataset[0][0].shape).cuda(non_blocking=True)] * 8
        summary(model, input_data=[i], col_names=[], depth=10)

        self.model = model
        self.nets_dir = args.nets_dir
        self.network = args.network
        self.device = args.device

        # Loss function
        if args.loss == "mse":
            print("Using decomposed mse loss")
            self.loss = decomposed_mse

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, fused=args.enable_fused)

        # Scheduler
        self.scheduler = None
        if args.scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, args.T
            )

        # Wandb and Loading Checkpoint
        self.wandb = args.wandb.mode == "online"
        if args.resume_ckpt_path is not None:
            self.load_checkpoint(args.resume_ckpt_path)
            if self.is_wandb_enabled():
                try:
                    wandb.init(
                        config=OmegaConf.to_container(args, resolve=True),
                        name=self.wandb_name,
                        dir=args.experiment_dir,
                        resume="must",
                        id=self.wandb_id,
                        **args.wandb,
                    )
                except:
                    wandb.init(
                        config=OmegaConf.to_container(args, resolve=True),
                        name=self.wandb_name,
                        dir=args.experiment_dir,
                        **args.wandb,
                    )
            elif is_main_process():
                warnings.warn("This checkpoint had wandb enabled, but wandb is not enabled now!")
        else:
            self.start_epoch = 1
            self.wandb_id = None
            self.wandb_name = (
                args.name + "//" + args.sub_name
                if hasattr(args, "sub_name")
                else ".LOCAL" + "//" + args.name
            )
            if self.is_wandb_enabled():
                wandb.init(
                    config=OmegaConf.to_container(args, resolve=True),
                    name=self.wandb_name,
                    dir=args.experiment_dir,
                    **args.wandb,
                )
                self.wandb_id = wandb.run.id

        # DDP Model
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        if "swin" in args.network:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.gpu], find_unused_parameters=True
            )
        elif "unet" in args.network:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.gpu]
            )
            # self.model._set_static_graph()

        # Training
        self.epochs = args.epochs
        self.hist = args.hist
        self.steps = args.steps
        self.save_freq = args.save_freq
        self.output_dir = args.output_dir
        self.network = args.network
        self.testing = args.testing
        self.N_val = args.N_val
        self.lag = args.lag
        self.interval = args.interval
        self.e_train = e_train

        self.init_validation_stores()

    def init_validation_stores(self):
        num_gpus = get_world_size()
        N = 72 // 4 * num_gpus  # 72 x 5 days ~ 1 year
        self.N_local = N // num_gpus

        grids = xr.open_dataset(os.path.join(self.data_dir, self.grid_file)).rename({"xu_ocean": "x", "yu_ocean": "y"})
        self.area = torch.from_numpy(grids["area_C"].to_numpy()).to(device="cpu", non_blocking=True)

        self.surface_wet_bool = np.array(self.surface_wet.cpu()).astype(bool)
        num_vars = len(self.VAR_SET)
        self.surface_indices = [i * self.levels for i in range(num_vars-1)] + [-1]
        surface_indices_str = [self.inputs[i] for i in self.surface_indices]

        self.val_data_set = []
        self.target_set = []
        self.surface_targets_set = []
        for i in range(num_gpus):
            val_data = data_CNN_Disk(
                self.data,
                self.inputs,
                self.extra_in,
                self.outputs,
                self.wet,
                self.data_mean,
                self.data_std,
                self.N_val,
                self.lag,
                self.interval,
                self.hist,
                self.e_train + i * self.N_local,
                long_rollout=True,
                device="cuda",
            )

            mean_in = val_data.in_mean.to_array().to_numpy().reshape(-1)
            std_in = val_data.in_std.to_array().to_numpy().reshape(-1)
            mean_out = val_data.out_mean.to_array().to_numpy().reshape(-1)
            std_out = val_data.out_std.to_array().to_numpy().reshape(-1)

            val_data.norm_vals = {
                "s_out": std_out,
                "s_in": std_in,
                "m_out": mean_out,
                "m_in": mean_in,
            }

            self.val_data_set.append(val_data)
            self.target_set.append(
                val_data[: (self.N_local) // (self.hist + 1)][1]
                .reshape((self.N_local, -1, *self.surface_wet.shape))
                .numpy()
            )

            # Surface Data
            surface_targets = data_CNN_Disk(
                self.data,
                surface_indices_str,
                self.extra_in,
                surface_indices_str,
                self.wet[0],
                self.data_mean,
                self.data_std,
                self.N_val,
                self.lag,
                self.interval,
                self.hist,
                self.e_train + i * self.N_local,
                long_rollout=False,
                device="cuda",
            )
            mean_in = surface_targets.in_mean.to_array().to_numpy().reshape(-1)
            std_in = surface_targets.in_std.to_array().to_numpy().reshape(-1)
            mean_out = surface_targets.out_mean.to_array().to_numpy().reshape(-1)
            std_out = surface_targets.out_std.to_array().to_numpy().reshape(-1)

            surface_targets.norm_vals = {
                "s_out": std_out,
                "s_in": std_in,
                "m_out": mean_out,
                "m_in": mean_in,
            }
            self.surface_targets_norm_vals = surface_targets.norm_vals
            self.surface_targets_set.append(
                surface_targets[: (self.N_local) // (self.hist + 1)][1]
                .reshape((self.N_local, -1, *self.surface_wet.shape))
                .numpy()
            )

    def run(self) -> None:
        best_loss = torch.tensor(1e8)
        if self.is_wandb_enabled():
            wandb.watch(self.model, log="all")

        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_sampler.set_epoch(epoch)

            train_stats = self.train_one_epoch(epoch)
            val_stats = self.validate()

            v_loss = val_stats["loss"]

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"eval_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
            }

            if is_main_process():
                with open(
                    Path(self.output_dir) / "log.txt", mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

                print("Achieved Validation Loss = {:5.3f}".format(v_loss))
                if v_loss < best_loss:
                    best_loss = v_loss
                    print("Saving best model at epoch {0}".format(epoch))
                    self.save_checkpoint(epoch, best=True)

                elif (epoch) % self.save_freq == 0:
                    print("Saving model at epoch {0}".format(epoch))
                    self.save_checkpoint(epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
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
            if self.testing and (data_iter_step + 1) % 5 == 0:
                break

            self.optimizer.zero_grad(set_to_none=True)
            data = [d.cuda(non_blocking=True) for d in data]
            with autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
                loss_per_channel = self.model(data, loss_fn=self.loss)
                loss = torch.mean(loss_per_channel)
            
            if self.amp_dtype == torch.float16: 
                self.scaler.scale(loss).backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            loss_value = loss.item()
            
            # Check if loss is nan
            if torch.isnan(loss):
                print("Loss is NaN")
                raise ValueError("Loss is NaN")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            if self.scheduler is not None:
                # self.scheduler.step()
                self.scheduler.step(epoch - 1 + data_iter_step / iters)
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

            if self.is_wandb_enabled():
                wandb.log(
                    {
                        "train/epoch": epoch,
                        "train/total_train_loss_per_batch": loss_value_reduce,
                        "train/lr_per_batch": lr,
                    }
                )
                # Loss per channel
                for i, var in enumerate(self.outputs):
                    wandb.log({"train/per_channel/" + var: loss_per_channel[i]})

                # Loss per depth
                for d in self.DEPTH_SET:
                    wandb.log(
                        {
                            "train/depth/depth_"
                            + str(d)
                            + "_loss": torch.mean(
                                loss_per_channel[self.DP_3D_IDX[d]]
                            ).item()
                        }
                    )

                # Loss per input variable
                for k in self.VAR_SET:
                    wandb.log(
                        {
                            "train/per_var/"
                            + k
                            + "_loss": torch.mean(
                                loss_per_channel[self.CH_3D_IDX[k]]
                            ).item()
                        }
                    )

        metric_logger.synchronize_between_processes()
        print("Averaged train stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.inference_mode()
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        rank = get_rank()
        with autocast(enabled=self.enable_amp, dtype=self.amp_dtype):
            model_pred = generate_model_rollout(
                self.N_local,
                self.val_data_set[rank],
                self.model.module,
                self.hist,
                self.N_out,
                self.N_extra,
                initial_input=None, 
                Nb=0, 
                region=self.region, 
                train=True
            )

            predictions = model_pred.transpose(0, 3, 1, 2)
            targets = self.target_set[rank]
            targets_transposed = targets.transpose(0, 2, 3, 1)

            predictions = torch.from_numpy(predictions)
            targets = torch.from_numpy(targets)

            full_mse = nn.functional.mse_loss(predictions, targets, reduction="none")
            loss_per_channel = torch.mean(full_mse, dim=(0, 2, 3))
            loss_value = torch.mean(loss_per_channel)

        model_pred_unnormalized = (
            model_pred * self.val_data_set[rank].norm_vals["s_out"]
            + self.val_data_set[rank].norm_vals["m_out"]
        )
        targets_unnormalized = (
            targets_transposed * self.val_data_set[rank].norm_vals["s_out"] 
            + self.val_data_set[rank].norm_vals["m_out"]
        )
        # Surface level evaluation
        surface_preds = model_pred_unnormalized[:, :, :, self.surface_indices]
        if self.VAR_SET == set(["uo", "vo", "thetao", "so", "zos"]): # TODO: Need surface eval func fixes. Hardcoded indices.
            (
                KE_corr,
                KE_rmse,
                temp_corr,
                temp_rmse,
                saline_corr,
                saline_rmse,
                zos_corr,
                zos_rmse,
                u_corr,
                u_rmse,
                v_corr,
                v_rmse,
            ) = get_corr_rmse(
                self.surface_targets_set[rank],
                self.surface_targets_norm_vals,
                surface_preds,
                self.area,
                self.surface_wet_bool,
                0,
                self.N_local,
            )
        else:
            KE_corr = KE_rmse = temp_corr = temp_rmse = saline_corr = saline_rmse = zos_corr = zos_rmse = u_corr = u_rmse = v_corr = v_rmse = 0

        all_reduce_mean(loss_value)

        if self.is_wandb_enabled():
            wandb.log({"eval/total_eval_loss_per_batch": loss_value})
            # Loss per channel
            for i, var in enumerate(self.outputs):
                wandb.log({"eval/per_channel/" + var: loss_per_channel[i].item()})

            # Loss per depth
            for d in self.DEPTH_SET:
                wandb.log(
                    {
                        "eval/depth/depth_"
                        + str(d)
                        + "_loss": torch.mean(
                            loss_per_channel[self.DP_3D_IDX[d]]
                        ).item()
                    }
                )

            # Loss per input variable
            for k in self.VAR_SET:
                if k == "zos":
                    continue
                wandb.log(
                    {
                        "eval/per_var/"
                        + k
                        + "_loss": torch.mean(
                            loss_per_channel[self.CH_3D_IDX[k]]
                        ).item()
                    }
                )
                
            # Plot prediction and target
            for i, var in enumerate(self.outputs):
                fig = plt.figure(figsize=(10, 5))
                plt.plot(
                    range(targets.shape[0]),
                    targets_unnormalized[:, :, :, i].mean(axis=(1, 2)),
                    label="Target",
                )
                min, max = plt.ylim()
                plt.plot(
                    range(predictions.shape[0]),
                    model_pred_unnormalized[:, :, :, i].mean(axis=(1, 2)),
                    label="Prediction",
                )
                if 'thetao' in var:
                    plt.ylim(min - 0.25, max + 0.25)
                elif 'so' in var:
                    plt.ylim(min - 0.2, max + 0.2)
                elif 'KE' in var:
                    plt.ylim(min - 0.5, max + 0.5)
                plt.title(var)
                plt.legend()
                wandb.log({f"eval/plots/{var}": wandb.Image(fig)})
                plt.close()

        if self.is_wandb_enabled():
            wandb.log(
                {
                    "eval/surface/KE_corr": KE_corr,
                    "eval/surface/KE_rmse": KE_rmse,
                    "eval/surface/temp_corr": temp_corr,
                    "eval/surface/temp_rmse": temp_rmse,
                    "eval/surface/saline_corr": saline_corr,
                    "eval/surface/saline_rmse": saline_rmse,
                    "eval/surface/zos_corr": zos_corr,
                    "eval/surface/zos_rmse": zos_rmse,
                    "eval/surface/u_corr": u_corr,
                    "eval/surface/u_rmse": u_rmse,
                    "eval/surface/v_corr": v_corr,
                    "eval/surface/v_rmse": v_rmse,
                }
            )
        return {"loss": loss_value.item()}

    def save_checkpoint(self, epoch, best=False):
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "wandb_id": self.wandb_id,
            "wandb_name": self.wandb_name,
        }
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        torch.save(
            checkpoint,
            Path(self.nets_dir)
            / "{0}_epoch_{1}_{2}.pt".format(
                self.network, epoch, ("best" if best else "") + self.str_video
            ),
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.wandb_id = checkpoint["wandb_id"]
        self.wandb_name = checkpoint["wandb_name"]

        print("Loaded checkpoint from", checkpoint_path)
        print("Start Epoch:", self.start_epoch)
        print("Wandb id:", self.wandb_id)
        print("Wandb name:", self.wandb_name)
        print("Optimizer LR:", self.optimizer.param_groups[-1]["lr"])

    def is_wandb_enabled(self):
        return self.wandb and is_main_process()

    def finish(self):
        if self.is_wandb_enabled():
            wandb.finish()


def main(args):
    trainer = Trainer(args)
    trainer.run()
