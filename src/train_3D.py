import os
import copy
import wandb
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
from torch.cuda import amp
from torchinfo import summary
from tqdm import tqdm

from constants import INPT_VARS, EXTRA_VARS, OUT_VARS, CH_3D_IDX, DP_3D_IDX
from utils.train_utils import decomposed_mse, SmoothedValue, MetricLogger
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
        cudnn.benchmark = True

        if not args.disk_mode:
            assert args.num_workers == 0 and args.pin_mem == False
        else:
            args.num_workers = torch.cuda.device_count() * 4
            args.pin_mem = True

        # Set seeds
        set_seed(args.rand_seed)

        # Wandb
        name = (
            args.name + "//" + args.sub_name
            if hasattr(args, "sub_name")
            else ".LOCAL" + "//" + args.name
        )
        wandb.init(
            config=OmegaConf.to_container(args, resolve=True),
            name=name,
            dir=args.experiment_dir,
            **args.wandb,
        )
        self.wandb = args.wandb.mode == "online"

        # Check dirs
        if not os.path.exists(args.nets_dir):
            os.makedirs(args.nets_dir, exist_ok=True)

        # Getting input, extra input and output
        self.inputs = INPT_VARS[args.exp_num_in]
        self.extra_in = EXTRA_VARS[args.exp_num_extra]
        self.outputs = OUT_VARS[args.exp_num_out]

        self.str_in = "".join([i + "_" for i in self.inputs])
        self.str_ext = "".join([i + "_" for i in self.extra_in])
        self.str_out = "".join([i + "_" for i in self.outputs])

        print("inputs: " + self.str_in)
        print("extra inputs: " + self.str_ext)
        print("outputs: " + self.str_out)

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

        print("Number of inputs: ", self.num_in)
        print("Number of outputs: ", self.N_out)

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
        self.surface_wet_file = args.surface_wet_file
        self.data_zarr = args.data_zarr
        self.data_means_zarr = args.data_means_zarr
        self.data_stds_zarr = args.data_stds_zarr
        self.grid_file = args.grid_file

        self.wet = torch.load(
            os.path.join(self.data_dir, self.wet_file)
        )
        self.data = xr.open_zarr(
            os.path.join(self.data_dir, self.data_zarr)
        )
        self.data_mean = xr.open_zarr(
            os.path.join(self.data_dir, self.data_means_zarr)
        )
        self.data_std = xr.open_zarr(
            os.path.join(self.data_dir, self.data_stds_zarr)
        )
            
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
            train_data, shuffle=True, seed=args.rand_seed
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=self.train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        # Model
        print("Getting model " + args.network)
        if "swin" == args.network:
            model = instantiate(
                args.swin,
                in_channels=self.num_in,
                output_channels=self.N_in,
                pretrain_img_size=[180, 360],
                wet=self.wet.cuda(),
                hist=args.hist,
            )
        elif "convnextunet" == args.network or "adamunet" == args.network:
            args.unet.ch_width[0] = self.num_in
            model = instantiate(args.unet, n_out=self.N_in, wet=self.wet.cuda(), hist=args.hist)
        else:
            raise NotImplementedError

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters: ", params)
        # summary(model)

        model = model.to(args.device)
        if args.preload:
            print("Loaded model from ", args.preload)
            model.load_state_dict(
                torch.load(args.preload, map_location=torch.device(args.device))
            )

        # Summary
        i = [torch.zeros(1, *self.train_loader.dataset[0][0].shape).cuda()] * 2
        summary(
            model,
            input_data=[i],
            col_names=["kernel_size", "output_size", "num_params"],
            depth=10,
        )

        i = [torch.zeros(1, *self.train_loader.dataset[0][0].shape).cuda()] * 8
        summary(model, input_data=[i], col_names=[], depth=10)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if "swin" in args.network:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
        elif "unet" in args.network:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        self.model = model
        self.nets_dir = args.nets_dir
        self.network = args.network
        self.device = args.device

        # Loss function
        if args.loss == "mse":
            print("Using decomposed mse loss")
            self.loss = decomposed_mse

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Scheduler
        self.scheduler = None
        if args.scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, args.T
            )

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
        N = 72  # 72 x 5 days ~ 1 year
        num_gpus = get_world_size()
        self.N_local = N // num_gpus

        grids = xr.open_dataset(
            os.path.join(self.data_dir, self.grid_file)
        ).rename({"dx": "dxu", "dy": "dyu"})

        self.area = torch.from_numpy(grids["area_C"].to_numpy()).to(device="cpu")

        self.surface_wet = torch.load(
            os.path.join(self.data_dir, self.surface_wet_file)
        )
        self.surface_wet_bool = np.array(self.surface_wet.cpu()).astype(bool)
        self.indices = [i * 19 for i in range(4)] + [-1]
        indices_str = [self.inputs[i] for i in self.indices]

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
            self.target_set.append(val_data[: self.N_local][1].numpy())

            # Surface Data
            surface_targets = data_CNN_Disk(
                self.data,
                indices_str,
                self.extra_in,
                indices_str,
                self.wet[0],
                self.data_mean,
                self.data_std,
                self.N_val,
                self.lag,
                self.interval,
                self.hist,
                self.e_train + i * self.N_local,
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
            self.surface_targets_set.append(surface_targets)

    def run(self) -> None:
        best_loss = torch.tensor(1e8)

        if self.wandb:
            wandb.watch(self.model, log="all")

        start_time = time.time()
        for epoch in range(self.epochs):
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

                if v_loss < best_loss:
                    print("Achieved Best Validation Loss = {:5.3f}".format(v_loss))
                    best_loss = v_loss
                    print("Saving best model at epoch {0}".format(epoch + 1))
                    torch.save(
                        self.model.module.state_dict(),
                        Path(self.nets_dir)
                        / "{0}_best_{1}.pt".format(self.network, self.str_video),
                    )

                if (epoch + 1) % self.save_freq == 0:
                    print("Saving model at epoch {0}".format(epoch + 1))
                    torch.save(
                        self.model.module.state_dict(),
                        Path(self.nets_dir)
                        / "{0}_epoch_{1}_{2}.pt".format(
                            self.network, epoch + 1, self.str_video
                        ),
                    )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

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

            self.optimizer.zero_grad()
            data = [d.cuda() for d in data]

            loss_per_channel = self.model(data, loss_fn=self.loss)
            loss = torch.mean(loss_per_channel)
            loss.backward()
            loss_value = loss.item()

            self.optimizer.step()
            if self.scheduler is not None:
                # self.scheduler.step()
                self.scheduler.step(epoch + data_iter_step / iters)
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

            if self.wandb:
                wandb.log(
                    {
                        "train/epoch": epoch,
                        "train/total_train_loss_per_batch": loss_value_reduce,
                        "train/lr_per_batch": lr,
                    }
                )
                # Loss per channel
                for i, var in enumerate(self.inputs):
                    wandb.log({"train/per_channel/" + var: loss_per_channel[i]})

                # Loss per depth
                for i in range(19):
                    wandb.log(
                        {
                            "train/depth/depth_"
                            + str(i)
                            + "_loss": torch.mean(loss_per_channel[DP_3D_IDX[i]]).item()
                        }
                    )

                # Loss per input variable
                for k in ["uo", "vo", "thetao", "so"]:
                    wandb.log(
                        {
                            "train/per_var/"
                            + k
                            + "_loss": torch.mean(loss_per_channel[CH_3D_IDX[k]]).item()
                        }
                    )

        metric_logger.synchronize_between_processes()
        print("Averaged train stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        rank = get_rank()

        model_pred = generate_model_rollout(
            self.N_local,
            self.val_data_set[rank],
            self.model.module,
            self.hist,
            self.N_in,
            self.N_extra,
            0,
            self.region,
            train=True,
        )

        predictions = model_pred.transpose(0, 3, 1, 2)
        targets = self.target_set[rank]

        predictions = torch.from_numpy(predictions)
        targets = torch.from_numpy(targets)

        full_mse = nn.functional.mse_loss(predictions, targets, reduction="none")
        loss_per_channel = torch.mean(full_mse, dim=(0, 2, 3))
        loss_value = torch.mean(loss_per_channel)

        # Surface level evaluation
        model_pred_unnormalized = (
            model_pred * self.val_data_set[rank].norm_vals["s_out"]
            + self.val_data_set[rank].norm_vals["m_out"]
        )
        surface_preds = model_pred_unnormalized[:, :, :, self.indices]

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
            surface_preds,
            self.area,
            self.surface_wet_bool,
            0,
            self.N_local,
        )

        all_reduce_mean(loss_value)

        if self.wandb:
            wandb.log({"eval/total_eval_loss_per_batch": loss_value})
            # Loss per channel
            for i, var in enumerate(self.inputs):
                wandb.log({"eval/per_channel/" + var: loss_per_channel[i].item()})

            # Loss per depth
            for i in range(19):
                wandb.log(
                    {
                        "eval/depth/depth_"
                        + str(i)
                        + "_loss": torch.mean(loss_per_channel[DP_3D_IDX[i]]).item()
                    }
                )

            # Loss per input variable
            for k in ["uo", "vo", "thetao", "so"]:
                wandb.log(
                    {
                        "eval/per_var/"
                        + k
                        + "_loss": torch.mean(loss_per_channel[CH_3D_IDX[k]]).item()
                    }
                )

        if self.wandb:
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


def main(args):
    trainer = Trainer(args)
    trainer.run()
