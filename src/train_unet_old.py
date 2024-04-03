import os
import wandb
import time
import datetime
import json
from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate

import torch
import torch.nn as nn
import torch_geometric
import torch.backends.cudnn as cudnn
import numpy as np
from torch.cuda import amp

from constants import INPT_VARS, EXTRA_VARS, OUT_VARS
from utils.train_utils import loss_KE_pointwise, SmoothedValue, MetricLogger
from utils.dist_utils import (
    set_seed,
    init_distributed_mode,
    get_world_size,
    get_rank,
    is_main_process,
    all_reduce_mean,
)
from utils.data_utils import RecUnetDataset


class Trainer:
    def __init__(self, args) -> None:
        # Distributed mode
        init_distributed_mode(args)
        cudnn.benchmark = True

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

        self.N_atm = len(self.extra_in)  # Number of atmosphere variables
        self.N_in = len(self.inputs)
        self.N_extra = (
            self.N_atm + self.N_in
        )  # Number of atmosphere variables + Lateral boundary variables
        self.N_out = len(self.outputs)

        self.num_in = int((args.hist + 1) * self.N_in + self.N_extra)

        print("Number of inputs: ", self.num_in)  # 3 (ocean speeds + ocean temp)(t) +
        # 3 (atm wind stresses + atm temp)(t) +
        # 3 (boundary ocean speeds + boundary ocean temp)(t) -> 3 (ocean speeds + ocean temp)(t+1)
        print("Number of outputs: ", self.N_out)  # 3

        self.str_video = (
            "steps_"
            + str(args.steps)
            + "_"
            + args.region
            + "_Test_in_"
            + self.str_in
            + "ext_"
            + self.str_ext
            + "_out"
            + self.str_out
            + "N_train_"
            + str(args.N_samples)
            + "_Lateral_Data_025_no_smooth"
        )

        print("Getting model")
        # Model
        model = instantiate(args.unet)

        train_data = RecUnetDataset(
            args.train_data_path,
            args.steps,
            model.input_time_dim,
            model.presteps,
            model.output_channels,
            args.Nb,
            args.wet_path,
        )
        val_data = RecUnetDataset(
            args.val_data_path,
            args.steps,
            model.input_time_dim,
            model.presteps,
            model.output_channels,
            args.Nb,
            args.wet_path,
        )
        model.set_input_size([*train_data[0][0].shape[2:]])

        model = model.to(args.device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        self.model = model
        self.nets_dir = args.nets_dir
        self.network = args.network
        self.device = args.device

        print("Loading data")

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, shuffle=True, seed=args.rand_seed
        )
        self.train_loader = torch_geometric.loader.DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=self.train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        self.test_sampler = torch.utils.data.DistributedSampler(
            val_data, num_replicas=get_world_size(), rank=get_rank(), shuffle=False
        )
        self.test_loader = torch_geometric.loader.DataLoader(
            val_data,
            batch_size=args.batch_size,
            sampler=self.test_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters: ", params)

        # Loss function
        lam = args.lam
        mse = nn.MSELoss()
        # self.loss = (
        #     lambda out, pred: mse(out, pred) * (1 - lam)
        #     + loss_KE_pointwise(out, pred) * lam
        # )
        self.loss = lambda out, pred: mse(out, pred)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.step_weights = [
            1.0
        ] * args.steps  # Constant weighting of losses across steps

        # Scheduler
        self.scheduler = None
        if args.scheduler:
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.T)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, args.T
            )

        # Gscaler
        self.gscaler = amp.GradScaler(enabled=args.grad_clip)

        # Training
        self.epochs = args.epochs
        self.hist = args.hist
        self.steps = args.steps
        self.save_freq = args.save_freq
        self.output_dir = args.output_dir

    def run(self) -> None:
        best_loss = torch.tensor(1e8)

        if self.wandb:
            wandb.watch(self.model, log="all")

        start_time = time.time()
        for epoch in range(self.epochs):
            self.train_sampler.set_epoch(epoch)
            self.test_sampler.set_epoch(epoch)

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

            # if (data_iter_step + 1) % 5 == 0:
            #     break

            # self.optimizer.zero_grad()
            self.model.zero_grad(set_to_none=True)
            with amp.autocast(enabled=self.gscaler is not None, dtype=torch.float16):
                outs = self.model(data[0].to(device=self.device))
                loss = self.loss(data[1].to(device=self.device), outs)

            # loss.backward()
            self.gscaler.scale(loss).backward()
            self.gscaler.unscale_(self.optimizer)
            curr_lr = (
                self.optimizer.param_groups[-1]["lr"]
                if self.scheduler is None
                else self.scheduler.get_last_lr()[0]
            )
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), curr_lr)

            self.gscaler.step(self.optimizer)
            self.gscaler.update()

            loss_value = loss.item()

            # self.optimizer.step()
            if self.scheduler is not None:
                # self.scheduler.step()
                self.scheduler.step(epoch + data_iter_step / iters)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            metric_logger.update(loss=loss_value)

            lr = curr_lr
            metric_logger.update(lr=lr)

            loss_value_reduce = all_reduce_mean(loss_value)

            if self.wandb:
                wandb.log(
                    {"train_loss_per_batch": loss_value_reduce, "lr_per_batch": lr}
                )

        metric_logger.synchronize_between_processes()
        print("Averaged train stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        for data, label in self.test_loader:
            with torch.no_grad():
                with amp.autocast(
                    enabled=self.gscaler is not None, dtype=torch.float16
                ):
                    outs = self.model(data.to(device=self.device))
                    loss = self.loss(label.to(device=self.device), outs)
                loss_value = loss.item()
                metric_logger.update(loss=loss_value)

                loss_value_reduce = all_reduce_mean(loss_value)
                if self.wandb:
                    wandb.log({"eval_loss_per_batch": loss_value_reduce})

        metric_logger.synchronize_between_processes()
        print("Averaged eval stats:", metric_logger)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    trainer = Trainer(args)
    trainer.run()


###
# Running without workflow
###
import hydra
import logging


@hydra.main(config_path="../configs/exp", config_name="train_without_workflow")
def run_without_workflow(args):
    num_gpus = torch.cuda.device_count()
    logging.info(
        f"Process ID {os.getpid()} executing task {args.experiment} with {num_gpus} gpu(s)."
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)


if __name__ == "__main__":
    run_without_workflow()
