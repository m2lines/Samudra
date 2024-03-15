import torch
import torch.nn as nn
import numpy as np
import pickle
import resource
import shutil
import signal
import sys
import tempfile
import time
import datetime
from collections import defaultdict, deque
from pathlib import Path
import torch.distributed as dist
import wandb
import logging

from .dist_utils import is_dist_avail_and_initialized, all_reduce_mean

log = logging.getLogger(__name__)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{value:.3f}({avg:.3f})", window_size=print_freq)
        data_time = SmoothedValue(fmt="{value:.3f}({avg:.3f})", window_size=print_freq)
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        log_msg.append("max cpu mem: {cpu_memory:.0f}")
        if torch.cuda.is_available():
            log_msg.append("max gpu mem: {gpu_memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        KB = 1024.0
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    log.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            cpu_memory=resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            / KB,
                            gpu_memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    log.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            cpu_memory=resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            / KB,
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log.info(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def loss_KE_pointwise(data, out):
    return ((data[:, :2] ** 2 - out[:, :2] ** 2) ** 2).mean()


def train_one_epoch(
    epoch,
    model,
    train_loader,
    N_in,
    N_extra,
    hist,
    loss_fn,
    optimizer,
    scheduler,
    steps,
    weight,
    device,
    wandb_flag,
):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    iters = len(train_loader)

    for data_iter_step, data in enumerate(
        metric_logger.log_every(train_loader, 1, header)
    ):

        optimizer.zero_grad()
        outs = model(data[0].to(device=device))
        outs = outs
        loss = loss_fn(data[1].to(device=device), outs) * weight[0]

        if len(weight) == 1:
            loss.backward()
        else:
            for step in range(1, steps):

                if (step == 1) or (hist == 0):
                    step_in = torch.concat(
                        (outs, data[int(step * 2)][:, N_in:].to(device=device)), 1
                    )
                    outs_old = outs
                elif (step > 1) and (hist == 1):
                    step_in = torch.concat(
                        (
                            outs,
                            data[int(step * 2)][:, N_in : (N_in + N_extra)].to(
                                device=device
                            ),
                            outs_old,
                        ),
                        1,
                    )
                    outs_old = outs
                else:
                    step_in = torch.concat(
                        (
                            outs,
                            data[int(step * 2)][:, N_in : (N_in + N_extra)].to(
                                device=device
                            ),
                            outs_old,
                            step_in[:, (N_in + N_extra) : -N_in],
                        ),
                        1,
                    )
                    outs_old = outs

                outs = model(step_in)
                outs = outs

                loss += (
                    loss_fn(data[int(step * 2 + 1)].to(device=device), outs)
                    * weight[step]
                )
            loss.backward()

        loss_value = loss.item()

        optimizer.step()
        if scheduler is not None:
            # scheduler.step()
            scheduler.step(epoch + data_iter_step/iters)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = all_reduce_mean(loss_value)

        if wandb_flag:
            wandb.log({"train_loss_per_batch": loss_value_reduce, "lr_per_batch": lr})

    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model, test_loader, device, wandb_flag):
    model.eval()
    mse = nn.MSELoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    for data, label in test_loader:
        with torch.no_grad():
            outs = model(data.to(device=device))
            loss = mse(outs, label.to(device=device))
            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            loss_value_reduce = all_reduce_mean(loss_value)
            if wandb_flag:
                wandb.log({"eval_loss_per_batch": loss_value_reduce})

    metric_logger.synchronize_between_processes()
    print("Averaged eval stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
