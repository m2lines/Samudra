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


def decomposed_mse_mae(pred, out, weight=0.01):
    full_mse = nn.functional.mse_loss(pred, out, reduction="none")
    mse_channels = torch.mean(full_mse, dim=(0, 2, 3))
    full_mae = nn.functional.l1_loss(pred, out, reduction="none")
    mae_channels = torch.mean(full_mae, dim=(0, 2, 3))
    return mse_channels + weight * mae_channels


def decomposed_mse(pred, out):
    full_mse = nn.functional.mse_loss(pred, out, reduction="none")
    mse_channels = torch.mean(full_mse, dim=(0, 2, 3))
    return mse_channels


def decomposed_mse_scaled(pred, out, scaling):
    full_mse = nn.functional.mse_loss(pred, out, reduction="none")
    mse_channels = torch.mean(full_mse, dim=(0, 2, 3)) * scaling
    return mse_channels


def decomposed_mse_diff_weighted(pred, out):
    # Split out into 2 parts - xt and xt+1
    # This works on the assumption that hist = 1.
    N, C, H, W = out.shape
    out_t = out[:, : C // 2, :, :]
    out_tp1 = out[:, C // 2 :, :, :]
    diff_weights = torch.sqrt(torch.mean((out_t - out_tp1) ** 2, dim=(0, 2, 3)))
    indices = np.arange(0, C // 2 - 1, 19)
    diff_weights[indices[:, None] + np.arange(11, 19)] = diff_weights[
        indices + 11, None
    ]
    diff_weights = diff_weights.repeat_interleave(2)

    full_mse = nn.functional.mse_loss(pred, out, reduction="none")
    mse_channels = torch.mean(full_mse, dim=(0, 2, 3))
    mse_channels = mse_channels
    return mse_channels


def decomposed_mse_cos_weighted(pred, out, cos):
    full_mse = nn.functional.mse_loss(pred, out, reduction="none")
    mse_channels_lat = torch.mean(full_mse, dim=(0, 3))
    mse_channels = torch.mean(mse_channels_lat * cos, dim=(1))
    return mse_channels


def extract_wet(wet_zarr, hist):
    depths = [depth_str.replace("_", ".") for depth_str in DEPTH_LEVELS]
    if "zos" in depths:
        zos_index = depths.index("zos")
        depths[zos_index] = str(wet_zarr.lev.values[0])
        assert depths[zos_index] == "2.5"
    depths = [float(depth) for depth in depths]
    wet = wet_zarr.sel(lev=depths)
    wet = torch.from_numpy(wet.to_array().to_numpy().squeeze())
    wet = torch.concat([wet] * (hist + 1), dim=0)
    print(wet.shape)
    return wet


def extract_surface_wet(wet_zarr):
    return torch.from_numpy(wet_zarr.isel(lev=0).to_array().to_numpy().squeeze())

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)