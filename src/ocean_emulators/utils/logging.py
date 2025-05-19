import datetime
import functools
import logging
import resource
import sys
import time
import traceback
import warnings
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

if TYPE_CHECKING:
    from ocean_emulators.datasets import TrainData


def handle_logging(cfg):
    # Set up logging
    logger = logging.getLogger()  # Use the root logger or specify a name if needed
    logger.setLevel(logging.DEBUG if cfg.debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    # STDOUT handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG if cfg.debug else logging.INFO)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    # Add experiment log file handler
    experiment_log_path = cfg.experiment.output_dir / "experiment.log"
    experiment_handler = logging.FileHandler(experiment_log_path)
    experiment_handler = logging.FileHandler(experiment_log_path)
    experiment_handler.setLevel(logging.INFO)  # Capture info and above
    experiment_handler.setFormatter(fmt)
    logger.addHandler(experiment_handler)

    # Add separate error log file handler
    error_log_path = cfg.experiment.output_dir / "error.log"
    error_handler = logging.FileHandler(error_log_path)
    error_handler.setLevel(logging.WARNING)  # Capture warnings and errors
    error_handler.setFormatter(fmt)
    logger.addHandler(error_handler)


def handle_warnings():
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.info("\n=== Warning Details ===")
        logging.info(f"Message: {message}")
        logging.info(f"Category: {category}")
        logging.info(f"File: {filename}")
        logging.info(f"Line: {lineno}")
        logging.info("\nFull stack trace:")
        stack = traceback.extract_stack()[:-1]  # Remove current frame
        for frame in stack:
            logging.info(
                f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
            )
            if frame.line:
                logging.info(f"    {frame.line}")
        logging.info("=====================\n")

    warnings.showwarning = warning_handler


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0
        self.fmt: str = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

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


class MetricLogger:
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
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, data_loader: DataLoader["TrainData"], print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.perf_counter()
        end = time.perf_counter()
        iter_time = SmoothedValue(fmt="{value:.3f}({avg:.3f})", window_size=print_freq)
        data_wait_time = SmoothedValue(
            fmt="{value:.3f}({avg:.3f})", window_size=print_freq
        )
        data_load_time = SmoothedValue(
            fmt="{value:.3f}({avg:.3f})", window_size=print_freq
        )
        self.meters["iter_time"] = iter_time
        self.meters["data_wait_time"] = data_wait_time
        self.meters["data_load_time"] = data_load_time
        space_fmt = ":" + str(len(str(len(data_loader)))) + "d"
        log_msg_list: list[str] = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
        ]
        log_msg_list.append("max cpu mem: {cpu_memory:.0f}")
        if torch.cuda.is_available():
            log_msg_list.append("max gpu mem: {gpu_memory:.0f}")
        log_msg = self.delimiter.join(log_msg_list)
        KB = 1024.0
        MB = 1024.0 * 1024.0
        for obj in data_loader:
            data_wait_time.update(time.perf_counter() - end)
            if obj.load_stats is not None:
                data_load_time.update(obj.load_stats.load_time_seconds)
            yield obj
            iter_time.update(time.perf_counter() - end)
            if i % print_freq == 0 or i == len(data_loader) - 1:
                eta_seconds = iter_time.global_avg * (len(data_loader) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                named_metrics = dict(
                    eta=eta_string,
                    meters=str(self),
                    cpu_memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB,
                )
                if torch.cuda.is_available():
                    named_metrics["gpu_memory"] = torch.cuda.max_memory_allocated() / MB

                logging.info(log_msg.format(i, len(data_loader), **named_metrics))

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            i += 1
            end = time.perf_counter()
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(
            f"{header} Total time: {total_time_str} "
            f"({total_time / len(data_loader):.4f} s / it)"
        )


def get_model_summary(model: torch.nn.Module, num_input_channels: int):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f"Number of parameters: {params}")
    logging.info(summary(model))


def elapsed(func=None, *, level: int = logging.INFO):
    """Log the time taken to execute a function.

    Implementation inspired by this blog post:
      https://pybit.es/articles/decorator-optional-argument/

    Args:
        func (callable): The function to decorate. If None, returns a partial
            function with the log_level argument.
        level (int): The logging level to use. Default is logging.INFO.
    """
    if func is None:
        return functools.partial(elapsed, level=level)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.log(
            level,
            "%s took %.4f seconds",
            func.__qualname__,
            end_time - start_time,
        )
        return result

    return wrapper
