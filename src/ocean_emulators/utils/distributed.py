import builtins
import datetime
import logging
import os
import random

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from ocean_emulators.config import DistributedConfig


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.benchmark = (
        True  # False # Set to True for better performance but lose reproducibility
    )
    # cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


builtin_print = builtins.print


def print(*args, **kwargs):
    force = kwargs.pop("force", False)
    force = force or (get_world_size() > 8)
    if force:
        now = datetime.datetime.now().time()
        builtin_print(f"[{now}] ", end="")  # print with time stamp
        builtin_print(*args, **kwargs)


def suppress_prints(is_master):
    """This function disables printing when not in master process."""
    if not is_master:
        builtins.print = print


def suppress_logging(is_master):
    """Suppress logging for non-master processes."""
    if not is_master:
        # Get root logger
        root = logging.getLogger()
        root.setLevel(logging.WARNING)

        # Also suppress any existing handlers on the root logger
        for handler in root.handlers:
            handler.setLevel(logging.WARNING)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode() -> DistributedConfig:
    cfg = DistributedConfig()

    if "RANK" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.gpu = int(os.environ["LOCAL_RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.dist_url = "env://"
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        tasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
        if tasks_per_node < torch.cuda.device_count():
            raise RuntimeError(
                f"There are SLURM_NTASKS_PER_NODE is {tasks_per_node}, but this node has {torch.cuda.device_count()} GPUs; the remainder will be idle. "
                "You probably want to use 1 GPU per task and multiple tasks, or else use `torchrun` on a single node. ",
            )
        n_nodes = int(os.environ["SLURM_NNODES"])
        cfg.world_size = tasks_per_node * n_nodes
        cfg.gpu = cfg.rank % torch.cuda.device_count()

        if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
            cfg.dist_url = "env://"
        else:
            cfg.dist_url = "tcp://localhost:40000"
    else:
        raise RuntimeError(
            "Distributed mode requires SLURM or RANK environment variables;"
            " did you mean to use `torchrun`?"
        )

    cfg.dist_backend = "nccl"
    torch.distributed.init_process_group(
        backend=cfg.dist_backend,
        init_method=cfg.dist_url,
        world_size=cfg.world_size,
        rank=cfg.rank,
        device_id=cfg.gpu,
    )
    torch.cuda.set_device(cfg.gpu)
    logger.info(
        f"| distributed init (rank {cfg.rank}), gpu {cfg.gpu}, "
        f"world_size {cfg.world_size}, dist_url {cfg.dist_url}"
    )
    torch.distributed.barrier()
    suppress_prints(cfg.rank == 0)
    suppress_logging(cfg.rank == 0)

    return cfg


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        dist.all_reduce(x)
        x /= world_size

    return x
