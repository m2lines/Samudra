import numpy as np
import random
import builtins
import datetime
import logging
import os

import torch
import torch.distributed as dist
from torch import inf

log = logging.getLogger(__name__)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def suppress_prints(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def suppress_logging(is_master):
    if not is_master:
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.WARN)


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


def init_distributed_mode(args):
    assert args.distributed

    if "RANK" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.dist_url = "env://"
        args.gpu = args.rank % torch.cuda.device_count()
    elif "SLURM_PROCID" in os.environ:
        if not args.dist_url:
            if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
                args.dist_url = "tcp://{}:{}".format(
                    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]
                )
            else:
                args.dist_url = "tcp://localhost:40000"
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    log.info(
        "| distributed init (rank {}): {}, gpu {}, world_size {}".format(
            args.rank, args.dist_url, args.gpu, args.world_size
        )
    )

    if not dist.is_initialized():
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.distributed.barrier()
        suppress_prints(args.rank == 0)
    else:
        torch.distributed.barrier()
    suppress_logging(args.rank == 0)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
