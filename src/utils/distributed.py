import builtins
import datetime
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


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


def suppress_prints(is_master):
    """This function disables printing when not in master process."""
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


def init_distributed_mode(cfg):
    # assert cfg.distributed
    if not cfg.distributed:
        return
    if "RANK" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.gpu = int(os.environ["LOCAL_RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.dist_url = "env://"
        cfg.gpu = cfg.rank % torch.cuda.device_count()
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
        cfg.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
        if "MASTER_ADDR" not in os.environ:
            cfg.dist_url = "tcp://localhost:40000"  # Local execution
        else:
            cfg.dist_url = None  # Slurm execution

    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = "nccl"
    logging.info(
        "| distributed init (rank {}), gpu {}, world_size {}".format(
            cfg.rank, cfg.gpu, cfg.world_size
        )
    )

    if not dist.is_initialized():
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        torch.distributed.barrier()
        suppress_prints(cfg.rank == 0)
    else:
        torch.distributed.barrier()
    suppress_logging(cfg.rank == 0)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        # Convert to tensor using recommended approach
        if torch.is_tensor(x):
            x_reduce = x.clone().detach().cuda()
        else:
            x_reduce = torch.FloatTensor([x]).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
