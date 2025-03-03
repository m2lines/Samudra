import logging

import torch

from config import DistributedConfig, EvalBackendConfig, TrainBackendConfig
from utils.device import set_device
from utils.distributed import init_distributed_mode


def init_train_backend(
    backend: TrainBackendConfig,
) -> tuple[torch.device, DistributedConfig | None]:
    """Given backend config, get the device and (if any) distributed configuration."""
    match backend:
        case "cpu":
            device = torch.device("cpu")
            dist_cfg = None
        case "cuda":
            device = torch.device("cuda")
            dist_cfg = None
        case "nccl":
            device = torch.device("cuda")
            dist_cfg = init_distributed_mode()
        case "auto" if torch.cuda.is_available():
            logging.info("auto backend detected CUDA")
            device = torch.device("cuda")
            try:
                dist_cfg = init_distributed_mode()
                logging.info("succeeded in initializing distributed mode")
            except RuntimeError as e:
                logging.info(
                    f"Failed to initialize distributed mode, running on single node.",
                    exc_info=e,
                )
                dist_cfg = None
        case "auto":
            logging.info("auto backend: cuda not found, using CPU")
            device = torch.device("cpu")
            dist_cfg = None
        case _:
            raise ValueError(f"Invalid backend: {backend}")

    # We set this globally so we don't need to hand the device around.
    # See https://github.com/suryadheeshjith/Ocean_Emulator/issues/87.
    set_device(device)

    return device, dist_cfg


def init_eval_backend(backend: EvalBackendConfig) -> torch.device:
    """Given evaluation backend config, get the device."""
    match backend:
        case "cpu":
            device = torch.device("cpu")
        case "cuda":
            device = torch.device("cuda")
        case "auto" if torch.cuda.is_available():
            logging.info("auto backend detected CUDA")
            device = torch.device("cuda")
        case "auto":
            logging.info("auto backend: cuda not found, using CPU")
            device = torch.device("cpu")
        case _:
            raise ValueError(f"Invalid backend: {backend}")

    return device
