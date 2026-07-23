# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

logger = logging.getLogger(__name__)

import os

from samudra.config import DistributedConfig, EvalBackendConfig, TrainBackendConfig
from samudra.utils.device import set_device
from samudra.utils.distributed import init_distributed_mode


def _cuda_diagnostics() -> str:
    """Return a short string that helps debug why CUDA is (not) available."""
    visible = os.getenv("CUDA_VISIBLE_DEVICES", None)
    return (
        "CUDA diagnostics: "
        f"torch={torch.__version__}, "
        f"torch.version.cuda={torch.version.cuda}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()}, "
        f"torch.cuda.device_count()={torch.cuda.device_count()}, "
        f"CUDA_VISIBLE_DEVICES={visible!r}"
    )


def init_train_backend(
    backend: TrainBackendConfig,
) -> tuple[torch.device, DistributedConfig | None]:
    """Given backend config, get the device and (if any) distributed configuration."""
    match backend:
        case "cpu":
            device = torch.device("cpu")
            dist_cfg = None
        case "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Requested backend='cuda' but CUDA is not available. "
                    + _cuda_diagnostics()
                )
            device = torch.device("cuda")
            dist_cfg = None
        case "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Requested backend='nccl' but CUDA is not available. "
                    + _cuda_diagnostics()
                )
            device = torch.device("cuda")
            dist_cfg = init_distributed_mode()
        case "auto" if torch.cuda.is_available():
            logger.info("auto backend detected CUDA")
            device = torch.device("cuda")
            try:
                dist_cfg = init_distributed_mode()
                logger.info("succeeded in initializing distributed mode")
            except RuntimeError as e:
                logger.info(
                    f"Failed to initialize distributed mode, running on single GPU.",
                    exc_info=e,
                )
                dist_cfg = None
        case "auto":
            logger.warning(
                "auto backend: cuda not found, using CPU. " + _cuda_diagnostics()
            )
            device = torch.device("cpu")
            dist_cfg = None
        case _:
            raise ValueError(f"Invalid backend: {backend}")

    # We set this globally so we don't need to hand the device around.
    # See https://github.com/m2lines/Samudra/issues/87.
    set_device(device)

    return device, dist_cfg


def init_eval_backend(backend: EvalBackendConfig) -> torch.device:
    """Given evaluation backend config, get the device."""
    match backend:
        case "cpu":
            device = torch.device("cpu")
        case "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Requested backend='cuda' but CUDA is not available. "
                    + _cuda_diagnostics()
                )
            device = torch.device("cuda")
        case "auto" if torch.cuda.is_available():
            logger.info("auto backend detected CUDA")
            device = torch.device("cuda")
        case "auto":
            logger.warning(
                "auto backend: cuda not found, using CPU. " + _cuda_diagnostics()
            )
            device = torch.device("cpu")
        case _:
            raise ValueError(f"Invalid backend: {backend}")

    set_device(device)
    return device
