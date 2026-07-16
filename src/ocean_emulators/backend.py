import logging

import torch

logger = logging.getLogger(__name__)

import os

from ocean_emulators.config import (
    DistributedConfig,
    EvalBackendConfig,
    TrainBackendConfig,
)
from ocean_emulators.utils.device import set_device
from ocean_emulators.utils.distributed import (
    init_distributed_mode,
    suppress_logging,
    suppress_prints,
)


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
    ddp_timeout_minutes: int = 60,
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
            dist_cfg = init_distributed_mode(timeout_minutes=ddp_timeout_minutes)
        case "auto" if torch.cuda.is_available():
            logger.info("auto backend detected CUDA")
            device = torch.device("cuda")
            try:
                dist_cfg = init_distributed_mode(timeout_minutes=ddp_timeout_minutes)
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
    # See https://github.com/suryadheeshjith/Ocean_Emulator/issues/87.
    set_device(device)

    return device, dist_cfg


def init_domain_parallel_backend(
    backend: TrainBackendConfig,
) -> tuple[torch.device, DistributedConfig, object]:
    """Initialize NCCL through PhysicsNeMo and return its manager.

    PhysicsNeMo must initialize the default process group itself so its
    DistributedManager singleton has the rank/device state needed to create a
    DeviceMesh. This path is deliberately separate from the existing DDP
    backend, which remains unchanged when domain parallelism is disabled.
    """
    if backend not in ("auto", "nccl"):
        raise ValueError(
            "domain_parallel.enabled=true requires backend='nccl' or 'auto'; "
            f"got {backend!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Domain parallel training requires CUDA. " + _cuda_diagnostics()
        )

    try:
        from physicsnemo.distributed import DistributedManager
    except ImportError as exc:
        raise RuntimeError(
            "domain_parallel.enabled=true but PhysicsNeMo is not importable. "
            "Install the pinned NVIDIA PhysicsNeMo build before launching."
        ) from exc

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dm = DistributedManager()
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "PhysicsNeMo did not initialize torch.distributed. Launch domain "
            "parallel training with torchrun or a supported multi-task SLURM job."
        )

    device = dm.device
    torch.cuda.set_device(device)
    set_device(device)
    dist_cfg = DistributedConfig(
        rank=dm.rank,
        world_size=dm.world_size,
        gpu=dm.local_rank,
        dist_backend="nccl",
        dist_url="env://",
    )
    logger.info(
        "PhysicsNeMo distributed backend initialized: rank=%s local_rank=%s "
        "world_size=%s device=%s",
        dm.rank,
        dm.local_rank,
        dm.world_size,
        device,
    )
    suppress_prints(dm.rank == 0)
    suppress_logging(dm.rank == 0)
    return device, dist_cfg, dm


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
