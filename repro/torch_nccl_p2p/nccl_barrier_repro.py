from __future__ import annotations

import datetime as dt
import os
import socket
import time

import torch
import torch.distributed as dist


def _env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return int(value)


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this repro.")

    rank = _env_int("RANK")
    local_rank = _env_int("LOCAL_RANK")
    world_size = _env_int("WORLD_SIZE")
    timeout_seconds = int(os.environ.get("TIMEOUT_SECONDS", "120"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=local_rank,
        timeout=dt.timedelta(seconds=timeout_seconds),
    )

    host = socket.gethostname()
    _log(
        " ".join(
            [
                f"[rank {rank}]",
                f"host={host}",
                f"local_rank={local_rank}",
                f"device={torch.cuda.current_device()}",
                f"world_size={world_size}",
                f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'unset')}",
                "starting barrier",
            ]
        )
    )

    barrier_start = time.monotonic()
    dist.barrier()
    barrier_elapsed = time.monotonic() - barrier_start
    _log(f"[rank {rank}] barrier completed in {barrier_elapsed:.3f}s")

    tensor = torch.tensor([rank + 1], device=device, dtype=torch.float32)
    reduce_start = time.monotonic()
    dist.all_reduce(tensor)
    reduce_elapsed = time.monotonic() - reduce_start
    _log(
        f"[rank {rank}] all_reduce completed in {reduce_elapsed:.3f}s "
        f"result={tensor.item():.1f}"
    )

    dist.destroy_process_group()
    _log(f"[rank {rank}] done")


if __name__ == "__main__":
    main()
