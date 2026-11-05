# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Check the shared NCCL startup and reductions before Torch qualification."""

import builtins
import json
import os

import torch
import torch.distributed as dist

from samudra.backend import init_train_backend


def main():
    report = builtins.print
    required = {"NCCL_P2P_DISABLE": "1", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"}
    observed = {name: os.environ.get(name) for name in required}
    if observed != required:
        raise RuntimeError(f"Missing required Torch RTX transport settings: {observed}")
    rank = int(os.environ["RANK"])
    report(
        json.dumps({"event": "nccl_preflight_start", "rank": rank, **observed}),
        flush=True,
    )
    init_train_backend("nccl")
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    for _ in range(10):
        value = torch.tensor([rank + 1], dtype=torch.int64, device=device)
        dist.all_reduce(value)
        if value.item() != world_size * (world_size + 1) // 2:
            raise AssertionError("NCCL reduction returned an incorrect sum")
    dist.barrier()
    report(
        json.dumps(
            {
                "event": "nccl_preflight_passed",
                "rank": rank,
                "world_size": world_size,
                "device": str(device),
                "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "reductions": 10,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
