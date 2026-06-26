#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Tiny Empire AI Beta NCCL/PyTorch smoke test using the baseline PhysicsNeMo
# container. Submit with different node counts to compare basic distributed
# communication independently from Samudra training.
#
# Example, 2 hosts x 4 GPUs:
#   sbatch --account=nyu_beta_test --partition=beta --qos=standard \
#     --nodes=2 --gres=gpu:4 --time=00:30:00 scripts/empireai_nccl_smoke.sbatch
#
# Example, 9 hosts x 4 GPUs:
#   sbatch --account=nyu_beta_test --partition=beta --qos=standard \
#     --nodes=9 --gres=gpu:4 --time=00:45:00 scripts/empireai_nccl_smoke.sbatch

#SBATCH --job-name=nccl-smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=beta
#SBATCH --account=nyu_beta_test
#SBATCH --qos=standard
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

set -euo pipefail

IMAGE_REF="${IMAGE_REF:-nvcr.io/nvidia/physicsnemo/physicsnemo:25.11}"
OUTPUT_BASE="${OUTPUT_BASE:-${HOME}/runs/nccl-smoke}"
MASTER_PORT="${MASTER_PORT:-29501}"
NCCL_TEST_ELEMENTS="${NCCL_TEST_ELEMENTS:-1 1024 1048576 4194304}"
NCCL_TEST_ITERS="${NCCL_TEST_ITERS:-5}"

mkdir -p "${OUTPUT_BASE}"
RUN_DIR="${OUTPUT_BASE}/${SLURM_JOB_ID:-manual}-nodes${SLURM_NNODES}"
mkdir -p "${RUN_DIR}"

gpus_on_node="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "${gpus_on_node}" ]]; then
  gpus_on_node="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*gres:gpu:\([0-9][0-9]*\).*/\1/p' | head -n 1)"
fi
GPUS_PER_NODE="$(printf '%s\n' "${gpus_on_node}" | grep -Eo '[0-9]+' | head -n 1 || true)"
if [[ -z "${GPUS_PER_NODE}" ]]; then
  echo "Could not determine GPUS_PER_NODE; set it explicitly." >&2
  exit 2
fi
export GPUS_PER_NODE

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_ADDR MASTER_PORT
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# Avoid many-node imports racing on the same shared ${HOME}/.cache/enroot path.
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/tmp/enroot-cache-${USER}}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/tmp/enroot-data-${USER}}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-/tmp/enroot-runtime-${USER}-${SLURM_JOB_ID:-manual}}"

cat > "${RUN_DIR}/nccl_smoke.py" <<'PY'
import datetime as dt
import json
import os
import socket
import time

import torch
import torch.distributed as dist


def log(payload):
    payload = {"time": dt.datetime.now(dt.UTC).isoformat(), **payload}
    print(json.dumps(payload, sort_keys=True), flush=True)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=5))
    hostname = socket.gethostname()
    props = torch.cuda.get_device_properties(device)

    if rank == 0:
        log(
            {
                "event": "versions",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                "world_size": world_size,
            }
        )

    log(
        {
            "event": "rank_ready",
            "rank": rank,
            "local_rank": local_rank,
            "hostname": hostname,
            "device": torch.cuda.get_device_name(device),
            "total_memory_gib": round(props.total_memory / 1024**3, 2),
        }
    )

    rank_tensor = torch.tensor([rank + 1], device=device, dtype=torch.float32)
    dist.all_reduce(rank_tensor, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    if rank_tensor.item() != expected:
        raise RuntimeError(f"rank-sum all_reduce failed: got {rank_tensor.item()}, expected {expected}")

    gathered = [torch.empty(1, device=device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([rank], device=device, dtype=torch.int64))
    got = [int(t.item()) for t in gathered]
    if got != list(range(world_size)):
        raise RuntimeError(f"all_gather failed: got {got}")

    elements = [int(x) for x in os.environ.get("NCCL_TEST_ELEMENTS", "").split()]
    iters = int(os.environ.get("NCCL_TEST_ITERS", "5"))
    for n in elements:
        x = torch.full((n,), float(rank + 1), device=device)
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x.fill_(float(rank + 1))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if rank == 0:
            mib = n * x.element_size() / 1024**2
            log(
                {
                    "event": "all_reduce_timing",
                    "elements": n,
                    "mib_per_rank": round(mib, 2),
                    "iterations": iters,
                    "seconds_total": round(elapsed, 6),
                    "seconds_per_iter": round(elapsed / iters, 6),
                }
            )

    dist.barrier()
    if rank == 0:
        log({"event": "success"})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

echo "Job id:        ${SLURM_JOB_ID}"
echo "Nodes:         ${SLURM_NNODES}"
echo "GPUs/node:     ${GPUS_PER_NODE}"
echo "World size:    $(( SLURM_NNODES * GPUS_PER_NODE ))"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Image:         ${IMAGE_REF}"
echo "Run dir:       ${RUN_DIR}"
echo "Enroot cache:  ${ENROOT_CACHE_PATH}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-<unset>}"
echo "NCCL_IB_HCA:        ${NCCL_IB_HCA:-<unset>}"
echo "NCCL_OOB_NET_ENABLE:${NCCL_OOB_NET_ENABLE:-<unset>}"
echo "MELLANOX_VISIBLE_DEVICES: ${MELLANOX_VISIBLE_DEVICES:-<unset>}"
echo "Elements:      ${NCCL_TEST_ELEMENTS}"
echo "Iterations:    ${NCCL_TEST_ITERS}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  --container-image="${IMAGE_REF}" \
  --container-mount-home \
  --container-mounts="${RUN_DIR}:${RUN_DIR}" \
  --container-workdir="${RUN_DIR}" \
  bash -lc '
    set -euo pipefail
    exec python -m torch.distributed.run \
      --nnodes="${SLURM_NNODES}" \
      --nproc_per_node="${GPUS_PER_NODE}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      --node_rank="${SLURM_PROCID}" \
      "${PWD}/nccl_smoke.py"
  '
