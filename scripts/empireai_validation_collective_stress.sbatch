#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Empire AI Beta validation-aggregation-shaped NCCL stress test.
#
# This intentionally mimics Samudra's W&B-enabled validation aggregation:
# scalar metric reductions followed by many full half-degree map reductions.
# The default mode more closely mimics Samudra's MapAggregator by accumulating
# into persistent [time, lat, lon] buffers, reducing select() views in place,
# and having rank 0 do CPU transfers while other ranks enter the next
# collective.
#
# Example, reproduce the intermittent 2-node placement:
#   sbatch --account=nyu_beta_test --partition=beta --qos=standard \
#     --nodes=2 --gres=gpu:4 --nodelist=b3-12-s1-dgx-03-c02,b3-16-s1-dgx-03-c06 \
#     scripts/empireai_validation_collective_stress.sbatch
#
# Useful overrides:
#   IMAGE_REF=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:<digest>
#   STRESS_REPEATS=10
#   STRESS_RENDER_SLEEP=0.05
#   STRESS_VIEW_MODE=1
#   STRESS_ALLOC_CHURN_MB=512
#   STRESS_DEBUG=1

#SBATCH --job-name=val-coll-stress
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=beta
#SBATCH --account=nyu_beta_test
#SBATCH --qos=standard
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00

set -euo pipefail

IMAGE_REF="${IMAGE_REF:-ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:79564be9d5aa48acc40a803bf60f5b277bacee0fb9ad11e60735ee599e407421}"
OUTPUT_BASE="${OUTPUT_BASE:-${HOME}/runs/validation-collective-stress}"
MASTER_PORT="${MASTER_PORT:-29507}"

# Samudra half-degree validation shape:
# thermo_dynamic_all = 19 levels * 4 variables + zos = 77 map variables.
STRESS_SCALARS="${STRESS_SCALARS:-231}"
STRESS_MAPS="${STRESS_MAPS:-154}"
STRESS_HEIGHT="${STRESS_HEIGHT:-360}"
STRESS_WIDTH="${STRESS_WIDTH:-720}"
STRESS_REPEATS="${STRESS_REPEATS:-5}"
STRESS_HIST="${STRESS_HIST:-1}"
STRESS_VAL_BATCHES="${STRESS_VAL_BATCHES:-8}"
STRESS_VIEW_MODE="${STRESS_VIEW_MODE:-1}"
STRESS_ALLOC_CHURN_MB="${STRESS_ALLOC_CHURN_MB:-0}"
STRESS_RENDER_SLEEP="${STRESS_RENDER_SLEEP:-0}"
STRESS_DEBUG="${STRESS_DEBUG:-0}"
export STRESS_SCALARS STRESS_MAPS STRESS_HEIGHT STRESS_WIDTH STRESS_REPEATS
export STRESS_HIST STRESS_VAL_BATCHES STRESS_VIEW_MODE STRESS_ALLOC_CHURN_MB
export STRESS_RENDER_SLEEP

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
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_4,mlx5_5}"
export MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-all}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ "${STRESS_DEBUG}" == "1" ]]; then
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-INFO}"
else
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi

# Avoid shared-cache import races across nodes.
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/tmp/enroot-cache-${USER}-${SLURM_JOB_ID:-manual}}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/tmp/enroot-data-${USER}-${SLURM_JOB_ID:-manual}}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-/tmp/enroot-runtime-${USER}-${SLURM_JOB_ID:-manual}}"

cat > "${RUN_DIR}/validation_collective_stress.py" <<'PY'
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


def all_reduce_mean(x, world_size):
    dist.all_reduce(x)
    x /= world_size
    return x


def alloc_churn(device, churn_mb):
    if churn_mb <= 0:
        return
    elements = churn_mb * 1024 * 1024 // 4
    scratch = torch.empty(elements, dtype=torch.float32, device=device)
    scratch.fill_(1.0)
    del scratch


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=10), device_id=device)
    hostname = socket.gethostname()

    scalars = int(os.environ["STRESS_SCALARS"])
    maps = int(os.environ["STRESS_MAPS"])
    height = int(os.environ["STRESS_HEIGHT"])
    width = int(os.environ["STRESS_WIDTH"])
    repeats = int(os.environ["STRESS_REPEATS"])
    hist = int(os.environ["STRESS_HIST"])
    val_batches = int(os.environ["STRESS_VAL_BATCHES"])
    view_mode = os.environ["STRESS_VIEW_MODE"] == "1"
    alloc_churn_mb = int(os.environ["STRESS_ALLOC_CHURN_MB"])
    render_sleep = float(os.environ["STRESS_RENDER_SLEEP"])
    target_time = hist

    if rank == 0:
        log(
            {
                "event": "versions",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                "world_size": world_size,
                "scalars_per_repeat": scalars,
                "maps_per_repeat": maps,
                "map_shape": [height, width],
                "repeats": repeats,
                "hist": hist,
                "val_batches": val_batches,
                "view_mode": view_mode,
                "alloc_churn_mb": alloc_churn_mb,
                "render_sleep": render_sleep,
            }
        )

    log(
        {
            "event": "rank_ready",
            "rank": rank,
            "local_rank": local_rank,
            "hostname": hostname,
            "device": torch.cuda.get_device_name(device),
        }
    )

    # Seed per rank so reductions have non-identical inputs but deterministic
    # validation checks.
    torch.manual_seed(1000 + rank)

    dist.barrier()
    for repeat in range(repeats):
        repeat_start = time.perf_counter()

        for i in range(scalars):
            scalar = torch.tensor(float(rank + i + repeat), device=device)
            all_reduce_mean(scalar, world_size)

        if rank == 0:
            log({"event": "map_phase_start", "repeat": repeat})

        if view_mode:
            gen_buffers = []
            target_buffers = []
            for i in range(maps):
                gen_buffer = torch.zeros((hist + 1, height, width), device=device)
                target_buffer = torch.zeros((hist + 1, height, width), device=device)
                for batch in range(val_batches):
                    gen_buffer += torch.full(
                        (hist + 1, height, width),
                        rank + repeat + batch / 10.0 + i / 1000.0,
                        device=device,
                    )
                    target_buffer += torch.full(
                        (hist + 1, height, width),
                        rank - repeat - batch / 10.0 - i / 1000.0,
                        device=device,
                    )
                gen_buffers.append(gen_buffer)
                target_buffers.append(target_buffer)
            alloc_churn(device, alloc_churn_mb)
        else:
            gen_buffers = []
            target_buffers = []

        for i in range(maps):
            if view_mode:
                gen = gen_buffers[i].select(dim=0, index=target_time)
                target = target_buffers[i].select(dim=0, index=target_time)
            else:
                gen = torch.full(
                    (height, width), rank + repeat + i / 1000.0, device=device
                )
                target = torch.full(
                    (height, width), rank - repeat - i / 1000.0, device=device
                )

            all_reduce_mean(gen, world_size)
            all_reduce_mean(target, world_size)

            if rank == 0:
                # Match the MapAggregator shape: error image, full-field gen,
                # full-field target. The CPU copies are the important part;
                # plotting can be approximated by an optional sleep.
                err_np = (gen - target).cpu().numpy()
                gen_np = gen.cpu().numpy()
                target_np = target.cpu().numpy()
                checksum = float(err_np[0, 0] + gen_np[0, 0] + target_np[0, 0])
                if i % 20 == 0 or i == maps - 1:
                    log(
                        {
                            "event": "rank0_render_checkpoint",
                            "repeat": repeat,
                            "map": i,
                            "checksum": round(checksum, 6),
                        }
                    )
                if render_sleep:
                    time.sleep(render_sleep)

        del gen_buffers
        del target_buffers
        dist.barrier()
        torch.cuda.synchronize()
        if rank == 0:
            log(
                {
                    "event": "repeat_done",
                    "repeat": repeat,
                    "seconds": round(time.perf_counter() - repeat_start, 6),
                }
            )

    dist.barrier()
    if rank == 0:
        log({"event": "success"})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

echo "Job id:          ${SLURM_JOB_ID}"
echo "Nodes:           ${SLURM_NNODES}"
echo "GPUs/node:       ${GPUS_PER_NODE}"
echo "World size:      $(( SLURM_NNODES * GPUS_PER_NODE ))"
echo "Master:          ${MASTER_ADDR}:${MASTER_PORT}"
echo "Image:           ${IMAGE_REF}"
echo "Run dir:         ${RUN_DIR}"
echo "NCCL socket IF:  ${NCCL_SOCKET_IFNAME}"
echo "NCCL IB HCAs:    ${NCCL_IB_HCA}"
echo "Mellanox devs:   ${MELLANOX_VISIBLE_DEVICES}"
echo "Debug logs:      ${STRESS_DEBUG}"
echo "Scalars/repeat:  ${STRESS_SCALARS}"
echo "Maps/repeat:     ${STRESS_MAPS}"
echo "Map shape:       ${STRESS_HEIGHT}x${STRESS_WIDTH}"
echo "Repeats:         ${STRESS_REPEATS}"
echo "Hist:            ${STRESS_HIST}"
echo "Val batches:     ${STRESS_VAL_BATCHES}"
echo "View mode:       ${STRESS_VIEW_MODE}"
echo "Alloc churn MB:  ${STRESS_ALLOC_CHURN_MB}"
echo "Rank0 sleep:     ${STRESS_RENDER_SLEEP}"

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
      "${PWD}/validation_collective_stress.py"
  '
