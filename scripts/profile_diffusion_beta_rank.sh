#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
python=/workspace/.venv/bin/python
if [[ "$1" == single ]]; then
  if [[ "$SLURM_NODEID" == 0 ]]; then
    pids=()
    gpu=0
    for variant in base channels_last no_sampler_sync; do
      profile=()
      if [[ "$variant" == base ]]; then profile+=(--profile); fi
      CUDA_VISIBLE_DEVICES=$gpu "$python" /profile-code/scripts/profile_diffusion_full_sampler.py --root "$DATA_ROOT" --output "$PROFILE_OUTPUT/single-$variant" --variants "$variant" --warmup 1 --repeats 2 "${profile[@]}" > "$PROFILE_OUTPUT/single-$variant.log" 2>&1 &
      pids+=("$!")
      gpu=$((gpu + 1))
    done
    result=0
    for pid in "${pids[@]}"; do wait "$pid" || result=1; done
    exit "$result"
  else
    CUDA_VISIBLE_DEVICES=0 "$python" /profile-code/scripts/profile_diffusion_decoder_batch.py --root "$DATA_ROOT" --output "$PROFILE_OUTPUT/decoder-batch.json" > "$PROFILE_OUTPUT/decoder-batch.log" 2>&1
  fi
else
  export NCCL_DEBUG_FILE="$PROFILE_OUTPUT/nccl-%h-%p.log"
  exec "$python" -m torch.distributed.run --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_NODEID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" /profile-code/scripts/profile_diffusion_full_sampler.py --root "$DATA_ROOT" --output "$PROFILE_OUTPUT/eight-base" --variants base --warmup 1 --repeats 2
fi
