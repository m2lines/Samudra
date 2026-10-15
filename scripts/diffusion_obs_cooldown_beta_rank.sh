#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
train() {
  /workspace/.venv/bin/python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_NODEID" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    -m samudra.experiments.diffusion_scratch_obs_cooldown \
    --root "$CAMPAIGN_ROOT" --parent "$COOLDOWN_PARENT" --output "$COOLDOWN_OUTPUT" \
    --updates 500 "$@"
}
# Exercise a real save/resume using the first four updates of the same budget.
if [[ ! -f "$COOLDOWN_OUTPUT/last.pt" ]]; then train --stop-at 4; fi
train --resume
variant=raw
if [[ "$SLURM_NODEID" == 1 ]]; then variant=ema; fi
/workspace/.venv/bin/python -m samudra.experiments.diffusion_scratch_evaluate \
  --root "$CAMPAIGN_ROOT" --checkpoint "$COOLDOWN_OUTPUT/final-$variant.pt" \
  --output "$COOLDOWN_OUTPUT/evaluation-$variant" --processor-prediction residual \
  --expected-om4-updates 7721 --expected-observation-updates 4500 --gpus 0,1,2
