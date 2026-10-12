#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
exec /workspace/.venv/bin/python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_NODEID" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m samudra.experiments.diffusion_scratch_distributed \
  --root "$CAMPAIGN_ROOT" --arm "$ARM" --updates 20320 --checkpoint-every 100 "$@"
