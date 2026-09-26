#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root="$1"
: "${RUN_GROUP:?Run group required}"
: "${QUALIFICATION_JOB:?Observation qualification required}"
: "${TRAIN_PHASE:?Training phase required}"
: "${TRAIN_UPDATES:?Update ceiling required}"
: "${TRAIN_HOURS:?Training time ceiling required}"
arms=(A B A B)
seeds=(1729 1729 1730 1730)
arm="${arms[$SLURM_LOCALID]}"
seed="${seeds[$SLURM_LOCALID]}"
name="$RUN_GROUP-$arm-$seed"
output="$root/runs/$RUN_GROUP/$arm-$seed/$TRAIN_PHASE"
extra=()
if [[ "$TRAIN_PHASE" == observation ]]; then
  extra+=(--pretrained "$root/runs/$RUN_GROUP/$arm-$seed/om4/best.pt")
fi
exec /workspace/.venv/bin/python -m samudra.experiments.diffusion_train \
  --root "$root" --arm "$arm" --phase "$TRAIN_PHASE" \
  --qualification "$root/runs/observation-qualification/$QUALIFICATION_JOB/$arm-16/QUALIFIED.json" \
  --output "$output" --name "$name-$TRAIN_PHASE" --seed "$seed" \
  --updates "$TRAIN_UPDATES" --hours "$TRAIN_HOURS" \
  --readers 4 --batch-size 1 --checkpoint-every 100 --validate-every 500 \
  --wandb-mode "${WANDB_MODE:-offline}" "${extra[@]}"
