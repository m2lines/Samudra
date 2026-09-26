#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
case "${QUALIFICATION_STAGE:-initial}" in
  initial)
    roles=(baseline deterministic diffusion geometry)
    exec /workspace/.venv/bin/python -m samudra.experiments.diffusion_beta_qualify \
      --root "$1" --role "${roles[$SLURM_LOCALID]}"
    ;;
  observation)
    arms=(A B B B)
    steps=(16 8 16 32)
    arm="${arms[$SLURM_LOCALID]}"
    count="${steps[$SLURM_LOCALID]}"
    exec /workspace/.venv/bin/python -m samudra.experiments.diffusion_observation_qualify \
      --root "$1" --arm "$arm" --sampling-steps "$count" \
      --output "$1/runs/observation-qualification/$SLURM_JOB_ID/$arm-$count"
    ;;
  *) echo "Unknown qualification stage" >&2; exit 2 ;;
esac
