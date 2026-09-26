#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root="$1"
: "${RUN_GROUP:?Completed A/B run group required}"
arms=(A B A B)
seeds=(1729 1729 1730 1730)
arm="${arms[$SLURM_LOCALID]}"
seed="${seeds[$SLURM_LOCALID]}"
run="$root/runs/$RUN_GROUP/$arm-$seed"
exec /workspace/.venv/bin/python -m samudra.experiments.diffusion_report \
  --root "$root" --checkpoint "$run/observation/best.pt" \
  --output "$run/report-eight-members" --members 8 --seed 4041729 --annual
