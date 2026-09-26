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
for phase in om4 observation; do
  /workspace/.venv/bin/python -m samudra.experiments.diffusion_native_controls \
    --root "$root" --checkpoint "$run/$phase/best.pt" \
    --output "$run/${CONTROL_LABEL:-native-controls-v1}/$phase"
done
