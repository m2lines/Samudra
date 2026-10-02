#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
stage=$1
arm=$2
root=$3
python=/workspace/.venv/bin/python
"$python" - <<'PY'
import torch, gsw
from samudra.experiments.observation_model import ObservationTransfer
from samudra.rust_data import create_rust_io_runtime
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda}, flush=True)
PY
if [[ "$stage" != train-all ]]; then
  exec "$python" /extent-code/scripts/run_extent_wave.py --root "$root" --stage "$stage" --arm "$arm"
fi
arms=(U-global L-global L-multitask U-multitask)
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu "$python" /extent-code/scripts/run_extent_wave.py \
    --root "$root" --stage train --arm "${arms[$gpu]}" \
    > "$root/logs/${arms[$gpu]}-$SLURM_JOB_ID.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
