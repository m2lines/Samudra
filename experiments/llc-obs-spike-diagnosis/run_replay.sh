#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
# Runs inside the #1037 container (replay.sbatch). Fork once, train, then gate the control.
set -euo pipefail
window=${1:?window} arm=${2:?arm}
root=${SPIKE_RUN_ROOT:?Set the replay run root}
python=/workspace/.venv/bin/python
export PYTHONPATH=/spike-code/src PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
here=/spike-code/experiments/llc-obs-spike-diagnosis
cd /spike-code
arm_dir=$("$python" "$here/make_arm.py" "$window" "$arm" --run-root "$root")
fork_args=$("$python" - "$arm_dir/arm.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))
print(f"--parent {a['parent']}" + (f" --reseed {a['reseed']}" if a['reseed'] is not None else ""))
PY
)
if [[ ! -f "$arm_dir/lineage.json" ]]; then
  # shellcheck disable=SC2086
  "$python" "$here/fork_replay.py" $fork_args --config "$arm_dir/train.yaml"
fi
"$python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m samudra.observations.train "$arm_dir/train.yaml"
if [[ "$arm" == control ]]; then
  original=$("$python" -c "import json;print(json.load(open('$arm_dir/arm.json'))['original_metrics'])")
  "$python" "$here/check_reproduction.py" --replay "$arm_dir/metrics.jsonl" --original "$original" \
    | tee "$arm_dir/reproduction.json"
fi
