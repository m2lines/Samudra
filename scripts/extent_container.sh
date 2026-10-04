#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
stage=$1
arm=$2
root=$3
python=/workspace/.venv/bin/python
"$python" - "$root" <<'PY'
import hashlib,json,pathlib,sys
import torch, gsw
from samudra.experiments.observation_model import ObservationTransfer
from samudra.rust_data import create_rust_io_runtime
root=pathlib.Path(sys.argv[1])
contract=json.loads((root/'paths.json').read_text())
extensions=list((root/'runtime').rglob('*.so'))
assert len(extensions)==1
assert hashlib.sha256(extensions[0].read_bytes()).hexdigest()==contract['rust_extension_sha256']
runtime=create_rust_io_runtime(4)
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda}, flush=True)
PY
if [[ "$stage" == diagnose ]]; then
  baseline=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline_root"])' "$root/paths.json")
  exec "$python" -m samudra.experiments.extent_diagnostics --root "$root" --baseline-root "$baseline"
fi
if [[ "$stage" == train-representation ]]; then
  pids=()
  for gpu in 0 1 2 3; do
    (
      export CUDA_VISIBLE_DEVICES=$gpu
      case "$gpu" in
        0) sequence=(U-aux01 A-global);;
        1) sequence=(U-aux10 A-multitask);;
        2) sequence=(W-global);;
        3) sequence=(W-multitask);;
      esac
      status=0
      for next_arm in "${sequence[@]}"; do
        "$python" /extent-code/scripts/run_extent_wave.py --root "$root" --stage train --arm "$next_arm" \
          > "$root/logs/$next_arm-$SLURM_JOB_ID.log" 2>&1 || status=1
      done
      exit "$status"
    ) &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  exit "$status"
fi
if [[ "$stage" != train-all && "$stage" != train-ablations && "$stage" != train-initializer ]]; then
  exec "$python" /extent-code/scripts/run_extent_wave.py --root "$root" --stage "$stage" --arm "$arm"
fi
arms=(U-global L-global L-multitask U-multitask)
if [[ "$stage" == train-ablations ]]; then
  arms=(U-omit-patch U-patch-lr01 U-patch-truth U-patch-1step)
fi
if [[ "$stage" == train-initializer ]]; then
  arms=(U-patch-detach U-patch-forecast U-patch-loss01 L-patch-truth)
fi
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
