#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$1 run=$2 arm=$3 mode=$4
export PYTHONPATH=/llc-code/src PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /llc-code
python=/workspace/.venv/bin/python
if [[ "$mode" == production ]]; then
 "$python" - "$root" "$run" <<'PY'
import hashlib,json,pathlib,sys,yaml
from samudra.observations.config import ObservationTrainConfig
root,run=map(pathlib.Path,sys.argv[1:])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((root/'qualification/COMPLETE.json').read_text())
identity=proof['checkpoints']['2000']['identity']
assert proof['verification_script_sha256']==sha(root/'scripts/verify_viewer.py')
assert identity['export_script_sha256']==sha(root/'scripts/export_viewer.py')
assert identity['extra_proof_sha256']==sha(root/'forcing/COMPLETE.json')
assert identity['hours']==720 and identity['origins']==2 and identity['world_size']==4
cfg=ObservationTrainConfig.model_validate(yaml.safe_load((run/'train.yaml').read_text()))
assert identity['manifest']['config']['model']==cfg.scientific_config()['model']
PY
fi
if [[ "$mode" == qualification ]]; then
 checkpoints=(main-2000.pt) updates=(2000) origins=2 output="$root/qualification"
else
 checkpoints=(main-10000.pt endpoint.pt) updates=(10000 11000) origins=8 output="$root/bundle/$arm"
fi
"$python" -m torch.distributed.run --standalone --nproc_per_node=4 "$root/scripts/export_viewer.py" \
 --run-root "$run" --extra "$root/forcing" --output "$output" --hours 720 --origins "$origins" --checkpoints "${checkpoints[@]}"
"$python" "$root/scripts/verify_viewer.py" --run-root "$run" --export-root "$output" --hours 720 --origins "$origins" --updates "${updates[@]}"
