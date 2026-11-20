#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
# Called inside the pinned PhysicsNeMo container, on four GPUs.
set -euo pipefail
root=${LLC_OBS_RUN_ROOT:?Set the experiment run directory}
mode=${1:?qualification or production}
python=/workspace/.venv/bin/python
export PYTHONPATH=/llc-code/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /llc-code
# Qualification and production must use the exact same code, inputs and model.
"$python" - "$root" <<'PY'
import hashlib,json,pathlib,sys,yaml,torch
root=pathlib.Path(sys.argv[1])
assert torch.cuda.is_available() and torch.cuda.device_count()==4
print(json.dumps(dict(torch=str(torch.__version__),cuda=torch.version.cuda,devices=[torch.cuda.get_device_name(i) for i in range(4)])),flush=True)
config=yaml.safe_load((root/'train.yaml').read_text())
proof=pathlib.Path(config['experiment']['data_root'])/config['data']['snapshots']/'COMPLETE.json'
if not proof.exists(): raise RuntimeError('Prepared data proof is missing')
signature={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {'code':root/'code.tar.gz','config':root/'train.yaml','data':proof}.items()}
signature['world_size']=4
path=root/'qualification-signature.json'
if path.exists() and json.loads(path.read_text())!=signature: raise RuntimeError('Qualification signature differs')
path.write_text(json.dumps(signature,sort_keys=True)+'\n')
PY
# Atomic directory ownership avoids concurrent qualifications across the two queues.
if [[ ! -f "$root/QUALIFIED.json" ]]; then
 if mkdir "$root/qualification.lock" 2>/dev/null; then
  trap 'rmdir "$root/qualification.lock"' EXIT
  export WANDB_MODE=disabled
  "$python" -m torch.distributed.run --standalone --nproc_per_node=4 -m samudra.observations.train "$root/qualification.yaml" --stop_after 4
  "$python" -m torch.distributed.run --standalone --nproc_per_node=4 -m samudra.observations.train "$root/qualification.yaml"
  "$python" - "$root" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
completion=json.loads((root/'outputs/llc-obs-six-hour-qualification/saved_nets/COMPLETE.json').read_text())
assert completion['updates']==8
for source in ('llc','obs'):
 assert completion['exposure'][source]==dict(updates=4,fresh=32,replay=0)
metrics=[json.loads(line) for line in (root/'outputs/llc-obs-six-hour-qualification/metrics.jsonl').read_text().splitlines()]
validation=[row for row in metrics if 'snapshot_val/obs/normalized_rmse/channel_mean' in row]
assert validation and validation[-1]['updates']==8
signature=json.loads((root/'qualification-signature.json').read_text())
(root/'QUALIFIED.json.tmp').write_text(json.dumps(dict(signature=signature,completion=completion,validation=validation[-1]),indent=2)+'\n')
(root/'QUALIFIED.json.tmp').replace(root/'QUALIFIED.json')
PY
  rmdir "$root/qualification.lock"
  trap - EXIT
 else
  # Avoid reserving a second allocation while the owner qualifies the experiment.
  echo 'Another allocation owns qualification; exit for monitor to resubmit if needed.' >&2
  exit 75
 fi
fi
"$python" - "$root" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
assert json.loads((root/'QUALIFIED.json').read_text())['signature']==json.loads((root/'qualification-signature.json').read_text())
PY
if [[ "$mode" == qualification ]]; then exit 0; fi
if [[ "$mode" != production ]]; then echo "Unknown mode: $mode" >&2; exit 2; fi
export WANDB_MODE=online
exec "$python" -m torch.distributed.run --standalone --nproc_per_node=4 -m samudra.observations.train "$root/train.yaml"
