#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=${LLC_OBS_RUN_ROOT:?Set continuation run directory}
mode=${1:?qualification or production}
[[ "$mode" == qualification || "$mode" == production ]]
python=/workspace/.venv/bin/python
export PYTHONPATH=/llc-code/src PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /llc-code
"$python" - "$root" <<'PY'
import copy,hashlib,json,os,pathlib,sys,torch,yaml
root=pathlib.Path(sys.argv[1]); config=yaml.safe_load((root/'train.yaml').read_text()); qual=yaml.safe_load((root/'qualification.yaml').read_text())
assert torch.cuda.is_available() and torch.cuda.device_count()==4
assert config['main_updates']==10000 and config['post_training_updates']==1000
expected=copy.deepcopy(config);expected.update(main_updates=2004,post_training_updates=4,save_freq=4,validation_freq=4,validation_examples=2,milestone_updates=[])
expected['experiment']['name']=config['experiment']['name']+'-qualification';expected['experiment']['wandb']['mode']='disabled'
assert qual==expected, 'Qualification differs beyond permitted changes'
proof=pathlib.Path(config['experiment']['data_root'])/config['data']['snapshots']/'COMPLETE.json'
digest=hashlib.sha256();code=pathlib.Path('/llc-code')
for path in sorted((code/'src').rglob('*.py'))+[code/'experiments/llc-observation-six-hour'/n for n in ('run_long.sh','extend_checkpoint.py')]:
 digest.update(str(path.relative_to(code)).encode());digest.update(path.read_bytes())
signature={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {'archive':root/'code.tar.gz','train':root/'train.yaml','qualification':root/'qualification.yaml','parent':root/'parent.json','data':proof}.items()}
signature.update(code=digest.hexdigest(),world_size=4,torch=str(torch.__version__),cuda=torch.version.cuda)
(root/f"signature-{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(signature,sort_keys=True)+'\n')
print(json.dumps(dict(signature=signature,devices=[torch.cuda.get_device_name(i) for i in range(4)])),flush=True)
PY
migrate() {
 "$python" - "$root" "$1" <<'PY'
import json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);stage=sys.argv[2];p=json.loads((root/'parent.json').read_text())
cmd=[sys.executable,'experiments/llc-observation-six-hour/extend_checkpoint.py','--parent',p['checkpoint'],'--parent-sha256',p['sha256'],'--parent-config-source',p['config_source'],'--config',str(root/('qualification.yaml' if stage=='qualification' else 'train.yaml'))]
if stage=='qualification':cmd+=['--qualification']
subprocess.run(cmd,check=True)
PY
}
if [[ ! -f "$root/QUALIFIED.json" ]]; then
 if ! mkdir "$root/qualification.lock" 2>/dev/null; then exit 75; fi
 printf '%s\n' "$SLURM_JOB_ID" > "$root/qualification.lock/job_id"
 trap 'rm -f "$root/qualification.lock/job_id"; rmdir "$root/qualification.lock"' EXIT
 migrate qualification
 progress=$("$python" - "$root" <<'PY'
import pathlib,sys,torch,yaml
root=pathlib.Path(sys.argv[1]);c=yaml.safe_load((root/'qualification.yaml').read_text());p=pathlib.Path(c['experiment']['base_output_dir'])/c['experiment']['name']/'saved_nets/ckpt.pt'
print(torch.load(p,map_location='cpu',weights_only=False)['updates'])
PY
)
 export WANDB_MODE=disabled
 if (( progress < 2004 )); then
  "$python" -m torch.distributed.run --standalone --nproc_per_node=4 -m samudra.observations.train "$root/qualification.yaml" --stop_after 2004
 fi
 "$python" -m torch.distributed.run --standalone --nproc_per_node=4 -m samudra.observations.train "$root/qualification.yaml"
 "$python" - "$root" <<'PY'
import json,math,os,pathlib,sys,torch,yaml
root=pathlib.Path(sys.argv[1]);c=yaml.safe_load((root/'qualification.yaml').read_text());out=pathlib.Path(c['experiment']['base_output_dir'])/c['experiment']['name']
receipt=json.loads((out/'saved_nets/COMPLETE.json').read_text());assert receipt['updates']==2008
llc=1002 if c['pretrain_mix']=='llc-obs' else 0
for source,count in [('llc',llc),('obs',2008-llc)]: assert receipt['exposure'][source]==dict(updates=count,fresh=count*8,replay=0)
rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
val=[r for r in rows if 'snapshot_val/obs/normalized_rmse/channel_mean' in r];assert val[-1]['updates']==2008 and all(math.isfinite(v) for r in val for v in r.values())
saved=torch.load(out/'saved_nets/ckpt.pt',map_location='cpu',weights_only=False);assert saved['in_post_training'] and len(saved['rank_rng'])==4
assert abs(saved['optimizer']['param_groups'][0]['lr']-1e-6)<1e-12
signature=json.loads((root/f"signature-{os.environ['SLURM_JOB_ID']}.json").read_text())
(root/'QUALIFIED.json.tmp').write_text(json.dumps(dict(signature=signature,completion=receipt,validation=val[-1]),indent=2)+'\n');(root/'QUALIFIED.json.tmp').replace(root/'QUALIFIED.json')
PY
 rm "$root/qualification.lock/job_id"
 rmdir "$root/qualification.lock"
 trap - EXIT
fi
"$python" - "$root" <<'PY'
import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]);assert json.loads((root/'QUALIFIED.json').read_text())['signature']==json.loads((root/f"signature-{os.environ['SLURM_JOB_ID']}.json").read_text())
PY
if [[ "$mode" == qualification ]]; then exit 0; fi
migrate production
export WANDB_MODE=online
exec "$python" -m torch.distributed.run --standalone --nproc_per_node=4 -m samudra.observations.train "$root/train.yaml"
