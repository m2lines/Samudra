#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-v2-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
TASK_ROOT=/scratch/jr7309/deterministic-obs-v2
TASK_COMMIT=acecbf86c9032fb3e13c25886b9d6c1a24acbc40
TASK_SIF=/scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
trap 'scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
bash "$TASK_ROOT/build-code.sh" "$TASK_COMMIT" "$TASK_SIF"
TASK_LAYER=/scratch/jr7309/.apptainer-code-layers/samudra-code-${TASK_COMMIT}.img
run_python() {
 singularity exec --overlay "${TASK_LAYER}:ro" --bind /scratch/jr7309:/scratch/jr7309 --pwd /opt/samudra-code "$TASK_SIF" /usr/bin/env PYTHONPATH=/opt/samudra-code/src PYTHONUNBUFFERED=1 /workspace/.venv/bin/python "$@"
}
run_python "$TASK_ROOT/frozen-data/prepare_observation_normalization.py" /scratch/jr7309/data/obs-d-pilot --output "$TASK_ROOT/normalization"
for attempt in $(seq 1 90); do
 if [[ -f "$TASK_ROOT/stage-exit-code" ]]; then break; fi
 sleep 60
done
test "$(cat "$TASK_ROOT/stage-exit-code")" = 0
run_python "$TASK_ROOT/frozen-data/prepare_observation_era5.py" /scratch/jr7309/data/obs-d-pilot "$TASK_ROOT/daily" /scratch/jr7309/data/om4_onedeg_v3 "$TASK_ROOT/era5-cache"
run_python - <<'PY'
import json
from pathlib import Path
from samudra.observations.config import ObservationTrainConfig
from samudra.observations.campaign import screen_configs
from samudra.observations.shared_era5 import digest
root=Path('/scratch/jr7309/deterministic-obs-v2')
base=ObservationTrainConfig.from_yaml(Path('/opt/samudra-code/src/samudra/configs/observations/deterministic.yaml'))
values=base.model_dump(mode='json')
values['data']=json.loads((root/'data-paths.json').read_text())
values['data']['era5_proof_sha256']=digest(root/'era5-cache/COMPLETE.json')
values['experiment'].update(base_output_dir=str(root/'qualification'),data_root='/scratch/jr7309')
values['experiment']['wandb'].update(mode='online',group='deterministic-obs-v2-qualification',tags=['qualification','deterministic','samudra-v2'])
base=ObservationTrainConfig.model_validate(values)
for qualification,directory in [(True,'qualification-configs'),(False,'screen-configs')]:
 folder=root/directory;folder.mkdir(exist_ok=True)
 for name,cfg in screen_configs(base,qualification=qualification).items():
  if not qualification:
   cfg.experiment.base_output_dir=str(root/'screen')
  cfg.save_yaml(folder/(name+'.yaml'))
(root/'PREPARED.json').write_text(json.dumps({'code_commit':'acecbf86c9032fb3e13c25886b9d6c1a24acbc40','era5_proof_sha256':values['data']['era5_proof_sha256'],'normalization_sha256':digest(root/'normalization/normalization.json')},indent=2)+'\n')
PY
