#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-D-beta-pilot
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
: "${CODE_COMMIT:?Set the immutable production source commit}"
root=/mnt/home/jrusak/data/obs_full_range/d-observation-pilot
code="$root/code/beta-source-$CODE_COMMIT"
test -f "$root/beta-run/source-contract/QUALIFIED.json"
test -f "$code/src/samudra/experiments/observation_allocation.py"
(cd "$root/samples" && sha256sum --check --status SHA256SUMS)
"/mnt/home/jrusak/data/obs_full_range/envs/grace-arm64/bin/python" - "$root" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); data=root/'samples'
proof=json.loads((root/'PUBLISHED.json').read_text())
assert hashlib.sha256((data/'SHA256SUMS').read_bytes()).hexdigest()==proof['sha256sums_sha256']
for split,n in [('train',243),('validation',9),('test',96)]:
    assert len(list((data/split).glob('*.npz')))==n
record={'ready_utc':datetime.datetime.now(datetime.UTC).isoformat(),
        'verification':'all 350 NPZ files checked against source SHA256SUMS',
        'source_publication':proof,'verified_location':str(data)}
temp=data/'DATA_READY.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(data/'DATA_READY.json')
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export SAMUDRA_CODE_COMMIT="$CODE_COMMIT"
export WANDB_MODE=online
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
srun --nodes=1 --ntasks=1 --container-image="$image" --container-mount-home \
  --container-mounts="$code:/pilot-code,$root:$root" \
  env PYTHONPATH=/pilot-code/src \
  /workspace/.venv/bin/python -m samudra.experiments.observation_allocation \
  --root="$root/beta-run/pilot-v1" --data="$root/samples" \
  --checkpoint="$root/beta-run/D-source.pt" \
  --source-contract="$root/beta-run/source-contract/QUALIFIED.json"
