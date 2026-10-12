#!/bin/bash -l
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=diffusion-scratch
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=beta
#SBATCH --qos=long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --segment=2
#SBATCH --time=7-00:00:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}"
: "${CODE_ID:?}"
: "${MODE:?qualify, train, or evaluate}"
code="$CAMPAIGN_ROOT/code/$CODE_ID"
export SAMUDRA_CODE_COMMIT="$CODE_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TORCHINDUCTOR_COMPILE_THREADS=2
export NCCL_MNNVL_ENABLE=1 NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=bond0
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29614
export NCCL_DEBUG_FILE="$CAMPAIGN_ROOT/logs/nccl-$SLURM_JOB_ID-%h-%p.log"
scontrol show job "$SLURM_JOB_ID" > "$CAMPAIGN_ROOT/logs/allocation-$SLURM_JOB_ID.txt"
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 # pragma: allowlist secret
container=(--container-image="$image" --container-mount-home --container-mounts="$code:/pilot-code,$CAMPAIGN_ROOT/native:/pilot-native,/projects/ny/lz1955/multiscale:/projects/ny/lz1955/multiscale")
execute() {
  srun --kill-on-bad-exit=1 "${container[@]}" env CC=/usr/bin/gcc CXX=/usr/bin/g++ PYTHONPATH=/pilot-code/src:/pilot-native "$@"
}
if [[ "$MODE" == qualify ]]; then
  for ARM in direct residual; do
    export ARM
    execute bash /pilot-code/scripts/diffusion_scratch_beta_rank.sh --smoke --stop-at 16
    execute bash /pilot-code/scripts/diffusion_scratch_beta_rank.sh --smoke --resume --stop-at 32
  done
  execute bash -c 'if [[ "$SLURM_NODEID" == 0 ]]; then /workspace/.venv/bin/python /pilot-code/scripts/qualify_diffusion_scratch_distributed.py --root "$CAMPAIGN_ROOT" --code-id "$CODE_ID"; fi'
elif [[ "$MODE" == train ]]; then
  : "${ARM:?direct or residual}"
  python3 - "$CAMPAIGN_ROOT/QUALIFIED.json" "$CODE_ID" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
assert receipt['qualified'] and receipt['producer'] == sys.argv[2], receipt
PY
  extra=()
  if [[ -f "$CAMPAIGN_ROOT/runs/$ARM/last.pt" ]]; then extra+=(--resume); fi
  execute bash /pilot-code/scripts/diffusion_scratch_beta_rank.sh --invocation-hours 166 "${extra[@]}"
elif [[ "$MODE" == evaluate ]]; then
  : "${ARM:?}"
  : "${CHECKPOINT_STEP:?}"
  : "${OM4_UPDATES:?}"
  : "${OBS_UPDATES:?}"
  execute /workspace/.venv/bin/python -m samudra.experiments.diffusion_scratch_evaluate \
    --root "$CAMPAIGN_ROOT" --checkpoint "$CAMPAIGN_ROOT/runs/$ARM/step-$CHECKPOINT_STEP.pt" \
    --output "$CAMPAIGN_ROOT/evaluation/$ARM-obs$OBS_UPDATES" \
    --processor-prediction "$ARM" --expected-om4-updates "$OM4_UPDATES" \
    --expected-observation-updates "$OBS_UPDATES" --gpus 0,1,2
else
  echo "Unknown MODE=$MODE" >&2
  exit 2
fi
