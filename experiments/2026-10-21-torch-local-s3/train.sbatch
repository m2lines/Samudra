#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=quarter-local-vs-s3
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=rtx6000_lzanna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=03:30:00
#SBATCH --chdir=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021
#SBATCH --output=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/train-%j.out
#SBATCH --error=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/train-%j.err
set -euo pipefail
base=$PWD
test -f "$base/PREFLIGHT_COMPLETE.json"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 WANDB_MODE=disabled
export PYTHONPATH="$base/extension:$base/repo/src:$base/repo"
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$base/runs-$SLURM_JOB_ID"
hostname
nvidia-smi -L
telemetry_pid=''
trap 'if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; fi' EXIT
for arm in local remote; do
  echo "START_ARM $arm $(date -Is)"
  nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 5 > "$base/runs-$SLURM_JOB_ID/gpu-$arm.csv" &
  telemetry_pid=$!
  srun --ntasks=1 apptainer exec --nv --bind "$HOME/.config/rclone/rclone.conf:/opt/osn-rclone.conf:ro" \
    /scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif \
    /workspace/.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$base/benchmark.py" --arm "$arm" --seconds 3600 --warmup 5
  kill "$telemetry_pid"
  wait "$telemetry_pid" || true
  telemetry_pid=''
  echo "END_ARM $arm $(date -Is)"
done
python3 "$base/summarize.py" "$base/runs-$SLURM_JOB_ID"
