#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
set -euo pipefail
export KE_ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/current-ke-aux-20261010
export KE_DATA=/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09
export CONTROL_ROOT=/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006
export EVAL_ROOT="$KE_ROOT/evaluations-$SLURM_JOB_ID"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
mkdir -p "$EVAL_ROOT"
cp "$KE_ROOT/eval-commit.txt" "$EVAL_ROOT/commit.txt"
srun --kill-on-bad-exit=1 --container-image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4 --container-mount-home --container-mounts="$KE_ROOT/eval-source/src:/workspace/src,$KE_ROOT/native:/workspace/native,$KE_ROOT:$KE_ROOT,$KE_DATA:$KE_DATA,$CONTROL_ROOT:$CONTROL_ROOT,$KE_ROOT/cache:/workspace/.data_cache" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
run_one() {
    local gpu="$1" arm="$2" kind="$3" smoke="${4:-}"
    local checkpoint flag=""
    case "$arm" in
        control) checkpoint="$CONTROL_ROOT/control/output/current-1deg-control-ga2-210664/saved_nets"; flag="--control" ;;
        aligned) checkpoint="$KE_ROOT/aligned/output/current-ke-aligned-seed15-211905/saved_nets" ;;
        seasonal) checkpoint="$KE_ROOT/seasonal/output/current-ke-seasonal-seed15-211906/saved_nets" ;;
    esac
    if [ "$kind" = ema ]; then checkpoint="$checkpoint/ema_ckpt.pt"; else checkpoint="$checkpoint/ckpt_70.pt"; fi
    CUDA_VISIBLE_DEVICES="$gpu" /workspace/.venv/bin/python "$KE_ROOT/eval-source/experiments/current-ke-aux/evaluate.py" --config "$KE_ROOT/eval-source/experiments/current-ke-aux/aligned-eval.yaml" --checkpoint "$checkpoint" --output "$EVAL_ROOT/$arm-$kind${smoke:+-smoke}" $flag $smoke > "$EVAL_ROOT/$arm-$kind${smoke:+-smoke}.log" 2>&1
}
run_one 0 aligned ema --smoke
run_one 0 control ema --smoke
run_one 0 control ema & p0=$!
(run_one 1 aligned ema; run_one 1 aligned raw) & p1=$!
(run_one 2 seasonal ema; run_one 2 seasonal raw) & p2=$!
run_one 3 control raw & p3=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
exit "$failed"
'
