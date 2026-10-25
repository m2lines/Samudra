#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=gate-inputs-diagnostics
#SBATCH --partition=beta
#SBATCH --qos=standard
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
set -euo pipefail
export TOP=/projects/ny/lz1955/multiscale/jrusak/runs/native-gate-inputs-20261024
export BASE=/projects/ny/lz1955/multiscale/jrusak/runs/current-stochastic-depth-1deg-20261006
export DATA_ROOT=/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg
on_exit() {
  rc=$?; trap - EXIT
  python3 "$HOME/.config/codex-followup/send.py" --host beta --event-id "$(cat "$TOP/diagnostics-followup-id")" || true
  exit "$rc"
}
trap on_exit EXIT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
image=ghcr.io/m2lines/ocean-emulator-physicsnemo@sha256:7d78e8fb9592abdf046d88e55004f15cfe34f0c8ee38b17d83e98cc05b6f82c4
mounts="$TOP/source/src:/workspace/src,$BASE/native:/workspace/native,$TOP:$TOP,$DATA_ROOT:$DATA_ROOT"
srun --kill-on-bad-exit=1 --container-image="$image" --container-mount-home --container-mounts="$mounts" bash -c '
set -euo pipefail
export PYTHONPATH=/workspace/src:/workspace/native
py=/workspace/.venv/bin/python
pids=()
gpu=0
for kind in direct nonlinear; do
  root="$TOP/$kind"
  outputs=("$root"/output/native-gated-"$kind"-1deg-seed15-*)
  test "${#outputs[@]}" = 1
  for mode in raw ema; do
    ckpt=ckpt_70.pt
    if [ "$mode" = ema ]; then ckpt=ema_ckpt.pt; fi
    out="$root/diagnostics/$mode"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" "$py" "$TOP/diagnose-inputs.py" --root "$root" --data "$DATA_ROOT" --checkpoint "${outputs[0]}/saved_nets/$ckpt" --label "gated_${kind}_${mode}" --output "$out" > "$out/run.log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu+1))
  done
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
'
