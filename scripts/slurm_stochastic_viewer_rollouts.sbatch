#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=stochastic-viewer-rollouts
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=rtx6000_lzanna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?}" "${CAMPAIGN_IMAGE:?}" "${VIEWER_EXPORT_SCRIPT:?}" "${VIEWER_EXPORT_OUTPUT:?}" "${CODEX_FOLLOWUP_EVENT:?}"
finish() {
  code=$?
  trap - EXIT
  python3 "$HOME/.config/codex-followup/send.py" --host torch --event-id "$CODEX_FOLLOWUP_EVENT" || true
  exit "$code"
}
trap finish EXIT
python3 "$HOME/.config/codex-followup/send.py" --host torch --event-id "$CODEX_FOLLOWUP_EVENT" --check
export APPTAINER_CACHEDIR=/scratch/jr7309/apptainer-cache
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
args=()
if [[ -n "${VIEWER_CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "$VIEWER_CHECKPOINT" --expected-arm "${VIEWER_EXPECTED_ARM:?}" \
    --expected-step "${VIEWER_EXPECTED_STEP:?}" --expected-contract "${VIEWER_EXPECTED_CONTRACT:?}" \
    --evaluation-root "${VIEWER_EVALUATION_ROOT:?}" --variant "${VIEWER_VARIANT:?}")
fi
if [[ -n "${VIEWER_OM4_INITIALIZER_REFERENCE:-}" ]]; then
  args+=(--export-om4-initialization --om4-initializer-reference "$VIEWER_OM4_INITIALIZER_REFERENCE" \
    --om4-initializer-variant "${VIEWER_VARIANT:?}")
fi
/share/apps/apptainer/bin/apptainer exec --nv \
 --bind "$CAMPAIGN_ROOT/code:/pilot-code:ro,/scratch/jr7309:/scratch/jr7309,$VIEWER_EXPORT_SCRIPT:/viewer-export.py:ro" \
 "$CAMPAIGN_IMAGE" /usr/bin/env \
 PYTHONPATH=/pilot-code/src PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 TORCH_CUDNN_V8_API_DISABLED=1 WANDB_MODE=disabled \
 /workspace/.venv/bin/python /pilot-code/scripts/run_without_thp.py /viewer-export.py \
 --root "$CAMPAIGN_ROOT" --data /scratch/jr7309/data/obs-d-pilot --output "$VIEWER_EXPORT_OUTPUT" \
 --om4 /scratch/jr7309/data/om4_onedeg_v3/OM4.zarr "${args[@]}"
