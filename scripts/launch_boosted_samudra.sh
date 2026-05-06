#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Submit the PCGB experiment chain on the HPC system.
#
# Phase 1: PCGB cold-start training (multi-GPU DDP via torchrun).
# Phase 2: Eval the resulting checkpoint on the test period (single GPU,
#          chained behind Phase 1 via SLURM dependency).
#
# Usage (from torch login node, in repo root):
#   export CONTAINER_TAG=25.11-latest      # or CONTAINER_HASH=<sha>
#   export WANDB_API_KEY=<key>             # optional but recommended
#   bash scripts/launch_boosted_samudra.sh
#
# Optional knobs:
#   export NAME_PREFIX="$(date +%Y-%m-%d)" # default: today's date
#   export SKIP_EVAL=1                     # skip the chained eval job
#   export EXTRA_ARGS_PCGB="--num_rounds=8" # forwarded to PCGB (CLI override)
#   export EXTRA_ARGS_EVAL=""              # forwarded to eval
#
# PCGB defaults: 1 node × 8 GPUs × 12 h. Eval defaults: 1 GPU × 2 h.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

NAME_PREFIX="${NAME_PREFIX:-$(date +%Y-%m-%d)}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EXTRA_ARGS_PCGB="${EXTRA_ARGS_PCGB:-}"
EXTRA_ARGS_EVAL="${EXTRA_ARGS_EVAL:-}"
SBATCH_ARGS_PCGB="${SBATCH_ARGS_PCGB:-}"
SBATCH_ARGS_EVAL="${SBATCH_ARGS_EVAL:-}"

PCGB_NAME="${NAME_PREFIX}-pcgb_samudra"
EVAL_NAME="${NAME_PREFIX}-pcgb_samudra_eval"

CURRENT_USER="${USER:-$(id -un)}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/${CURRENT_USER}/runs}"
PCGB_CKPT_PATH="${OUTPUT_BASE}/${PCGB_NAME}/saved_nets/pcgb_final.pt"

echo "── Submitting PCGB cold-start training..."
PCGB_OUT=$(
  NAME="${PCGB_NAME}" \
  CONFIG="configs/samudra_om4_v2/boosted_pcgb.yaml" \
  ARGS="${EXTRA_ARGS_PCGB}" \
  sbatch ${SBATCH_ARGS_PCGB} scripts/slurm_apptainer_pcgb.sbatch
)
echo "${PCGB_OUT}"
PCGB_JOBID=$(awk '/^Submitted batch job/ {print $4}' <<<"${PCGB_OUT}" | tail -n1)
if [[ -z "${PCGB_JOBID}" ]]; then
  echo "ERROR: failed to parse PCGB jobid." >&2
  exit 1
fi
echo "── PCGB jobid: ${PCGB_JOBID}"

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "── Eval skipped (SKIP_EVAL=1)."
  exit 0
fi

echo "── Submitting eval, queued after PCGB completes..."
EVAL_OUT=$(
  NAME="${EVAL_NAME}" \
  CONFIG="configs/samudra_om4_v2/boosted_eval.yaml" \
  ARGS="--ckpt_path=${PCGB_CKPT_PATH} ${EXTRA_ARGS_EVAL}" \
  sbatch --dependency=afterok:${PCGB_JOBID} ${SBATCH_ARGS_EVAL} \
    scripts/slurm_apptainer_eval.sbatch
)
echo "${EVAL_OUT}"
EVAL_JOBID=$(awk '/^Submitted batch job/ {print $4}' <<<"${EVAL_OUT}" | tail -n1)
if [[ -z "${EVAL_JOBID}" ]]; then
  echo "ERROR: failed to parse eval jobid." >&2
  exit 1
fi
echo "── Eval jobid: ${EVAL_JOBID} (waits on ${PCGB_JOBID})"

echo ""
echo "Submitted. Inspect:  squeue -u \$USER"
echo "Cancel PCGB:  scancel ${PCGB_JOBID}"
echo "Cancel eval:  scancel ${EVAL_JOBID}"
echo "PCGB ckpt will land at: ${PCGB_CKPT_PATH}"
