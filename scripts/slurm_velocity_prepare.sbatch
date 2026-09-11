#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Stream two velocity channels using the same immutable code layer as training.
# Required: INPUT_PATH, OUTPUT_ROOT, SOURCE_NAME, SOURCE_KIND, CODE_LAYER, SIF_PATH.
# Submit with scratch-backed --chdir, --output and --error paths.
#SBATCH --job-name=velocity-prepare
#SBATCH --account=torch_pr_347_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
: "${INPUT_PATH:?required}"
: "${OUTPUT_ROOT:?required}"
: "${SOURCE_NAME:?required}"
: "${SOURCE_KIND:?required}"
: "${CODE_LAYER:?required}"
: "${SIF_PATH:?required}"
[[ -d "${INPUT_PATH}" && -d "${OUTPUT_ROOT}" && -r "${SIF_PATH}" ]]
[[ "${SOURCE_NAME}" =~ ^[a-z0-9_]+$ ]]
[[ "${SOURCE_KIND}" == "duacs" || "${SOURCE_KIND}" == "om4" ]]
[[ ! -e "${OUTPUT_ROOT}/${SOURCE_NAME}" ]]
CODE_LAYER="$(realpath "${CODE_LAYER}")"
(
  cd "$(dirname "${CODE_LAYER}")"
  sha256sum --check "$(basename "${CODE_LAYER}").sha256"
)
export APPTAINER_CACHEDIR="/scratch/${USER}/apptainer-cache"
export SINGULARITY_CACHEDIR="/scratch/${USER}/singularity-cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
apptainer exec --overlay "${CODE_LAYER}:ro" \
  --bind "${INPUT_PATH}:${INPUT_PATH}:ro,${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  --pwd /opt/samudra-code "${SIF_PATH}" bash -c '
    set -euo pipefail
    cmp -s /workspace/uv.lock /opt/samudra-code/uv.lock
    cmp -s /workspace/pyproject.toml /opt/samudra-code/pyproject.toml
    export PYTHONPATH=/opt/samudra-code/src
    /workspace/.venv/bin/python -m samudra.experiments.velocity_transfer.prepare \
      --input "$1" --output "$2" --kind "$3" --batch-times "$4"
    cp /opt/samudra-code/source-manifest.json "$2/source-manifest.json"
  ' _ "${INPUT_PATH}" "${OUTPUT_ROOT}/${SOURCE_NAME}" "${SOURCE_KIND}" "${BATCH_TIMES:-4}"
