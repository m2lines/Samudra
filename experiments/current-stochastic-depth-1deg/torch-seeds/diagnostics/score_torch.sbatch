#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=06:00:00
set -euo pipefail
if command -v apptainer >/dev/null 2>&1; then
  runtime=$(command -v apptainer)
elif test -x /share/apps/apptainer/bin/apptainer; then
  runtime=/share/apps/apptainer/bin/apptainer
else
  if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
  module load singularity-ce/4.5.1
  runtime=$(command -v singularity)
fi
root=/scratch/jr7309/runs/current-sd-seeds16-17
layer=/scratch/jr7309/.apptainer-code-layers/samudra-code-c3df51b5e0a8e497201377b27423eaf91d712c3e.img
(cd "$(dirname "$layer")" && sha256sum --check --status "$(basename "$layer").sha256")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH=/opt/samudra-code/src
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
srun "$runtime" exec --overlay "$layer:ro" --bind /scratch:/scratch \
  /scratch/jr7309/.apptainer-images/sd-seeds-base-0b5320248.sif \
  /workspace/.venv/bin/python "$root/observations/score_torch.py" "$@"
