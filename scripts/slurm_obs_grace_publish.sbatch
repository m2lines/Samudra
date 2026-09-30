#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# Submit with afterok on the successful preparation job; retain all local data.
#SBATCH --job-name=obs-grace-publish
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=grace
#SBATCH --qos=standard
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
set -euo pipefail
: "${WORK_ROOT:?Set the persistent observation work root}"
: "${REPO_DIR:?Set the unchanged processing checkout}"
: "${EXPECTED_COMMIT:?Set the processing revision used by preparation}"
: "${PUBLISH_SCRIPT:?Set the fixed portable publisher path}"
[[ "$(uname -m)" == aarch64 ]]
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]]
export OBS_PYTHON="$WORK_ROOT/envs/grace-arm64/bin/python"
export PATH="$WORK_ROOT/envs/grace-arm64/bin:$HOME/bin:$PATH"
export PYTHONPATH="$REPO_DIR/data${PYTHONPATH:+:$PYTHONPATH}"
export RCLONE_MODULE='' RCLONE_S3_NO_CHECK_BUCKET=true
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1
export DEST="${DEST:-nyu-osn:emulators/jr7309/data/full_range}"
RUN_LOG="$WORK_ROOT/logs/publish-$SLURM_JOB_ID"
mkdir -p "$RUN_LOG"
exec 8>"$WORK_ROOT/.grace-publish-all.lock"
flock -n 8 || { echo 'Another full publication job is active'; exit 2; }
# Acquire the processing lock too: no publication while preparation is active.
exec 7>"$WORK_ROOT/.grace-full-range.lock"
flock -n 7 || { echo 'Preparation is still active'; exit 2; }
printf 'Host=%s ProcessingCommit=%s Publisher=%s\n' "$(hostname)" "$EXPECTED_COMMIT" "$PUBLISH_SCRIPT"
sha256sum "$PUBLISH_SCRIPT"
rclone version
# Revalidate all products before writing any object to the final OSN prefix.
for product in duacs oisst argo-iap; do
    "$OBS_PYTHON" -m ocean_preprocessing.obs_preprocessing full_range validate \
        --manifest_path="$WORK_ROOT/manifests/$product.json" \
        --store="$WORK_ROOT/prepared/$product.zarr" > "$RUN_LOG/$product.validation.log" 2>&1
done
for product in duacs oisst argo-iap; do
    printf '%s Uploading %s\n' "$(date --iso-8601=seconds)" "$product"
    PRODUCT="$product" bash "$PUBLISH_SCRIPT" > "$RUN_LOG/$product.log" 2>&1
    printf '%s\tverified\n' "$product" >> "$RUN_LOG/completed.tsv"
done
printf 'All three stores published and read-back verified at %s; local raw and prepared data retained.\n' "$DEST"
