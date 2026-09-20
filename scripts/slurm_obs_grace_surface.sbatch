#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=obs-grace-surface
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=grace
#SBATCH --qos=standard
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
set -euo pipefail
: "${WORK_ROOT:?}" "${REPO_DIR:?}" "${EXPECTED_COMMIT:?}" "${PRODUCT:?}" "${STAGE:?}"
[[ "$(uname -m)" == aarch64 ]]
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]]
case "$PRODUCT" in era5-surface|duacs-ssh) ;; *) exit 2 ;; esac
export OBS_PYTHON="$WORK_ROOT/envs/grace-arm64/bin/python"
export PATH="$WORK_ROOT/envs/grace-arm64/bin:$HOME/bin:$PATH"
export PYTHONPATH="$REPO_DIR/data"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1 NUMEXPR_NUM_THREADS=1
export RCLONE_MODULE='' RCLONE_S3_NO_CHECK_BUCKET=true
export DEST="${DEST:-nyu-osn:emulators/jr7309/data/full_range}"
manifest="$WORK_ROOT/manifests/$PRODUCT.json"
output="$WORK_ROOT/prepared"
[[ "$PRODUCT" != duacs-ssh ]] || output="$WORK_ROOT/surface-staging"
mkdir -p "$output" "$WORK_ROOT/logs/surface-$SLURM_JOB_ID"
exec 8>"$WORK_ROOT/.surface-$PRODUCT.lock"
flock -n 8 || { echo 'Another surface job owns this product'; exit 2; }
printf 'Host=%s Commit=%s Product=%s Stage=%s\n' "$(hostname)" "$EXPECTED_COMMIT" "$PRODUCT" "$STAGE"
cli=("$OBS_PYTHON" -m ocean_preprocessing.obs_preprocessing.surface)
case "$STAGE" in
    prepare)
        # A successful clean pause is continued by the next afterok segment.
        # A failed source/check never becomes an apparently successful pause.
        "${cli[@]}" run --manifest_path="$manifest" --output_root="$output" --workers=24 --max_seconds="${MAX_SECONDS:-36000}"
        ;;
    publish)
        "${cli[@]}" validate --manifest_path="$manifest" --store="$output/$PRODUCT.zarr"
        if [[ "$PRODUCT" == duacs-ssh ]]; then
            # Share locks with the original preparation/publication workflow.
            exec 7>"$WORK_ROOT/.grace-full-range.lock"
            flock -n 7
            exec 6>"$WORK_ROOT/.grace-publish-all.lock"
            flock -n 6
            "${cli[@]}" merge_duacs --manifest_path="$manifest" \
                --extension_store="$output/duacs-ssh.zarr" \
                --original_manifest="$WORK_ROOT/manifests/duacs.json" \
                --merged_manifest="$WORK_ROOT/manifests/duacs-with-ssh.json" \
                --store="$WORK_ROOT/prepared/duacs.zarr"
            export PREVIOUS_MANIFEST_PATH="$WORK_ROOT/manifests/duacs.json"
            export MANIFEST_PATH="$WORK_ROOT/manifests/duacs-with-ssh.json"
            export PRODUCT=duacs
        else
            export MANIFEST_PATH="$manifest"
        fi
        bash "$REPO_DIR/scripts/publish_obs_full_range.sh"
        printf 'Published and read-back verified %s; prepared store retained locally.\n' "$PRODUCT"
        ;;
    *) echo "Unsupported stage $STAGE" >&2; exit 2 ;;
esac
