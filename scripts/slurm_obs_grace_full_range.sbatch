#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# Persistent native-cadence preparation only; OSN publication is a separate step.
#SBATCH --job-name=obs-grace-full-range
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=grace
#SBATCH --qos=standard
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to the fixed processing checkout}"
: "${WORK_ROOT:?Set WORK_ROOT to the persistent Empire AI data directory}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the reviewed processing revision}"
[[ "$(uname -m)" == aarch64 ]] || { echo 'Requires an ARM64 compute node'; exit 2; }
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo 'Checkout revision changed'; exit 2; }
[[ -z "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]] || { echo 'Checkout has uncommitted changes'; exit 2; }
OBS_ENV="${WORK_ROOT}/envs/grace-arm64"
INVENTORY_DIR="${WORK_ROOT}/manifests"
RUN_LOG="${WORK_ROOT}/logs/grace-${SLURM_JOB_ID}"
export PATH="$OBS_ENV/bin:$HOME/bin:$PATH"
export PYTHONPATH="${REPO_DIR}/data${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="${WORK_ROOT}/tmp/grace-arm64"
mkdir -p "$RUN_LOG" "$TMPDIR"
exec 9>"${WORK_ROOT}/.grace-full-range.lock"
flock -n 9 || { echo 'Another Grace full-range job is active'; exit 2; }
cd "$REPO_DIR/data"
printf 'Host=%s Architecture=%s Commit=%s\n' "$(hostname)" "$(uname -m)" "$EXPECTED_COMMIT"
# Reuse the validated native environment without modifying it during processing.
"$OBS_ENV/bin/python" -m ocean_preprocessing.obs_preprocessing.check
"${WORK_ROOT}/tools/grace-arm64/uv" pip freeze --python "$OBS_ENV/bin/python" > "$RUN_LOG/environment-freeze.txt"
for product in duacs oisst argo-iap; do
    test -s "$INVENTORY_DIR/$product.json"
done
sha256sum "$INVENTORY_DIR/duacs.json" "$INVENTORY_DIR/oisst.json" "$INVENTORY_DIR/argo-iap.json" > "$RUN_LOG/inventories.sha256"

run_product() (
    product="$1"
    threads="$2"
    stage=starting
    # Preserve each product's last stage and exit code even if a sibling fails.
    trap 'rc=$?; printf "%s\t%s\t%s\n" "$product" "$stage" "$rc" > "$RUN_LOG/$product.status.tmp"; mv "$RUN_LOG/$product.status.tmp" "$RUN_LOG/$product.status"' EXIT
    for stage in download prepare validate; do
        printf '%s Product=%s Stage=%s\n' "$(date --iso-8601=seconds)" "$product" "$stage"
        printf '%s\t%s\trunning\n' "$product" "$stage" > "$RUN_LOG/$product.status"
        args=("$stage" --manifest_path="$INVENTORY_DIR/$product.json")
        case "$stage" in
            download) args+=(--raw_root="$WORK_ROOT/raw" --workers=4) ;;
            prepare) args+=(--raw_root="$WORK_ROOT/raw" --output_root="$WORK_ROOT/prepared" --threads="$threads") ;;
            validate) args+=(--store="$WORK_ROOT/prepared/$product.zarr") ;;
        esac
        /usr/bin/time -v "$OBS_ENV/bin/python" -m ocean_preprocessing.obs_preprocessing full_range "${args[@]}"
    done
    stage=complete
)

pids=()
products=(duacs oisst argo-iap)
threads=(16 8 8)
for index in "${!products[@]}"; do
    product="${products[$index]}"
    run_product "$product" "${threads[$index]}" > "$RUN_LOG/$product.log" 2>&1 &
    pids+=("$!")
    printf 'Started %s pipeline, pid=%s log=%s/%s.log\n' "$product" "$!" "$RUN_LOG" "$product"
done
failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'Validated local store: %s/prepared/%s.zarr\n' "$WORK_ROOT" "${products[$index]}"
    else
        rc=$?
        printf 'FAILED product=%s exit=%s; inspect %s/%s.log\n' "${products[$index]}" "$rc" "$RUN_LOG" "${products[$index]}" >&2
        failed=1
    fi
done
[[ "$failed" == 0 ]] || exit 1
printf 'All three local stores validated. Raw downloads and prepared stores retained. OSN publication remains separate.\n'
