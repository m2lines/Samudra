#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# Small pilot only: never launches the full-range download or cancels Torch jobs.
#SBATCH --job-name=obs-grace-pilot
#SBATCH --account=ny_lz1955_multiscale
#SBATCH --partition=grace
#SBATCH --qos=test
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to the fixed ARM pilot checkout}"
: "${WORK_ROOT:?Set WORK_ROOT to the persistent Empire AI data directory}"
: "${INVENTORY_DIR:?Set INVENTORY_DIR to the frozen full-range inventory snapshots}"
[[ "$(uname -m)" == aarch64 ]] || { echo 'Pilot requires an ARM64 compute node'; exit 2; }
OBS_ENV="${WORK_ROOT}/envs/grace-arm64"
UV_BIN="${WORK_ROOT}/tools/grace-arm64/uv"
PILOT_ROOT="${PILOT_ROOT:-${WORK_ROOT}/pilot/20260918}"
PILOT_PRODUCTS="${PILOT_PRODUCTS:-duacs oisst argo-iap}"
export PIP_CACHE_DIR="${WORK_ROOT}/cache/grace-arm64"
export UV_CACHE_DIR="${WORK_ROOT}/cache/grace-arm64/uv"
export TMPDIR="${WORK_ROOT}/tmp/grace-arm64"
export PYTHONPATH="${REPO_DIR}/data${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1
mkdir -p "$PILOT_ROOT" "$TMPDIR" "$PIP_CACHE_DIR"
exec 9>"${WORK_ROOT}/.grace-pilot.lock"
flock -n 9 || { echo 'Another Grace pilot is active'; exit 2; }
cd "$REPO_DIR/data"
printf 'Host=%s Architecture=%s Commit=%s\n' "$(hostname)" "$(uname -m)" "$(git -C "$REPO_DIR" rev-parse HEAD)"
"$UV_BIN" --version
# Grace's system Python lacks ensurepip; uv does not require python3-venv.
"$UV_BIN" venv --python python3 --allow-existing "$OBS_ENV"
export PATH="$OBS_ENV/bin:$HOME/bin:$PATH"
# numcodecs 0.15.1 has no matching ARM64 wheel; pip builds that pin natively.
"$UV_BIN" pip install --python "$OBS_ENV/bin/python" -r "$REPO_DIR/scripts/requirements-observations.txt"
"$UV_BIN" pip freeze --python "$OBS_ENV/bin/python" > "$PILOT_ROOT/environment-freeze.txt"
python -m ocean_preprocessing.obs_preprocessing.check

# A tiny retained probe outside the final dataset prefix verifies OSN read/write.
rclone lsf nyu-osn:emulators/jr7309/data/ --max-depth 1 > "$PILOT_ROOT/osn-listing.txt"
probe="$PILOT_ROOT/osn-probe-${SLURM_JOB_ID}.txt"
printf 'Grace observation pilot %s on %s\n' "$SLURM_JOB_ID" "$(hostname)" > "$probe"
remote="nyu-osn:emulators/jr7309/data/full_range_pilot_checks/grace-${SLURM_JOB_ID}.txt"
# The existing OSN bucket can permit object access without bucket creation.
rclone copyto --s3-no-check-bucket "$probe" "$remote"
rclone cat "$remote" > "$probe.readback"
cmp "$probe" "$probe.readback"
printf 'OSN probe verified: %s\n' "$remote"

for product in $PILOT_PRODUCTS; do
    /usr/bin/time -v python -m ocean_preprocessing.obs_preprocessing.pilot \
        --inventory-dir "$INVENTORY_DIR" --root "$PILOT_ROOT" \
        --product "$product" --threads "${SLURM_CPUS_PER_TASK:-4}"
done
printf 'Selected native-cadence pilots passed (%s): %s\n' "$PILOT_PRODUCTS" "$PILOT_ROOT"
